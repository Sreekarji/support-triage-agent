"""
agent.py -- LLM-powered support triage agent.

Ties together the rule-based classifier, TF-IDF retriever, and
Groq API (Llama 3) to process each support ticket end-to-end.
"""

import os
import re
import json
import time
from typing import Dict, Optional

from dotenv import load_dotenv

# Load .env from repo root (two levels up from code/)
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from classifier import classify
from retriever import Retriever, SearchResult


# ======================================================================
# Singleton retriever (expensive to init -- only build once)
# ======================================================================

_retriever: Optional[Retriever] = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


# ======================================================================
# Groq client (lazy init)
# ======================================================================

_client = None


def _get_client():
    """Return a cached Groq client. Reads GROQ_API_KEY from env."""
    global _client
    if _client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. "
                "Add it to your .env file or export it as an env var."
            )
        _client = Groq(api_key=api_key)
    return _client


MODEL = "llama-3.1-8b-instant"


# ======================================================================
# System prompt
# ======================================================================

SYSTEM_PROMPT = """You are a support triage agent for three product ecosystems:
HackerRank (developer hiring & assessment platform),
Claude (Anthropic's AI assistant), and
Visa (payment card & financial services).

## Your rules -- follow them exactly

1. Answer ONLY using the support corpus passages provided below.
2. NEVER invent, fabricate, or hallucinate policies, URLs, steps,
   phone numbers, or procedures that do not appear in the passages.
3. If the passages do NOT contain enough information to answer safely,
   set status to "escalated" and explain why in the justification.
4. Be concise, professional, and helpful.
5. When possible, include specific links or references from the passages.

## Output format

Return ONLY a valid JSON object with exactly these keys:
{
  "status": "replied" or "escalated",
  "product_area": "<the most relevant support category or domain area>",
  "response": "<user-facing answer grounded in the passages>",
  "justification": "<concise explanation of your routing/answering decision>",
  "request_type": "<one of: product_issue, feature_request, bug, invalid>"
}

Do NOT wrap the JSON in markdown code fences. Return raw JSON only."""


# ======================================================================
# Public API
# ======================================================================

def process_ticket(
    issue: str,
    subject: str = "",
    company: str = "",
) -> Dict[str, str]:
    """
    Process a single support ticket end-to-end.

    Returns dict with keys: status, product_area, response,
    justification, request_type.
    """

    # ---- 1. Rule-based classification ----
    cls = classify(issue, subject, company)
    domain = cls["domain"]
    request_type = cls["request_type"]
    force_escalate = cls["force_escalate"]
    escalate_reason = cls["escalate_reason"]

    # ---- 2. Fast-path: forced escalation ----
    if force_escalate:
        return {
            "status": "escalated",
            "product_area": _infer_product_area_from_domain(domain),
            "response": (
                "This issue requires attention from a specialist. "
                "Your ticket has been escalated to our support team "
                "who will follow up with you directly."
            ),
            "justification": escalate_reason,
            "request_type": request_type,
        }

    # ---- 3. Fast-path: clearly invalid / out-of-scope ----
    if request_type == "invalid":
        return {
            "status": "replied",
            "product_area": "",
            "response": (
                "I am sorry, this is outside the scope of my capabilities "
                "as a support agent for HackerRank, Claude, and Visa."
            ),
            "justification": (
                "The request is not related to any supported product area "
                "and does not require escalation."
            ),
            "request_type": "invalid",
        }

    # ---- 4. Retrieve relevant passages ----
    retriever = _get_retriever()
    company_filter = domain if domain != "unknown" else None
    results = retriever.search(issue, company=company_filter, top_k=5)

    # If no useful results at all, try without domain filter
    if not results or results[0].score < 0.05:
        results = retriever.search(issue, company=None, top_k=5)

    # ---- 5. Build the LLM prompt ----
    passages_block = _format_passages(results)

    user_prompt = _build_user_prompt(
        issue=issue,
        subject=subject,
        company=company,
        domain=domain,
        request_type=request_type,
        passages=passages_block,
    )

    # ---- 6. Call the LLM ----
    raw = _call_llm(user_prompt)

    # ---- 7. Parse structured output ----
    output = _parse_llm_output(raw, cls)

    # ---- 8. Ensure response meets minimum length (20 words) ----
    output = _ensure_min_response_length(output)

    return output


# ======================================================================
# Internal helpers
# ======================================================================

def _infer_product_area_from_domain(domain: str) -> str:
    """Fallback product_area when we skip retrieval."""
    return {
        "hackerrank": "general_support",
        "claude": "general_support",
        "visa": "general_support",
    }.get(domain, "")


def _ensure_min_response_length(output: dict, min_words: int = 20) -> dict:
    """Pad responses that are too short to meet minimum word count."""
    resp = output.get("response", "")
    word_count = len(resp.split())

    if word_count < min_words:
        status = output.get("status", "replied")
        if status == "escalated":
            output["response"] = (
                resp + " "
                "We understand this may be urgent. A member of our support "
                "team will review your case and reach out to you shortly "
                "with further assistance. Please do not hesitate to provide "
                "any additional details that may help us resolve your issue."
            )
        else:
            output["response"] = (
                resp + " "
                "If you could provide more details about what you are "
                "experiencing, including any error messages or steps you "
                "have already tried, we would be happy to assist you further."
            )
    return output


def _format_passages(results: list) -> str:
    """Format retrieved passages for the LLM context window."""
    if not results:
        return "(No relevant passages found in the support corpus.)"

    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(
            f"--- Passage {i} (score={r.score:.3f}, "
            f"domain={r.domain}, source={r.source_file}) ---\n"
            f"Title: {r.title}\n\n"
            f"{r.text}"
        )
    return "\n\n".join(blocks)


def _build_user_prompt(
    issue: str,
    subject: str,
    company: str,
    domain: str,
    request_type: str,
    passages: str,
) -> str:
    """Assemble the user message for the LLM."""
    parts = [
        "## Support Corpus Passages\n",
        passages,
        "\n\n## Ticket\n",
        f"Company: {company}",
        f"Subject: {subject}" if subject else "",
        f"Issue: {issue}",
        f"\n## Pre-classification hints (from rule engine)\n",
        f"domain: {domain}",
        f"request_type: {request_type}",
        "\nUsing ONLY the passages above, produce the JSON output.",
    ]
    return "\n".join(p for p in parts if p)


def _call_llm(user_prompt: str, max_retries: int = 3) -> str:
    """Call Groq API and return the raw text response.

    Retries with exponential backoff on rate-limit errors.
    """
    client = _get_client()
    delays = [10, 30, 60]  # seconds between retries

    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if ("429" in error_str or "rate" in error_str.lower()) and attempt < max_retries:
                wait = delays[min(attempt, len(delays) - 1)]
                print(f"  [agent] Rate limited. "
                      f"Retrying in {wait}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise


def _parse_llm_output(raw: str, cls: dict) -> Dict[str, str]:
    """
    Parse the LLM's JSON response.  Falls back gracefully if the
    model returns malformed output.
    """
    # Strip markdown fences if the model wrapped them anyway
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Last resort: treat entire response as the answer
        return {
            "status": "replied",
            "product_area": _infer_product_area_from_domain(cls["domain"]),
            "response": raw.strip(),
            "justification": "LLM output was not valid JSON; returning raw text.",
            "request_type": cls["request_type"],
        }

    # Normalise and validate
    status = str(data.get("status", "replied")).lower().strip()
    if status not in ("replied", "escalated"):
        status = "replied"

    req_type = str(data.get("request_type", cls["request_type"])).lower().strip()
    if req_type not in ("product_issue", "feature_request", "bug", "invalid"):
        req_type = cls["request_type"]

    return {
        "status": status,
        "product_area": str(data.get("product_area", "")).strip(),
        "response": str(data.get("response", "")).strip(),
        "justification": str(data.get("justification", "")).strip(),
        "request_type": req_type,
    }


# ======================================================================
# Smoke test
# ======================================================================

if __name__ == "__main__":
    import sys

    test_cases = [
        # Escalated by classifier (no LLM call)
        ("My identity has been stolen, what should I do",
         "Identity Theft", "Visa"),
        # Invalid / out-of-scope (no LLM call)
        ("What is the name of the actor in Iron Man?",
         "Urgent, please help", "None"),
        # Normal ticket (LLM call)
        ("How do I add extra time for a candidate taking a test?",
         "Extra time", "HackerRank"),
    ]

    print("=" * 60)
    print("agent.py smoke test")
    print("=" * 60)

    for i, (issue, subject, company) in enumerate(test_cases, 1):
        print(f"\n{'=' * 60}")
        print(f"Test {i}: {issue[:60]}...")
        print(f"{'=' * 60}")
        try:
            result = process_ticket(issue, subject, company)
            for k, v in result.items():
                val = str(v)[:120]
                print(f"  {k:16s}: {val}")
        except Exception as e:
            print(f"  ERROR: {e}")
