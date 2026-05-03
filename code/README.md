# Support Triage Agent

A terminal-based AI agent that processes customer support tickets for three product ecosystems -- HackerRank, Claude (Anthropic), and Visa -- and produces structured triage decisions. The agent retrieves relevant documentation from a support corpus, classifies tickets using rule-based heuristics, and generates grounded responses via LLM, escalating high-risk cases to human agents.

Built for the **HackerRank Orchestrate** hackathon (May 2026).

---

## Architecture

```
support_tickets.csv
        |
        v
  +------------+     +--------------+     +-------------+
  | main.py    | --> | classifier.py| --> | retriever.py |
  | (pipeline) |     | (rules)      |     | (TF-IDF)    |
  +------------+     +--------------+     +-------------+
        |                                       |
        v                                       v
  +------------+                         corpus passages
  | agent.py   | <------------------------------|
  | (LLM call) |
  +------------+
        |
        v
    output.csv
```

### File Overview

| File | Lines | Purpose |
|------|-------|---------|
| **`retriever.py`** | ~310 | Loads all markdown files from `data/{hackerrank,claude,visa}/`, chunks them by headings/paragraphs, builds a TF-IDF index (bigrams, sublinear TF, 20K features), and provides `search(query, company, top_k)` with cosine similarity ranking. Loads 6,944 passages from 774 files. |
| **`classifier.py`** | ~450 | Rule-based ticket classifier. Determines domain, request type, and whether to force-escalate. Uses regex patterns across 8 escalation categories: fraud/stolen, score manipulation, security vulnerabilities, billing IDs, malicious requests, prompt injection, legal threats, and system outages. |
| **`agent.py`** | ~360 | Orchestration layer. For each ticket: (1) runs classifier, (2) fast-paths escalation or invalid tickets, (3) retrieves top-5 corpus passages, (4) calls Groq API (Llama 3.1 8B) with a system prompt that constrains the model to corpus-only answers, (5) parses structured JSON output. |
| **`main.py`** | ~240 | Entry point. Reads input CSV, processes each ticket through the agent pipeline, prints progress, writes output CSV. Supports `--dry-run` mode (no API calls), `--input`/`--output` path overrides, and graceful error handling. |

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `scikit-learn` -- TF-IDF vectorization and cosine similarity
- `numpy` -- numerical operations
- `python-dotenv` -- environment variable loading from `.env`
- `groq` -- Groq API client for LLM inference
- `pandas` -- CSV reading/writing (used by main.py)

**Python version:** 3.10+

---

## Setup

1. Copy the environment template:
   ```bash
   cp .env.example .env
   ```

2. Add your Groq API key to `.env`:
   ```
   GROQ_API_KEY=gsk_your_api_key_here
   ```
   Get a free key at [console.groq.com](https://console.groq.com).

3. Never commit `.env` to git (it is already in `.gitignore`).

---

## How to Run

### Full run (with LLM)
```bash
python code/main.py
```

### Dry-run (classifier + retriever only, no API calls)
```bash
python code/main.py --dry-run
```

### Custom input/output paths
```bash
python code/main.py --input support_tickets/sample_support_tickets.csv --output support_tickets/sample_output.csv
```

### Expected output
```
Loading tickets from: support_tickets/support_tickets.csv
Found 29 tickets to process.

[retriever] Loaded 6944 passages from 774 files across 3 domains
[retriever] TF-IDF index built: 6944 docs x 20000 features

============================================================
  Processing 29 tickets  [LIVE]
============================================================

[ 1/29] Claude access lost                    -> replied
[ 2/29] Test Score Dispute                    -> escalated
...
[29/29] Visa card minimum spend               -> replied

  Total processed : 29/29
  Errors          : 0

Done! Output saved to support_tickets/output.csv
```

---

## Design Decisions

### TF-IDF over embeddings

The corpus is 774 markdown files with heavy domain jargon ("CodePair", "HackerRank Certified", "Visa Secure"). TF-IDF with bigrams picks up these compound terms directly -- a query mentioning "CodePair interview" will surface the CodePair docs because those exact tokens are rare and highly weighted. Embeddings would map "CodePair" to a generic dense vector where it might compete with unrelated terms that happen to sit nearby in embedding space.

The real trade-off is recall on paraphrased queries. If someone writes "how do I remove a team member" and the docs say "deactivate a user," TF-IDF misses it because the surface forms don't overlap. Embeddings would catch that. But in practice, most support tickets reuse the product's own terminology -- users say "test" not "assessment," "candidate" not "applicant" -- so TF-IDF's weakness rarely fires on this dataset.

The other factor is that embedding models require either an API call per query (adding latency and a failure point) or a local model (adding 500MB+ to dependencies). For 29 tickets against a 7K-passage index, scikit-learn's sparse matrix math finishes the entire batch in under 3 seconds total. That's fast enough that optimization would be wasted effort.

### Rule-based escalation over LLM-based

The classifier handles two categories that should never touch an LLM: **safety-critical routing** and **obvious filtering**.

For safety: a ticket saying "my identity has been stolen" must always escalate. If the LLM decides it can "help" with identity theft from a knowledge base article, that's a liability. Regex gives a hard guarantee -- if the pattern matches, the ticket escalates, regardless of what the LLM would have said. This is the same reason production fraud systems use deterministic rules as a first pass before ML scoring.

For filtering: tickets like "who played Iron Man" or "thanks for your help" are clearly out of scope. Burning an API call to have an LLM figure that out is wasteful when a keyword check against the corpus domains takes microseconds.

The cost is maintenance. Every new escalation category requires hand-writing regex patterns and testing them against edge cases. The current set covers 8 categories (fraud, score manipulation, security, billing, malicious, prompt injection, legal, outages). A 9th category would mean more regex, more tests, more risk of false positives. At scale, an LLM-based classifier with a few-shot prompt would be more maintainable -- but for 8 categories and 29 tickets, explicit rules are more trustworthy.

### Groq / Llama 3.1 8B for response generation

This was a pragmatic choice driven by the hackathon constraints. The original plan was Gemini 2.0 Flash, but the free-tier quota hit zero mid-development. Groq's free tier offered enough headroom (30 RPM, 6K tokens/min) and Llama 3.1 8B is the smallest model that reliably follows the structured JSON output format the agent requires.

The 8B model's weakness shows on ambiguous tickets. When a ticket is vague ("it's not working, help"), a 70B model would more reliably say "I don't have enough information" and escalate. The 8B model sometimes attempts an answer from loosely related passages. The `_ensure_min_response_length` function exists specifically to catch cases where the model's answer is too thin, padding it with a follow-up prompt.

If this were production, I'd use Llama 3.1 70B (still on Groq, still fast) or Gemini Flash with a billed account. The 8B model is the minimum viable quality for this task.

### Grounding strategy

The system prompt constrains the LLM with four hard rules:
1. Answer ONLY using the provided corpus passages
2. NEVER invent policies, URLs, or procedures
3. If the passages don't contain the answer, set status to "escalated"
4. Return structured JSON with all required fields

This works because the retriever pre-filters to 5 relevant passages, so the model doesn't need to decide *what* to cite -- it just needs to synthesize from what it's given. The failure mode is when retrieval brings back irrelevant passages (because TF-IDF matched on surface keywords but not meaning), and the model tries to stitch together an answer from bad context. The escalation fallback catches most of these cases.

---

## What Breaks at 10,000 Tickets

The honest answer: **the API rate limit hits first, then the retriever.**

1. **Groq rate limits.** At 30 RPM on the free tier, 10,000 tickets would take ~5.5 hours of pure API time. With the current sequential processing (no parallelism), that's the hard bottleneck. Fix: batch with asyncio, use a paid tier (6K RPM), or add a local model fallback for tickets that don't need high quality.

2. **TF-IDF index rebuild.** The index is rebuilt from scratch every run. At 7K passages this takes 2 seconds. The index scales as O(n * m) where n=documents and m=features. The corpus itself won't grow (it's static), but if it did grow to 70K passages, index time would balloon. Fix: serialize the fitted vectorizer and matrix to disk with `joblib.dump()`, only rebuild when the corpus changes.

3. **Classifier false positives.** At 29 tickets, the 8-category regex set has zero false positives. At 10K tickets, you'd hit edge cases: a ticket about "stealing market share" would match the fraud pattern, a ticket mentioning "legal team" in passing would trigger the legal-threat rule. Fix: add a confidence layer -- regex flags candidates, then a lightweight classifier (even logistic regression on TF-IDF features) confirms or overrides.

4. **Memory.** The TF-IDF matrix for 7K passages is ~50MB in RAM. At 70K passages it would be ~500MB. Still manageable on a laptop, but it's a dense matrix that gets fully loaded. Fix: use sparse matrix storage (which scikit-learn already does internally) and lazy-load the corpus.

5. **Error recovery.** The current pipeline processes tickets sequentially and skips failures. At 10K tickets, a single Groq outage could stall the run for minutes. Fix: implement checkpointing -- write partial results to disk after every N tickets, resume from the last checkpoint on restart.

---

## Known Limitations

1. **Keyword mismatch on paraphrased queries.** TF-IDF can't bridge vocabulary gaps. "Remove a team member" won't match "deactivate user account" unless those words co-occur in the corpus. This affects maybe 10-15% of queries.

2. **8K context window.** Five retrieved passages plus the system prompt consume ~3K tokens, leaving ~5K for the response. Tickets with very long issue descriptions may get truncated before the model sees the full context.

3. **No conversation history.** Each ticket is processed in isolation. If ticket #15 says "as I mentioned in my last email," the agent can't look up that prior context.

4. **Regex escalation is English-only.** The prompt injection detector catches French and Spanish attempts via specific patterns, but a prompt injection in Mandarin or Arabic would likely slip through.

5. **No response verification.** The agent trusts the LLM's output. If the model cites a procedure that's subtly wrong (e.g., swaps two steps), there's no automated check against the source passage.

