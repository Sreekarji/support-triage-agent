"""
main.py -- Entry point for the support triage agent.

Reads support_tickets/support_tickets.csv, processes each ticket
through the agent pipeline, and writes results to
support_tickets/output.csv.

Usage:
    python code/main.py                  # full run (calls Gemini API)
    python code/main.py --dry-run        # classifier + retriever only (no API)
"""

import os
import sys
import csv
import random
import argparse
import traceback
from pathlib import Path

# Ensure code/ is on the path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import pandas as pd
from classifier import classify
from retriever import Retriever


# ======================================================================
# Dry-run processor (no LLM)
# ======================================================================

def process_ticket_dry(issue: str, subject: str, company: str,
                       retriever: Retriever) -> dict:
    """
    Classify + retrieve only.  No Gemini API call.
    Produces a best-effort result using only rule-based logic.
    """
    cls = classify(issue, subject, company)
    domain = cls["domain"]
    request_type = cls["request_type"]

    # --- Forced escalation ---
    if cls["force_escalate"]:
        return {
            "status": "escalated",
            "product_area": "general_support",
            "response": (
                "This issue requires attention from a specialist. "
                "Your ticket has been escalated to our support team "
                "who will follow up with you directly."
            ),
            "justification": cls["escalate_reason"],
            "request_type": request_type,
        }

    # --- Invalid / out-of-scope ---
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

    # --- Retrieve top passages ---
    company_filter = domain if domain != "unknown" else None
    results = retriever.search(issue, company=company_filter, top_k=3)

    if not results or results[0].score < 0.05:
        results = retriever.search(issue, company=None, top_k=3)

    if results:
        top = results[0]
        # Infer product_area from the source file path
        parts = Path(top.source_file).parts
        product_area = "/".join(parts[1:-1]) if len(parts) > 2 else parts[0]
        response_text = (
            f"[DRY-RUN] Based on retrieved docs: {top.title}\n"
            f"Source: {top.source_file} (score={top.score:.3f})"
        )
        justification = (
            f"Top retrieval hit: {top.title} "
            f"(score={top.score:.3f}, domain={top.domain})"
        )
    else:
        product_area = "general_support"
        response_text = "[DRY-RUN] No relevant passages found."
        justification = "No passages matched; would escalate in live mode."

    return {
        "status": "replied",
        "product_area": product_area,
        "response": response_text,
        "justification": justification,
        "request_type": request_type,
    }


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Support Triage Agent -- process tickets from CSV"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run classifier + retriever only, skip Gemini API calls",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to input CSV (default: support_tickets/support_tickets.csv)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output CSV (default: support_tickets/output.csv)",
    )
    args = parser.parse_args()

    # --- Paths ---
    repo_root = Path(__file__).resolve().parent.parent
    input_path = Path(args.input) if args.input else repo_root / "support_tickets" / "support_tickets.csv"
    output_path = Path(args.output) if args.output else repo_root / "support_tickets" / "output.csv"

    # --- Seed for reproducibility ---
    random.seed(42)

    # --- Load input CSV ---
    print(f"Loading tickets from: {input_path}")
    df = pd.read_csv(input_path)
    total = len(df)
    print(f"Found {total} tickets to process.\n")

    # --- Initialise retriever (shared across all tickets) ---
    retriever = Retriever()
    print()

    # --- Lazy-import agent only when needed ---
    if not args.dry_run:
        from agent import process_ticket
        # Inject our retriever into the agent module to avoid double-init
        import agent as agent_module
        agent_module._retriever = retriever

    mode_label = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"{'=' * 60}")
    print(f"  Processing {total} tickets  [{mode_label}]")
    print(f"{'=' * 60}\n")

    # --- Process each ticket ---
    rows_out = []
    errors = []

    for idx, row in df.iterrows():
        ticket_num = idx + 1
        issue = str(row.get("Issue", "")).strip()
        subject = str(row.get("Subject", "")).strip()
        company = str(row.get("Company", "")).strip()

        print(f"[{ticket_num:>2d}/{total}] {subject[:50] or '(no subject)':<50s} ", end="")

        try:
            if args.dry_run:
                result = process_ticket_dry(issue, subject, company, retriever)
            else:
                result = process_ticket(issue, subject, company)

            status = result.get("status", "replied")
            print(f" -> {status}")

            rows_out.append({
                "Issue": issue,
                "Subject": subject,
                "Company": company,
                "Response": result.get("response", ""),
                "Product Area": result.get("product_area", ""),
                "Status": result.get("status", "replied"),
                "Request Type": result.get("request_type", "product_issue"),
                "Justification": result.get("justification", ""),
            })

        except Exception as e:
            error_msg = f"Ticket {ticket_num}: {type(e).__name__}: {e}"
            print(f" -> ERROR: {e}")
            errors.append(error_msg)
            traceback.print_exc()

            # Write a fallback row so output CSV always has all rows
            rows_out.append({
                "Issue": issue,
                "Subject": subject,
                "Company": company,
                "Response": "An error occurred processing this ticket.",
                "Product Area": "",
                "Status": "escalated",
                "Request Type": "product_issue",
                "Justification": f"Processing error: {e}",
            })

    # --- Write output CSV ---
    print(f"\n{'=' * 60}")
    print(f"  Writing results to: {output_path}")
    print(f"{'=' * 60}")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)

    # --- Summary ---
    status_counts = out_df["Status"].value_counts().to_dict()
    type_counts = out_df["Request Type"].value_counts().to_dict()

    print(f"\n  Total processed : {len(rows_out)}/{total}")
    print(f"  Errors          : {len(errors)}")
    print(f"  Status breakdown: {status_counts}")
    print(f"  Type breakdown  : {type_counts}")

    if errors:
        print(f"\n  Errors encountered:")
        for err in errors:
            print(f"    - {err}")

    print(f"\nDone! Output saved to {output_path}")


if __name__ == "__main__":
    main()
