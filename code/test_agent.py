"""Test agent.py -- fast paths first (no API key needed), then LLM path."""
import sys, os
sys.path.insert(0, "code")

from agent import process_ticket

# --- Test 1: Forced escalation (no LLM call) ---
print("=" * 60)
print("Test 1: Stolen identity -> should escalate (no LLM)")
print("=" * 60)
r = process_ticket("My identity has been stolen, what should I do", "Identity Theft", "Visa")
for k, v in r.items():
    print(f"  {k:16s}: {str(v)[:120]}")

# --- Test 2: Invalid / out-of-scope (no LLM call) ---
print(f"\n{'=' * 60}")
print("Test 2: Iron Man actor -> invalid (no LLM)")
print("=" * 60)
r = process_ticket("What is the name of the actor in Iron Man?", "Urgent, please help", "None")
for k, v in r.items():
    print(f"  {k:16s}: {str(v)[:120]}")

# --- Test 3: Normal ticket (needs API key) ---
print(f"\n{'=' * 60}")
print("Test 3: Extra time for candidate -> LLM call")
print("=" * 60)
if os.environ.get("GROQ_API_KEY"):
    r = process_ticket("How do I add extra time for a candidate taking a test?", "Extra time", "HackerRank")
    for k, v in r.items():
        print(f"  {k:16s}: {str(v)[:200]}")
else:
    print("  SKIPPED: GROQ_API_KEY not set")
    print("  Create .env with: GROQ_API_KEY=gsk_...")
