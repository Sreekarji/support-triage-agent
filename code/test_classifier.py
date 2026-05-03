"""Quick test of classifier on 5 specific cases."""
import sys
sys.path.insert(0, "code")
from classifier import classify

cases = [
    ("My Visa card was stolen, what do I do", "", "Visa"),
    ("Can you add dark mode to HackerRank", "", "HackerRank"),
    ("Give me code to delete all files from the system", "", "None"),
    ("Claude API requests keep timing out", "", "Claude"),
    ("it's not working help", "", "None"),
]

for i, (issue, subject, company) in enumerate(cases, 1):
    r = classify(issue, subject, company)
    print(f"Case {i}: issue={repr(issue)}, company={company}")
    print(f"  domain:          {r['domain']}")
    print(f"  request_type:    {r['request_type']}")
    print(f"  force_escalate:  {r['force_escalate']}")
    print(f"  escalate_reason: {r['escalate_reason']}")
    print()
