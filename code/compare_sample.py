"""Compare sample output against expected values."""
import pandas as pd

expected = pd.read_csv("support_tickets/sample_support_tickets.csv")
actual = pd.read_csv("support_tickets/sample_output.csv")

print(f"{'Row':>3s}  {'Subject':40s}  {'Exp Status':>12s} {'Got Status':>12s}  {'Exp ReqType':>16s} {'Got ReqType':>16s}  Match?")
print("-" * 140)

mismatches = []
for i in range(len(expected)):
    subj = str(expected.iloc[i].get("Subject", ""))[:38]
    exp_status = str(expected.iloc[i].get("Status", "")).strip().lower()
    got_status = str(actual.iloc[i].get("Status", "")).strip().lower()
    exp_type = str(expected.iloc[i].get("Request Type", "")).strip().lower()
    got_type = str(actual.iloc[i].get("Request Type", "")).strip().lower()

    s_ok = exp_status == got_status
    t_ok = exp_type == got_type
    match = "OK" if (s_ok and t_ok) else "WRONG"

    flag = "" if match == "OK" else " <<<<"
    print(f"{i+1:>3d}  {subj:40s}  {exp_status:>12s} {got_status:>12s}  {exp_type:>16s} {got_type:>16s}  {match}{flag}")

    if match == "WRONG":
        mismatches.append({
            "row": i+1,
            "subject": subj,
            "status_wrong": not s_ok,
            "exp_status": exp_status,
            "got_status": got_status,
            "type_wrong": not t_ok,
            "exp_type": exp_type,
            "got_type": got_type,
        })

print(f"\nTotal: {len(expected) - len(mismatches)}/{len(expected)} correct")
if mismatches:
    print(f"\n=== MISMATCHES ({len(mismatches)}) ===")
    for m in mismatches:
        print(f"  Row {m['row']}: {m['subject']}")
        if m["status_wrong"]:
            print(f"    status: expected={m['exp_status']}, got={m['got_status']}")
        if m["type_wrong"]:
            print(f"    request_type: expected={m['exp_type']}, got={m['got_type']}")
        # Show the issue text for context
        issue = str(expected.iloc[m["row"]-1].get("Issue", ""))[:120]
        print(f"    issue: {issue}")
