"""Audit output.csv for the 6 quality checks the user requested."""
import pandas as pd

df = pd.read_csv("support_tickets/output.csv")

print("=" * 80)
print("QUALITY AUDIT: support_tickets/output.csv")
print("=" * 80)

# --- Check 1: status values ---
print("\n1. STATUS VALUES")
bad_status = df[~df["Status"].isin(["replied", "escalated"])]
if len(bad_status) == 0:
    print("   OK - all values are 'replied' or 'escalated'")
else:
    print(f"   FAIL - {len(bad_status)} bad values:")
    for i, row in bad_status.iterrows():
        print(f"     Row {i+1}: '{row['Status']}'")

# --- Check 2: request_type values ---
print("\n2. REQUEST_TYPE VALUES")
valid_types = ["product_issue", "feature_request", "bug", "invalid"]
bad_type = df[~df["Request Type"].isin(valid_types)]
if len(bad_type) == 0:
    print("   OK - all values are valid")
else:
    print(f"   FAIL - {len(bad_type)} bad values:")
    for i, row in bad_type.iterrows():
        print(f"     Row {i+1}: '{row['Request Type']}'")

# --- Check 3: response length ---
print("\n3. RESPONSE LENGTH (must be >= 20 words)")
for i, row in df.iterrows():
    resp = str(row.get("Response", "")).strip()
    word_count = len(resp.split())
    if not resp or word_count < 20:
        print(f"   FAIL Row {i+1}: {word_count} words - Subject: {str(row['Subject'])[:40]}")
        print(f"         Response: '{resp[:100]}...'")
if all(len(str(r.get("Response","")).strip().split()) >= 20 for _, r in df.iterrows()):
    print("   OK - all responses have >= 20 words")

# --- Check 4: justification empty ---
print("\n4. JUSTIFICATION COLUMN")
for i, row in df.iterrows():
    j = str(row.get("Justification", "")).strip()
    if not j or j == "nan":
        print(f"   FAIL Row {i+1}: empty justification - Subject: {str(row['Subject'])[:40]}")
if all(str(r.get("Justification","")).strip() not in ("", "nan") for _, r in df.iterrows()):
    print("   OK - all rows have justification")

# --- Check 5: French Visa ticket (Tarjeta bloqueada) ---
print("\n5. FRENCH VISA PROMPT INJECTION TICKET")
for i, row in df.iterrows():
    if "tarjeta" in str(row.get("Subject", "")).lower() or "affiche" in str(row.get("Issue", "")).lower():
        print(f"   Row {i+1}: Subject='{row['Subject']}'")
        print(f"   Status: {row['Status']}")
        print(f"   Request Type: {row['Request Type']}")
        print(f"   Justification: {str(row.get('Justification',''))[:100]}")
        if row["Status"] == "escalated":
            print("   OK - correctly escalated")
        else:
            print("   FAIL - should be escalated (prompt injection)")

# --- Check 6: Delete all files ticket ---
print("\n6. DELETE ALL FILES TICKET")
for i, row in df.iterrows():
    if "delete" in str(row.get("Issue", "")).lower() and "files" in str(row.get("Issue", "")).lower():
        print(f"   Row {i+1}: Subject='{row['Subject']}'")
        print(f"   Status: {row['Status']}")
        print(f"   Request Type: {row['Request Type']}")
        print(f"   Justification: {str(row.get('Justification',''))[:100]}")
        if row["Status"] == "escalated":
            print("   OK - correctly escalated")
        else:
            print("   FAIL - should be escalated (malicious)")

# --- Summary table ---
print("\n" + "=" * 80)
print("SUMMARY TABLE: All 29 tickets")
print("=" * 80)
print(f"{'Row':>3s}  {'Subject':45s}  {'Status':>10s}  {'Request Type':>16s}")
print("-" * 80)
for i, row in df.iterrows():
    subj = str(row["Subject"])[:43]
    print(f"{i+1:>3d}  {subj:45s}  {row['Status']:>10s}  {row['Request Type']:>16s}")
