"""Pre-submission check."""
import pandas as pd
import os
import re

SEP = "=" * 60

# 1. Output CSV
print(SEP)
print("1. OUTPUT CSV EXISTS + ROW COUNT")
print(SEP)
path = "support_tickets/output.csv"
if os.path.exists(path):
    df = pd.read_csv(path)
    print(f"   EXISTS: {path}")
    print(f"   ROWS: {len(df)} (expected 29)")
    print("   OK" if len(df) == 29 else "   FAIL")
else:
    print(f"   MISSING: {path}")
    df = None

# 2. Column headers
print(f"\n{SEP}")
print("2. COLUMN HEADERS")
print(SEP)
if df is not None:
    expected = ["Issue", "Subject", "Company", "Response", "Product Area",
                "Status", "Request Type", "Justification"]
    actual = list(df.columns)
    print(f"   Expected: {expected}")
    print(f"   Actual:   {actual}")
    if actual == expected:
        print("   OK - exact match")
    else:
        missing = set(expected) - set(actual)
        extra = set(actual) - set(expected)
        if missing:
            print(f"   MISSING cols: {missing}")
        if extra:
            print(f"   EXTRA cols: {extra}")

# 3. No API keys
print(f"\n{SEP}")
print("3. NO API KEYS IN code/")
print(SEP)
key_patterns = [
    r"gsk_[a-zA-Z0-9]{20,}",
    r"sk-[a-zA-Z0-9]{20,}",
    r"AIza[a-zA-Z0-9_\-]{30,}",
]
found_keys = False
for root, dirs, files in os.walk("code"):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in files:
        if f.endswith(".pyc"):
            continue
        fp = os.path.join(root, f)
        try:
            content = open(fp, "r", encoding="utf-8", errors="ignore").read()
            for pat in key_patterns:
                matches = re.findall(pat, content)
                if matches:
                    print(f"   FAIL: Key pattern in {fp}: {pat}")
                    found_keys = True
        except Exception:
            pass
if not found_keys:
    print("   OK - no API keys found in code/")

# 4. README
print(f"\n{SEP}")
print("4. README EXISTS")
print(SEP)
readme = "code/README.md"
if os.path.exists(readme):
    sz = os.path.getsize(readme)
    print(f"   EXISTS: {readme} ({sz} bytes)")
    print("   OK")
else:
    print(f"   MISSING: {readme}")

# 5. Files list
print(f"\n{SEP}")
print("5. FILES IN code/ FOR SUBMISSION")
print(SEP)
for root, dirs, files in os.walk("code"):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in sorted(files):
        if f.endswith(".pyc"):
            continue
        fp = os.path.join(root, f)
        sz = os.path.getsize(fp)
        print(f"   {fp:45s}  {sz:>8} bytes")

# 6. Zip command
print(f"\n{SEP}")
print("6. ZIP COMMAND")
print(SEP)
print("   PowerShell:")
print('   Compress-Archive -Path code, support_tickets, requirements.txt, .env.example, AGENTS.md, README.md, data -DestinationPath code_submission.zip -Force')
print()
print("   Or with 7z (excludes .env, __pycache__, .pyc):")
print('   7z a code_submission.zip code support_tickets requirements.txt .env.example AGENTS.md README.md data -xr!__pycache__ -xr!*.pyc -x!.env')

# 7. Log path
print(f"\n{SEP}")
print("7. LOG FILE PATH")
print(SEP)
home = os.path.expanduser("~")
log_path = os.path.join(home, "hackerrank_orchestrate", "log.txt")
print(f"   {log_path}")
if os.path.exists(log_path):
    sz = os.path.getsize(log_path)
    print(f"   EXISTS ({sz} bytes)")
else:
    print("   NOT FOUND")
