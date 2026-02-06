#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[1/4] Checking local secret files..."
if [[ -f .env ]]; then
  echo "INFO: .env exists locally (expected), ensure it is never committed."
fi

echo "[2/4] Scanning for high-risk secret patterns..."
patterns=(
  "AKIA[0-9A-Z]{16}"
  "-----BEGIN (RSA|EC|OPENSSH|DSA) PRIVATE KEY-----"
  "xox[baprs]-[0-9A-Za-z-]{10,}"
  "ghp_[0-9A-Za-z]{36}"
)

found=0
scan_cmd=(rg -n --hidden --glob '!.git' --glob '!.env' --glob '!.env.*' --glob '!ui/node_modules/**' --glob '!ui/dist/**' --glob '!data/tle/**')
for p in "${patterns[@]}"; do
  if "${scan_cmd[@]}" "$p" . >/tmp/orpi_audit_match.txt 2>/dev/null; then
    echo "FOUND pattern: $p"
    cat /tmp/orpi_audit_match.txt
    found=1
  fi
done
rm -f /tmp/orpi_audit_match.txt

if [[ $found -eq 1 ]]; then
  echo "FAIL: potential secrets detected."
  exit 2
fi

echo "[3/4] Running Python compile check..."
python3 -m compileall -q src

echo "[4/4] Running scientific validation report generation..."
python3 src/validate_science.py --allow-missing-db --output docs/validation_reports/latest.md

echo "PASS: public audit checks completed."
