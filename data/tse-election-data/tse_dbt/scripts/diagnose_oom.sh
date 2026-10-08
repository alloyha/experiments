#!/usr/bin/env bash
set -euo pipefail
echo "=== memory ==="
free -h
echo
echo "=== swap ==="
swapon --show || true
echo
echo "=== filesystem ==="
df -h .
echo
echo "=== WSL kernel OOM evidence (may require sudo) ==="
dmesg 2>/dev/null | grep -Ei 'out of memory|oom|killed process' | tail -20 || true
