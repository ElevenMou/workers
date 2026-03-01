#!/usr/bin/env bash
# Run pip-audit against pinned requirements.
# Exit non-zero if any known vulnerability is found.
# Usage: ./scripts/audit-deps.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQUIREMENTS="${SCRIPT_DIR}/../requirements.txt"

echo "==> Running pip-audit on ${REQUIREMENTS}"
pip-audit -r "${REQUIREMENTS}" --strict --desc
echo "==> No known vulnerabilities found."
