#!/usr/bin/env bash
# Run the four SlotDiff diffusion-conditioning ablations. See --help.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
else
  PYTHON_BIN=python
fi
exec "$PYTHON_BIN" "$ROOT/run_ablation.py" "$@"
