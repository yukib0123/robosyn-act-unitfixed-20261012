#!/bin/bash
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
"${PYTHON_BIN:-python}" "$root/run_task.py" "$@"
