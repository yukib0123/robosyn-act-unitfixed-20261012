#!/bin/bash
# Run after this adapter has been installed in the official evaluator.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
task="${1:?Pass a task name}"
setting="${2:-random}"
gpu="${3:-0}"
if (( $# > 3 )); then shift 3; else set --; fi
python="${PYTHON_BIN:-python}"
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
"$python" scripts/eval_policy.py --config policy/team_act/deploy_policy.yml --overrides \
    --task_name "$task" --setting "$setting" --gpu_id "$gpu" \
    --pytorch_device cuda --headless True "$@"
