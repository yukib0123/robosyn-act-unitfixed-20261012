#!/bin/bash
# Use a working official simulator environment; see README for ACT package pins.
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
target="$root/third_party/RoboSynChallenge"
revision=ecbf88ae26bbd8e369eba868e23354c8f64e083f
if [[ ! -d "$target" ]]; then
    mkdir -p "$target"
    git -C "$target" init -q
    git -C "$target" remote add origin https://github.com/EDEM-AI/RoboSynChallenge.git
    git -C "$target" fetch --depth 1 origin "$revision"
    git -C "$target" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$target" rev-parse HEAD)" == "$revision" ]] || { echo 'Unexpected official evaluator revision.' >&2; exit 1; }
mkdir -p "$target/policy/team_act"
for name in __init__.py eval.sh deploy_policy.py deploy_policy.yml checkpoints.json; do
    cp "$root/policy/team_act/$name" "$target/policy/team_act/$name"
done
"${PYTHON_BIN:-python}" "$root/verify_environment.py"
echo 'Pinned ACT adapter installed; default evaluation setting is random.'
