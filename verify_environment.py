"""Import/API check without requesting a GPU or running a simulator."""
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.configs.policies import PreTrainedConfig
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'third_party/RoboSynChallenge'))
from policy.team_act import get_model,eval,reset_model
assert all(callable(fn) for fn in (get_model,eval,reset_model))
assert metadata.version('lerobot')=='0.3.3','Install the documented LeRobot commit'
manifest=json.loads((root/'task_mapping.json').read_text())
assert len(manifest)==10 and all(len(row['revision'])==40 for row in manifest.values())
assert all(row['family']=='act' for row in manifest.values())
assert manifest==json.loads((root/'policy/team_act/checkpoints.json').read_text())
print(json.dumps(dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,
    lerobot=metadata.version('lerobot'),safetensors=metadata.version('safetensors'),tasks=len(manifest),policy_entrypoints='OK')))
