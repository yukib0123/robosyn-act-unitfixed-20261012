"""Verify the released loader against a real recorded observation for all tasks."""
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import torch
from huggingface_hub import snapshot_download
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'third_party/RoboSynChallenge'))
from policy.team_act.deploy_policy import get_model,action_to_environment
from train_click_bell import ModelDataset,LeRobotDataset
data_root=Path(os.environ['VERIFY_DATA_ROOT'])
device=os.environ.get('VERIFY_DEVICE','cuda')
torch.set_num_threads(4)
manifest=json.loads((root/'task_mapping.json').read_text())
report={}
low=np.full(14,-3.14,dtype=np.float32);high=np.full(14,3.14,dtype=np.float32)
low[[6,13]]=0;high[[6,13]]=.05
env=SimpleNamespace(unwrapped=SimpleNamespace(device='cpu',single_action_space=SimpleNamespace(low=low,high=high)))
for task,entry in manifest.items():
    checkpoint=Path(snapshot_download(entry['repo_id'],revision=entry['revision'],
        allow_patterns=[(entry['subdirectory']+'/' if entry['subdirectory'] else '')+'*'],token=False))/entry['subdirectory']
    model=get_model(dict(task_name=task,checkpoint_path=str(checkpoint),pytorch_device=device,n_action_steps=50))
    dataset_root=data_root/task
    split=json.loads((dataset_root/'split.json').read_text());info=json.loads((dataset_root/'meta/info.json').read_text())
    data=ModelDataset(LeRobotDataset(split['repo_id'],root=dataset_root,episodes=[split['validation_episodes'][0]],
        revision=split['revision'],video_backend='pyav',delta_timestamps={'action':[i/info['fps'] for i in range(50)]}))
    row=data[min(10,len(data)-1)]
    batch={k:v.unsqueeze(0).to(device) for k,v in row.items() if k.startswith('observation.')}
    with torch.inference_mode():
        chunk=model.predict_action_chunk(batch);model.reset();action=model.select_action(batch)
        converted=action_to_environment(action,env,task)
    assert chunk.shape==(1,50,14) and action.shape==(1,14) and torch.isfinite(chunk).all() and torch.isfinite(converted).all()
    sha=hashlib.sha256((checkpoint/'model.safetensors').read_bytes()).hexdigest()
    if entry.get('weight_sha256'):assert sha==entry['weight_sha256']
    report[task]=dict(repo_id=entry['repo_id'],revision=entry['revision'],weight_sha256=sha,strict_reload=True,
        finite_prediction=True,chunk_shape=list(chunk.shape),raw_grippers=action[0,[6,13]].cpu().tolist(),
        physical_grippers=converted[0,[6,13]].tolist(),execution_length=model.config.n_action_steps)
    print(task,'RELEASE LOADER OK',flush=True)
    (root/'verification_selected.json').write_text(json.dumps(report,indent=2))
    del model,data,chunk,action,batch
    if device=='cuda':torch.cuda.empty_cache()
print('ALL TEN RELEASED CHECKPOINTS VERIFIED',flush=True)
