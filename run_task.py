"""Resolve the exact per-task artifact and invoke the pinned official evaluator."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent

def command(task, setting, gpu, extra):
    manifest = json.loads((ROOT/'task_mapping.json').read_text())
    entry = manifest[task]
    local_root = os.environ.get('ROBOSYN_LOCAL_CHECKPOINT_ROOT')
    if local_root:
        checkpoint = str(Path(local_root)/task)
    else:
        if not entry['revision']:
            raise RuntimeError(f'{task}: checkpoint is not published/pinned yet; release is a draft')
        files = ['config.json','model.safetensors','policy_preprocessor*','policy_postprocessor*']
        prefix = entry['subdirectory'] + '/' if entry['subdirectory'] else ''
        checkpoint = str(Path(snapshot_download(entry['repo_id'],revision=entry['revision'],
            allow_patterns=[prefix+name for name in files],token=False))/entry['subdirectory'])
    if entry['family']!='act':
        raise ValueError('This submitted release uses ACT for all ten tasks')
    cmd = [sys.executable,'scripts/eval_policy.py','--config','policy/team_act/deploy_policy.yml',
        '--overrides','--task_name',task,'--setting',setting,'--gpu_id',str(gpu),
        '--pytorch_device','cuda','--headless','True','--renderer','fast-rt',
        '--eval_max_seed_attempts_per_episodes','100']
    cmd += ['--checkpoint_path',checkpoint]
    cmd += ['--n_action_steps',str(entry['n_action_steps']),'--act_step',str(entry['outer_action_group'])]
    # Optional external site wrapper; default is the unchanged official evaluator.
    # Used on SYNKA only for assigned CUDA/Vulkan alignment and native path limits.
    if os.environ.get('ROBOSYN_EVALUATOR_RUNNER'):
        cmd[1] = os.environ['ROBOSYN_EVALUATOR_RUNNER']
    return cmd + extra

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task',choices=json.loads((ROOT/'task_mapping.json').read_text()))
    parser.add_argument('setting',nargs='?',default='random',choices=['random','clear'])
    parser.add_argument('gpu',nargs='?',default=0,type=int)
    args,extra = parser.parse_known_args()
    target = ROOT/'third_party/RoboSynChallenge'
    if not (target/'policy/team_act/deploy_policy.py').exists():
        raise RuntimeError('Run setup.sh first')
    os.environ['HF_HUB_DISABLE_XET']='1'
    os.environ['EMBODICHAIN_SIM_EXIT_PROCESS']='0'
    subprocess.run(command(args.task,args.setting,args.gpu,extra),cwd=target,check=True)
