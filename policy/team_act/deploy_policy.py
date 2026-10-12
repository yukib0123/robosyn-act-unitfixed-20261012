"""ACT adapter for the official RoboSynChallenge evaluator (all ten tasks)."""
import json
from pathlib import Path
import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from policy.inference_timing import start_inference, finish_inference

NORMALIZED_GRIPPER_TASKS = frozenset({
    'water_pouring', 'handle_basket', 'items_handover', 'drawer_open_place',
    'mixer_operating', 'item_assembly', 'manipulate_pipette', 'sample_loading',
})
PHYSICAL_GRIPPER_TASKS = frozenset({'table_rearrangement', 'click_bell'})
GRIPPER_INDICES = (6, 13)


def action_to_environment(action, env, task):
    """Convert once from checkpoint dataset units to physical joint commands."""
    if task not in NORMALIZED_GRIPPER_TASKS | PHYSICAL_GRIPPER_TASKS:
        raise ValueError(f'Unknown task gripper units: {task}')
    result = action.detach().to(env.unwrapped.device, dtype=torch.float32).clone()
    if task in NORMALIZED_GRIPPER_TASKS:
        space = env.unwrapped.single_action_space
        low = torch.as_tensor(np.asarray(space.low).copy(), device=result.device, dtype=result.dtype)
        high = torch.as_tensor(np.asarray(space.high).copy(), device=result.device, dtype=result.dtype)
        indices = list(GRIPPER_INDICES)
        result[..., indices] = low[indices] + result[..., indices].clamp(0, 1) * (high[indices] - low[indices])
    return result


def physical_gripper_positions(env, target=False):
    robot = env.unwrapped.robot
    qpos = robot.get_qpos(target=target).detach()
    active = env.unwrapped.active_joint_ids
    return qpos[..., active][..., list(GRIPPER_INDICES)].cpu().reshape(-1).tolist()


def get_model(usr_args):
    task = usr_args.get('task_name')
    if task not in NORMALIZED_GRIPPER_TASKS | PHYSICAL_GRIPPER_TASKS:
        raise ValueError(f'Unknown task gripper units: {task}')
    checkpoint = usr_args.get('checkpoint_path')
    if not checkpoint:
        manifest = json.loads((Path(__file__).parent / 'checkpoints.json').read_text())
        entry = manifest[usr_args['task_name']]
        checkpoint = snapshot_download(entry['repo_id'], revision=entry['revision'],
                                       allow_patterns=['config.json', 'model.safetensors',
                                                       entry.get('subdirectory', '') + '/*'], token=False)
        if entry.get('subdirectory'):
            checkpoint = str(Path(checkpoint) / entry['subdirectory'])
    device = usr_args.get('pytorch_device', 'cuda')
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = device
    config.pretrained_backbone_weights = None
    if usr_args.get('n_action_steps') is not None:
        n_action_steps = int(usr_args['n_action_steps'])
        if not 1 <= n_action_steps <= config.chunk_size:
            raise ValueError('n_action_steps must be between 1 and chunk_size')
        config.n_action_steps = n_action_steps
    model = ACTPolicy.from_pretrained(checkpoint, config=config, strict=True).to(device)
    model.eval()
    model.act_device = next(model.parameters()).device
    model.act_task_name = task
    model.gripper_trace = bool(usr_args.get('gripper_trace', False))
    model.trace_episode = -1
    model.trace_step = 0
    model.act_step = int(usr_args.get('act_step', 10))
    if model.act_step < 1:
        raise ValueError('act_step must be positive')
    print('ACT execution config:', dict(chunk_size=model.config.chunk_size,
        n_action_steps=model.config.n_action_steps,
        temporal_ensemble_coeff=model.config.temporal_ensemble_coeff,
        act_step=model.act_step, task=task,
        gripper_units='normalized_0_1' if task in NORMALIZED_GRIPPER_TASKS else 'physical'), flush=True)
    return model


def encode_obs(obs, model):
    state = torch.as_tensor(obs['robot']['qpos'], dtype=torch.float32, device=model.act_device)
    if state.ndim == 1:
        state = state.unsqueeze(0)
    batch = {'observation.state': state}
    for key in model.config.image_features:
        camera = key.removeprefix('observation.images.')
        image = torch.as_tensor(obs['sensor'][camera]['color'], device=model.act_device)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        image = image[..., :3].permute(0, 3, 1, 2).contiguous().float()
        if image.max() > 1.5:
            image = image / 255.0
        batch[key] = image
    return batch


def eval(env, model, obs):
    times = []
    info = None
    truncated = False
    for _ in range(model.act_step):
        query = model.config.temporal_ensemble_coeff is not None or len(model._action_queue) == 0
        started = start_inference(model.act_device) if query else None
        action = model.select_action(encode_obs(obs, model))
        if action.ndim == 1:
            action = action.unsqueeze(0)
        expected = int(np.prod(env.unwrapped.single_action_space.shape))
        if action.ndim != 2 or action.shape[-1] != expected:
            raise ValueError(f'Action shape {tuple(action.shape)} does not match environment dimension {expected}')
        raw_grippers = action.detach()[..., list(GRIPPER_INDICES)].cpu().reshape(-1).tolist() if model.gripper_trace else None
        action = action_to_environment(action, env, model.act_task_name)
        converted = action[..., list(GRIPPER_INDICES)].cpu().reshape(-1).tolist() if model.gripper_trace else None
        if query:
            finish_inference(started, times, model.act_device)
        obs, reward, terminated, truncated, info = env.step(action)
        if model.gripper_trace:
            observed = torch.as_tensor(obs['robot']['qpos'])[..., list(GRIPPER_INDICES)].reshape(-1).tolist()
            print('ACT_GRIPPER_TRACE ' + json.dumps(dict(
                episode=model.trace_episode, step=model.trace_step, task=model.act_task_name,
                raw_checkpoint_action=raw_grippers, converted_action=converted,
                post_step_may_be_reset=bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()),
                sent_to_env=action[..., list(GRIPPER_INDICES)].cpu().reshape(-1).tolist(),
                robot_target_m=physical_gripper_positions(env, target=True),
                robot_actual_m=physical_gripper_positions(env), observed_normalized=observed)), flush=True)
            model.trace_step += 1
        success = env.get_wrapper_attr('is_task_success')()
        if bool(torch.as_tensor(success).any()) or bool(torch.as_tensor(truncated).any()):
            break
    return obs, info, truncated, times


def reset_model(model):
    model.reset()
    model.trace_episode += 1
    model.trace_step = 0
