"""Fine-tune public ACT on pinned demos without importing a simulator."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.constants import OBS_IMAGES

ALIASES = {
    'observation.state': 'observation.qpos',
    'observation.images.cam_high': 'cam_high.color',
    'observation.images.cam_right_wrist': 'cam_right_wrist.color',
    'observation.images.cam_left_wrist': 'cam_left_wrist.color',
}


class ModelDataset(Dataset):
    def __init__(self, base):
        self.base = base
        # LeRobot 0.3.3 stores compact episode offsets for a subset but indexes
        # them with the original episode ID in _get_query_indices. Keep those
        # original IDs and map them to offsets in this loaded subset.
        if base.episodes is not None:
            ids = torch.tensor(base.episodes, dtype=torch.long)
            compact = base.episode_data_index
            if len(ids) != len(compact['from']):
                raise RuntimeError('Episode offset count differs from selected episodes')
            for position, episode in enumerate(base.episodes):
                start = int(compact['from'][position])
                if int(base.hf_dataset[start]['episode_index']) != episode:
                    raise RuntimeError('Selected episode ordering differs from loaded dataset')
            remapped = {}
            for key in ('from', 'to'):
                offsets = torch.full((base.meta.total_episodes,), -1, dtype=torch.long)
                offsets[ids] = compact[key]
                remapped[key] = offsets
            base.episode_data_index = remapped

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        row = self.base[index]
        batch = {key: row[key] if key in row else row[source] for key, source in ALIASES.items()}
        batch['action'] = row['action']
        batch['action_is_pad'] = row['action_is_pad']
        return batch


def device_batch(batch):
    return {key: value.to('cuda', non_blocking=True) for key, value in batch.items()}


def prior_alignment_loss(policy, batch):
    """Train the actual z=0 inference path while retaining the CVAE objective.

    eval() disables latent sampling and dropout but does not disable gradients.
    The future-action target is not consumed by the model in this mode.
    """
    normalized = policy.normalize_inputs(batch)
    normalized = policy.normalize_targets(normalized)
    normalized = dict(normalized)
    normalized[OBS_IMAGES] = [normalized[key] for key in policy.config.image_features]
    was_training = policy.model.training
    policy.model.eval()
    try:
        predicted = policy.model(normalized)[0]
    finally:
        policy.model.train(was_training)
    valid = ~normalized['action_is_pad'].unsqueeze(-1)
    loss = ((predicted-normalized['action']).abs() * valid).mean()
    return loss, predicted


@torch.inference_mode()
def validate(policy, loader):
    policy.eval()
    absolute_error = 0.0
    count = 0
    for raw in loader:
        batch = device_batch(raw)
        inputs = {key: value for key, value in batch.items() if key.startswith('observation.')}
        predicted = policy.predict_action_chunk(inputs)
        mask = ~batch['action_is_pad'].unsqueeze(-1)
        absolute_error += ((predicted - batch['action']).abs() * mask).sum().item()
        count += int(mask.sum()) * predicted.shape[-1]
    policy.train()
    return absolute_error / count


def save_checkpoint(policy, optimizer, step, directory, metrics):
    directory.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(directory)
    # Optimizer state is for local recovery and excluded from the released model.
    torch.save({'optimizer': optimizer.state_dict(), 'step': step,
                'rng_state': torch.get_rng_state(), 'cuda_rng_state': torch.cuda.get_rng_state()},
               directory / 'training_state.pt')
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--validation-freq', type=int, default=250)
    parser.add_argument('--validation-samples', type=int, default=160)
    parser.add_argument('--prior-weight', type=float, default=0.0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expected exactly one Slurm-visible GPU')
    if args.steps < 1 or args.batch_size < 1 or args.validation_freq < 1:
        raise ValueError('Invalid training size')
    if args.prior_weight < 0:
        raise ValueError('prior-weight must be nonnegative')
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory; existing training is preserved')
    args.output_dir.mkdir(parents=True)
    seed = 20261012
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    split = json.loads((args.dataset_root / 'split.json').read_text())
    info = json.loads((args.dataset_root / 'meta/info.json').read_text())
    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    config.device = 'cuda'
    config.pretrained_backbone_weights = None
    config.optimizer_lr = args.lr
    config.optimizer_lr_backbone = args.lr
    config.push_to_hub = False
    policy = ACTPolicy.from_pretrained(args.checkpoint, config=config, strict=True).to('cuda')
    initial_weight_hash = hashlib.sha256((args.checkpoint / 'model.safetensors').read_bytes()).hexdigest()
    deltas = {'action': [offset / info['fps'] for offset in range(config.chunk_size)]}
    def dataset(episodes):
        return ModelDataset(LeRobotDataset(split['repo_id'], root=args.dataset_root,
                            episodes=episodes, revision=split['revision'],
                            delta_timestamps=deltas, video_backend='pyav'))
    training = dataset(split['train_episodes'])
    validation = dataset(split['validation_episodes'])
    indices = np.linspace(0, len(validation) - 1, min(args.validation_samples, len(validation)), dtype=int).tolist()
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                         pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(training, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed), **loader_kwargs)
    validation_loader = DataLoader(Subset(validation, indices), shuffle=False, **loader_kwargs)
    if len(train_loader) == 0:
        raise RuntimeError('Training data smaller than one batch')
    optimizer = torch.optim.AdamW(policy.get_optim_params(), lr=args.lr,
                                 weight_decay=config.optimizer_weight_decay)
    parameter_name, parameter = next((name, value) for name, value in policy.named_parameters()
                                      if name.startswith('model.action_head') and value.requires_grad)
    initial_parameter = parameter.detach().clone()
    normalization_before = {name: value.detach().clone() for name, value in policy.state_dict().items()
                            if name.startswith(('normalize_', 'unnormalize_'))}
    metrics = dict(seed=seed, source_checkpoint=str(args.checkpoint),
                   initial_weight_sha256=initial_weight_hash, dataset=split,
                   train_frames=len(training), validation_frames=len(validation),
                   validation_indices=indices, optimizer='AdamW', lr=args.lr,
                   batch_size=args.batch_size, chunk_size=config.chunk_size,
                   n_action_steps=config.n_action_steps, steps=args.steps,
                   amp='bfloat16', prior_weight=args.prior_weight,
                   objective='cvae_plus_prior_zero_l1' if args.prior_weight else 'cvae',
                   validation_note=split['note'])
    baseline = validate(policy, validation_loader)
    metrics['initial_validation_raw_mae'] = baseline
    metrics['history'] = []
    print(json.dumps({'event': 'initial_validation', 'raw_mae': baseline,
                      'train_frames': len(training), 'validation_samples': len(indices)}), flush=True)
    best = math.inf
    iterator = iter(train_loader)
    started = time.monotonic()
    policy.train()
    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        batch = device_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, loss_parts = policy.forward(batch)
            if args.prior_weight:
                prior_loss, _ = prior_alignment_loss(policy, batch)
                loss_parts['prior_l1_loss'] = float(prior_loss.detach())
                loss = loss + args.prior_weight * prior_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at step {step}')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f'Non-finite gradients at step {step}')
        optimizer.step()
        if step == 1 or step % 50 == 0:
            record = dict(step=step, loss=float(loss), grad_norm=float(grad_norm),
                          seconds=round(time.monotonic() - started, 2), **loss_parts)
            print(json.dumps(record), flush=True)
        if step % args.validation_freq == 0 or step == args.steps:
            mae = validate(policy, validation_loader)
            delta = (parameter.detach() - initial_parameter).abs().max().item()
            if delta == 0:
                raise RuntimeError('Optimizer did not change the monitored model weights')
            for name, original in normalization_before.items():
                if not torch.equal(policy.state_dict()[name], original):
                    raise RuntimeError('Stored checkpoint normalization statistics changed')
            record = dict(step=step, validation_raw_mae=mae,
                          monitored_parameter=parameter_name, max_weight_change=delta)
            metrics['history'].append(record)
            print(json.dumps(record), flush=True)
            metrics['latest_step'] = step
            if mae < best:
                best = mae
                metrics['best_step'] = step
                metrics['best_validation_raw_mae'] = mae
                save_checkpoint(policy, optimizer, step, args.output_dir / 'best', metrics)
            save_checkpoint(policy, optimizer, step, args.output_dir / 'latest', metrics)
            (args.output_dir / 'run_metrics.json').write_text(json.dumps(metrics, indent=2))
    print('Fine-tuning completed. Best checkpoint:', args.output_dir / 'best', flush=True)


if __name__ == '__main__':
    main()
