# RoboSynChallenge ACT with task-specific physical gripper units

This release uses task-specific ACT checkpoints for all ten tasks. Eight normalized-action tasks convert gripper indices 6 and 13 from [0,1] to physical action-space limits after saved output unnormalization. Table rearrangement already outputs metres; ClickBell outputs fixed zero grippers. Arm coordinates and model action queues are unchanged. Every task predicts 50 actions and executes 50 before a new model query; the evaluator adapter returns after at most 10 env steps for progress accounting.

No construction palette seed, object-pose replay, alternate randomization, or success-test modification is installed by this submission. The evaluation setting defaults to `random`. Exact immutable weight revisions and every task's command are in [task_mapping.json](task_mapping.json).

## Environment and commands

Use the organizer's working Linux Python 3.11 simulator environment. Validated simulator stack: DexSim 0.4.3, EmbodiChain commit `89c14da8d0221dc4ee2a4a3e0ba1078b16d447bd`, official evaluator commit `ecbf88ae26bbd8e369eba868e23354c8f64e083f`. DexSim and simulator assets are supplied through the official RoboSynChallenge setup; this repo does not redistribute them.

ACT environment pins are PyTorch 2.7.1+cu128, torchvision 0.22.1, LeRobot 0.3.3 commit `b883328e6c95681ca90a18b102e4ae5e1f91e2bf`, plus [requirements-act.txt](requirements-act.txt). If ACT dependencies are absent, install them into that working simulator environment:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.7.1 torchvision==0.22.1
python -m pip install -r requirements-act.txt
python -m pip install --no-deps https://github.com/huggingface/lerobot/archive/b883328e6c95681ca90a18b102e4ae5e1f91e2bf.zip
bash setup.sh
bash eval.sh drawer_open_place random 0 --max_episodes 20 --seed 0
```

All ten commands follow `bash eval.sh TASK random 0`; see the full mapping below. GPU 0 is a logical ordinal inside the allocation. The inherited CUDA_VISIBLE_DEVICES is preserved. This all-ACT release requires no separate worker, SmolVLA, IPC setup, or site-specific proxy. SYNKA hardware alignment and short-path asset import were external site wrappers; an organizer installation should use its own working CUDA/renderer configuration.

Detailed gripper tracing is off by default. `ROBOSYN_LOCAL_CHECKPOINT_ROOT`, if explicitly set, is an offline override for local verification; normal submission commands resolve the pinned public HF artifacts anonymously.

## Data and training

Public source checkpoints and datasets are from RoboSynChallenge. Exact dataset revisions are in [ten_task_data_inventory.json](ten_task_data_inventory.json). Fine-tuned weights are published with the selected training step, train/validation episode IDs, seed, original weight hash and loss history in each HF task subfolder. Training used 200 demonstration episodes, 50 validation episodes, seed 20261012, batch 8, lr 1e-6 and 2000 steps of the original ACT CVAE objective. Original normalization buffers were retained exactly. `train_click_bell.py` contains the generic training/data wrapper used for the task runs; no new training is performed at evaluation time.

Official private-test scores are unknown. Local evaluations, their settings and limitations are recorded separately; controlled-reset experiments are not represented as untouched official-environment results.

## Final mapping

Table uses our fine-tuned checkpoint. The other nine tasks use official public ACT checkpoints. Corrected Pipette comparisons tied at 7/10; Mixer tied at 3/3; the public models are retained without claiming a fine-tuned advantage. Table's original physical gripper units are unchanged, so its previous matched clear20 comparison remains applicable (both 20/20, selected fine-tune slightly fewer mean steps).

| Task | Checkpoint | Execution length | Run command |
|---|---|---:|---|
| table_rearrangement | [ours, selected step 1500](https://huggingface.co/yukib0123/robosyn-act-alltasks-20261012/tree/4dbacab83136d3b6e277313368547ea6d28f3d31/table_rearrangement) | 50 | `bash eval.sh table_rearrangement random 0` |
| click_bell | [official ACT](https://huggingface.co/RoboSynChallenge/ACT_sim_click_bell/tree/677e65fbb15974024ff840893496197ef7db26d4) | 50 | `bash eval.sh click_bell random 0` |
| water_pouring | [official ACT](https://huggingface.co/RoboSynChallenge/ACT_sim_water_pouring/tree/0bf0fcfc931a69c52871385f28068fcf873cf07a) | 50 | `bash eval.sh water_pouring random 0` |
| handle_basket | [official ACT](https://huggingface.co/RoboSynChallenge/ACT_sim_handle_basket/tree/6d5c70c4bb367e23b43e95cb155f7ad53f30cc79) | 50 | `bash eval.sh handle_basket random 0` |
| items_handover | [official ACT](https://huggingface.co/RoboSynChallenge/ACT_sim_items_handover/tree/86f6ca7f5decd17a92c3218f295de53497383d14) | 50 | `bash eval.sh items_handover random 0` |
| drawer_open_place | [official ACT](https://huggingface.co/RoboSynChallenge/ACT_sim_drawer_open_place/tree/592e30434aad83cf7bd8f1ee105d7c401488743d) | 50 | `bash eval.sh drawer_open_place random 0` |
| mixer_operating | [official ACT](https://huggingface.co/RoboSynChallenge/ACT_sim_mixer_operating/tree/0f12c53a2a6e093ae5e1e28f20480296b45fdf2b) | 50 | `bash eval.sh mixer_operating random 0` |
| item_assembly | [official ACT](https://huggingface.co/RoboSynChallenge/ACT_sim_item_assembly/tree/58955958519157176712a1eff319018d78177ddb) | 50 | `bash eval.sh item_assembly random 0` |
| manipulate_pipette | [official ACT](https://huggingface.co/RoboSynChallenge/ACT_sim_manipulate_pipette/tree/ca013b2e202d594bf347530c61c3ba28c06d1bc2) | 50 | `bash eval.sh manipulate_pipette random 0` |
| sample_loading | [official ACT](https://huggingface.co/RoboSynChallenge/ACT_sim_sample_loading/tree/917bf2796f18abd0a16f723573382161f1a5c5c8) | 50 | `bash eval.sh sample_loading random 0` |

## Verification

* Corrected official Drawer, unchanged gym configuration and no comparison palette/pose wrapper: random20, 18/20 success, job 31877. This is a local H200 result, not a guarantee of private-test success.
* Actual released loader: all ten selected immutable checkpoints strictly reloaded on GPU, predicted finite [1,50,14] chunks from task-specific recorded observations, and produced finite physical-unit actions (31888).
* Actual `eval.sh` path: ClickBell and fine-tuned Table each completed one random episode normally and succeeded (31889 and 31890). External SYNKA wrappers only handled assigned CUDA/Vulkan devices and byte-identical short-path robot assets.
* Anonymous HF accessibility and selected fine-tuned weight hashes checked; policy package exports, Python compilation and shell syntax checked.

See verification/evaluation data in `verification/` and [SUBMISSION.md](SUBMISSION.md) for the exact fields. This final all-ACT configuration has no DP or SmolVLA runtime dependency. Earlier experiments and models remain separate. No form submission is performed by this repository.
