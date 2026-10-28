# Rebuttal ablation run: 2026-10-25

The active study compares `diffusion_slot_mode = normal, shuffled, stopgrad,
noisy_slots` in `SlotDiffModel` with slot diffusion enabled. The independent
variable is the denoiser's slot conditioning/gradient path. Routing score and
losses are tracked during training; held-out evaluation is a separate step.

All modes use the six `Train_ALL` tasks, shared on-the-fly instance generators,
50 customers, POMO 50 (40 in backhaul environments), 16 slots, batch 128,
20,000 instances per epoch, seed 2023, Adam learning rate 1e-4, weight decay
1e-6, and the original learning-rate milestone 4501. Loss weights remain
diffusion 0.1, slot diversity 0.01, and reconstruction 0. The stopping rule is
**epoch 2000**, independent of observed performance. Each model can later resume
to epoch 5000. This is one seed; no conclusions about variance are implied.

Run directory:

```text
results/ablations/rebuttal_2000_20261025_1815/
```

| Mode | GPU | Training log |
| --- | --- | --- |
| normal | 1 | `normal.log` |
| shuffled | 2 | `shuffled.log` |
| stopgrad | 1 | `stopgrad.log` |
| noisy_slots | 2 | `noisy_slots.log` |

GPU 0 was occupied and is not assigned a training job. Slurm commands could not
connect to a configured controller on this host. The launcher runs inside the
detached tmux session `slotdiff_ablation_2000`, with two jobs per free GPU:

```bash
.venv/bin/python -u run_ablation.py \
  --gpus 1 2 --epochs 2000 --jobs-per-gpu 2 \
  --run-dir results/ablations/rebuttal_2000_20261025_1815 \
  --extra '--model_save_interval 10'
```

The launcher is already running; this command documents it and should not be
executed again for the same directory. `status.json` records child PIDs and
running/completed/failed status. `manifest.json` records exact commands and
source hashes; `source_snapshot/` preserves the training sources. Each mode's
timestamped subdirectory contains its actual `config.json`, `train_log.csv`,
and checkpoints. Checkpoints are saved every 10 epochs and at epoch 2000.
The launcher log is next to the run directory with suffix `.launcher.log`.

Progress logs print every 10 batches, including RL, diffusion, and diversity
losses. These runs enable `--strict_aux_losses`: an auxiliary-loss exception
or non-finite total loss fails the process instead of silently omitting a loss.
The launcher continues the other modes and records a failed mode. A successful
exit is considered completed only if its final checkpoint exists.

Before launching the full study, all four modes completed a one-batch preflight
at batch size 128, with real diffusion/diversity losses and final checkpoints.
Those checks are stored under `results/ablations/preflight_20261025_1813/`.
The existing 27 integration tests passed after the training/launcher changes.

To extend an individual completed ablation, preserve its original settings and
mode, and set the final epoch to 5000:

```bash
.venv/bin/python train.py --model_type SlotDiffModel --enable_slot_diffusion \
  --diffusion_slot_mode stopgrad --epochs 5000 --gpu_id 1 \
  --checkpoint results/ablations/rebuttal_2000_20261025_1815/stopgrad/<timestamp>/epoch-2000.pt \
  --model_save_interval 10 --strict_aux_losses --log_interval 10
```
