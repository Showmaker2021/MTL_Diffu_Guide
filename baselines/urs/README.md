# Integrated URS baseline

Train and evaluate through the same root commands as SlotDiff and MTLKD:

```bash
pip install -r requirements.txt
python train.py --model_type URS --gpu_id 0
python test.py --model_type URS --gpu_id 0 \
  --checkpoint results/URS/<run>/epoch-500.pt --problem ALL
```

URS trains from scratch; no teacher, pretrained model, external dataset, or
separate upstream checkout is needed. The root environments generate the six
training tasks and provide all 16 test variants. Evaluation reads this
repository's `data/` and reference solutions through the shared `Tester.py`.
The original local `URS/` checkout is ignored by the outer Git repository.

## Original defaults and adaptation

The original generalist training default is **500 epochs**, not the 300-epoch
specialist checkpoint names shown in the upstream evaluation examples. Defaults
were verified in the pinned upstream
[args.py](https://github.com/CIAM-Group/URS/blob/69c6766219e40d18fd8b371fedfc208241e5be9e/args.py).

| Setting | Integrated URS default |
| --- | --- |
| Epochs | 500 |
| Encoder | 12 layers, embedding 128, feed-forward width 512 |
| Decoder | Upstream adaptive parameter generation and mixed-bias modules |
| Attention | Original adaptive attention-free modules; no replacement Transformer |
| Node/problem features | Original unified representation and 13-dimensional problem codes |
| Logit clipping | 50 |
| Normalization | Instance normalization after residuals |
| Demand normalization | Divide by each instance's maximum demand, as upstream |
| Optimizer | AdamW, learning rate 1e-4, weight decay 1e-6 |
| Learning-rate milestone | 451, multiplying by 0.1 |
| Training tasks | CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW |
| Data | Fresh root `envs/` instances, 50 customers |
| Training budget | 20,000 instances per epoch, batch 128 |
| POMO | 50 requested; shared backhaul environments cap at 40 |
| Training seed | 2023 |
| Checkpoint interval | 100 epochs and the final epoch |
| Validation/model selection | No training-time validation, matching current SlotDiff |
| Test protocol | First 1000 instances per task, best POMO, greedy decoding |
| Augmentation | Same 8-fold coordinate transforms, also reporting unaugmented score |
| Gap | Mean per-instance percentage gap against the shared reference files |

Only the model and its feature adaptation are vendored. Shared root environments
determine every action mask, transition, stopping condition, and route cost.
The adapter converts the root `[C,O,B,L,TW]` flags to URS names such as `cvrpb`
and `ocvrpltw`, and maps coordinates, signed normalized demands, time windows,
and service times into the upstream nine-field input format. It retains the
original demand rescaling, distance normalization, hypernetwork, and attention
formulas. Length limits and time-window feasibility remain enforced by the shared
masks; no new dynamic inputs are added to the original model. The shared forced
customer start is also used for POMO=1, whereas upstream samples that case.

The objective is the original POMO REINFORCE loss with a within-instance rollout
baseline and one optimizer update per instance batch. No slot or diffusion losses
are added. Greedy inference maps the root `argmax` setting to upstream `greedy`;
training always uses upstream `sampling`. Slot-consistency diagnostics run only
for models that actually have slot attention.

The upstream generalist trained on 11 tasks, 100 nodes, and 2000 batches of 128
instances per epoch (128 million instances over 500 epochs), with validation
every epoch. This adaptation uses the requested shared task/data protocol:
500 epochs of 20,000 instances is 10 million instances. SlotDiff defaults to
100 million instances and MTLKD to 18 million. These are different total budgets.
To match SlotDiff's total instances and scheduler, explicitly use
`--epochs 5000 --milestones 4501`; to reproduce upstream's epoch size on the
shared tasks, use `--train_episodes 256000`. Do not compare defaults as if the
complete original training protocols were identical.

The shared scheduler steps after each epoch: milestone 451 changes the rate
for the following epoch. Upstream's manual decay ran at the start of epoch
451; use `--milestones 450` to match that exact first reduced-rate epoch.

## Overrides, evaluation and resume

The common `--epochs`, `--problem`, `--problem_size`, `--pomo_size`,
`--train_episodes`, `--train_batch_size`, `--seed`, `--lr`, `--weight_decay`,
`--milestones`, `--gamma`, and `--model_save_interval` flags are supported.
Use `--no_demand_max1` to disable the optional upstream demand rescaling.
Architecture changes such as a smaller encoder must be repeated at evaluation,
as for SlotDiff. Evaluation requires only the student checkpoint.

```bash
python train.py --model_type URS --epochs 300 --gpu_id 0
python train.py --model_type URS --epochs 500 --gpu_id 0 \
  --checkpoint results/URS/<run>/epoch-300.pt
```

The second command runs epochs 301–500. Preserve the same architecture and
experiment flags. The shared Trainer restores model, optimizer, scheduler, and
RNG states. Checkpoints and training CSVs go to `results/URS/<run>/`; test CSVs
go to `results/test/URS/<run>/`. For a quick CPU smoke check, use `--no_cuda
--epochs 1 --train_episodes 2 --train_batch_size 2 --pomo_size 3`.

## Attribution and source

Vendored files come from [CIAM-Group/URS](https://github.com/CIAM-Group/URS),
commit `69c6766219e40d18fd8b371fedfc208241e5be9e`. The original copyright headers
and [LICENSE](LICENSE) are retained. The model sources are changed only to use
package-relative imports; `models/URSModel.py` implements the shared-state
adapter. `upstream.json` records source hashes and original defaults.

Please cite Changliang Zhou et al., *URS: A Unified Neural Routing Solver for
Cross-Problem Zero-Shot Generalization*, ICML 2026, when using this baseline.

## Verification

Tests cover all 16 shared tasks, equality of the adapted greedy policy with
the vendored upstream model, exact shared route costs and POMO starts, and
gradient/AdamW updates on all six training tasks. Public CLI tests check the
500-epoch default, original architecture/optimizer defaults, and epoch overrides.
Operational verification logs and checkpoints are stored locally under
`results/urs_checks/`; smoke checks are not completed paper experiments.

Completed verification: **63 tests passed** in the workspace and an independent
Git clone without either upstream checkout. The root command trained and resumed
the full model on an NVIDIA RTX A5000 and evaluated two instances for each of all
16 shared tasks with 8-fold augmentation. A further full GPU epoch processed
20,000 instances at batch 128/POMO 50 across all six tasks, performed 157 AdamW
updates, and saved a checkpoint with finite weights. Observed process memory
peaked at 4626 MiB. `results/urs_checks/verification.json` records commands,
checkpoints, exit codes, and CUDA telemetry. No full 500-epoch study was launched.
