# MTL-KD baseline

MTL-KD is integrated into the root training and evaluation commands. It uses
this repository's `envs/`, online training generators, `data/` test instances,
reference solutions, `utils.py`, and `checkpointing.py`. No separate upstream
checkout or MTL-KD dataset is required.

```bash
python train.py --model_type MTLKD --gpu_id 0
python test.py --model_type MTLKD --gpu_id 0 \
  --checkpoint results/MTLKD/<run>/epoch-900.pt --problem ALL
```

Both methods use the root `requirements.txt`. See the root README for setup.
The original `MTLKD/` checkout and ZIP, if present locally, are ignored by Git;
the required model sources and upstream MIT license are vendored here.

## Pretrained teachers

The first MTLKD training command automatically downloads missing teachers into
`pretrained/mtlkd/` (about 87 MiB total). These are model weights, not datasets.
They come from the fixed upstream commit recorded in `teachers.json`, and
both file size and SHA-256 are verified before a download is installed. They
are cached outside version control. Evaluation needs only the trained student
checkpoint, so it does not download or load teachers.

To download teachers separately, including before an offline run:

```bash
python -m baselines.mtlkd.assets
python train.py --model_type MTLKD --gpu_id 0 --no_download_teachers
```

Custom teachers can be selected with `--teacher_dir /path/to/teachers` and
`--teacher_size 50` (files named `cvrp50.pt`, `ovrp50.pt`, etc.). Only the six
published 100-node teachers have automatic downloads. Existing custom weights
are loaded strictly and their hashes are recorded in every run configuration.

## Experiment settings

| Setting | Integrated defaults |
| --- | --- |
| Training tasks | CVRP, OVRP, VRPB, VRPL, VRPTW, OVRPTW |
| Task sampling | One uniformly sampled task per instance batch |
| Training data | Fresh instances from the same root environment generators |
| Problem size | 50 customers |
| POMO | 50 requested; shared backhaul environments cap this at 40 |
| Route starts | Depot followed by the shared environment's distinct POMO starts |
| Epochs | MTLKD: 900 (original default); SlotDiff: 5000 |
| Instances per epoch | 20,000 total across tasks |
| Instance batch size | 128 |
| Optimizer | Adam, learning rate 1e-4, weight decay 1e-6 |
| Learning-rate decay | Multiply by 0.1 at milestone 4501 |
| Training seed | 2023 |
| Checkpoints | Every 100 epochs and at the final epoch |
| Validation during training | Disabled, matching the current SlotDiff loop |
| Evaluation | First 1000 instances per task, all 16 variants, batch 100 |
| Test files | Root `data/<PROBLEM>/<problem>50_uniform.pkl` |
| Reference costs | Shared `utils.get_opt_sol_path` files |
| Decoding | Greedy best POMO route, with/without 8-fold augmentation |
| Gap metric | Mean of per-instance percentage gaps |
| Test seed | 2024 |

The MTLKD epoch default is **900**, verified against
[the original launcher](https://github.com/CIAM-Group/MTLKD/blob/a11a9b26a7b7775753a35af4e9f2138d477b9d32/student_model/run.py#L31).
Its original 150,000 instances per epoch correspond to 135 million instances;
the adapted 20,000-instance epochs correspond to 18 million instances at the
900-epoch default. SlotDiff's default budget is 100 million instances. For a
comparison with the same total instance budget, explicitly pass `--epochs 5000`
to MTLKD. Changing the default epoch count does not change the shared learning
rate schedule: the default milestone 4501 is not reached in a 900-epoch run.
The upstream schedule was `--milestones 500 600 700 800 850 --gamma 0.5`, which
can be explicitly selected if that schedule is part of the planned protocol.

`--problem_size`, `--pomo_size`, `--epochs`, `--train_episodes`,
`--train_batch_size`, `--seed`, and optimizer flags use the same root interface.
The shared environments determine all feasibility masks, transitions, and costs.
A shared seed controls the random generators but does not guarantee identical
instances across architectures with different random draws.

MTL-KD retains its one-layer encoder, six-layer decoder, 128-dimensional
embeddings, eight attention heads, no normalization, and teacher-to-student KL
objective. No SlotDiff auxiliary losses are added. The student samples routes,
updates after decisions with `selected_count > 2`, and clips gradient norm to
1.0, as in the upstream implementation. Unlike SlotDiff's per-batch update,
MTL-KD updates per routing decision; equal instance budgets do not imply equal
optimizer steps, FLOPs, or runtime. Cumulative updates are logged.

The published teachers were pretrained at size 100 and work on size 50 inputs.
Their pretraining is additional to the student budget and must be disclosed.
The original upstream protocol used 100 nodes, POMO 1, mixed-task batches,
150,000 instances per epoch, 900 epochs, and a different scheduler. The controls
above are adaptations for this paper's comparison.

The heavy decoder processes eight POMO trajectories at a time by default
(`--pomo_chunk_size 8`). Each chunk's loss is weighted by its trajectory count;
gradients accumulate before a single update per decision. This keeps the full
128-instance batch and all requested POMO starts while fitting 24 GB GPUs.
The tests verify gradient equivalence to the unchunked objective.

## Resume and outputs

`--epochs` always names the final epoch, including completed epochs. To extend
the original 900-epoch default to a 5000-epoch comparison:

```bash
python train.py --model_type MTLKD --gpu_id 0
python train.py --model_type MTLKD --epochs 5000 --gpu_id 0 \
  --checkpoint results/MTLKD/<run>/epoch-900.pt
```

An explicit shorter intermediate stop is also supported:

```bash
python train.py --model_type MTLKD --epochs 2000 --gpu_id 0
python train.py --model_type MTLKD --epochs 5000 --gpu_id 0 \
  --checkpoint results/MTLKD/<run>/epoch-2000.pt
```

The second command runs epochs 2001–5000. Preserve the original experiment
flags, including POMO chunk size, when resuming. Architecture, training controls,
and teacher hashes are checked. Student, optimizer, scheduler, and RNG states
are restored; logs go into a new timestamped directory.

Test results go into `results/test/MTLKD/<run>/` with
`test_results_<PROBLEM>.csv` (same summary columns as SlotDiff) and
`instance_scores_<PROBLEM>.csv`. Evaluation reads architecture from the student
checkpoint. Select another shared data directory with `--data_dir`, or a custom
single-task pickle with `--test_set_path` and `--test_set_opt_sol_path`.
The R3C search and upstream benchmark-specific scripts are separate protocols.

## Upstream attribution

The vendored `student.py` and `teacher.py` are the original model definitions
from [CIAM-Group/MTLKD](https://github.com/CIAM-Group/MTLKD), commit
`a11a9b26a7b7775753a35af4e9f2138d477b9d32`. The original MIT license is retained
in [LICENSE](LICENSE). Training, state adaptation, evaluation, and asset download
in this directory implement the integration with this repository.

Please cite the MTL-KD paper alongside the main method when using this baseline:
Yuepeng Zheng et al., *MTL-KD: Multi-Task Learning via Knowledge Distillation for
Generalizable Neural Vehicle Routing Solver*, NeurIPS 2025.

## Integration verification

A separate Git clone of the candidate source was tested without the legacy
`MTLKD/` directory, ZIP, or pre-existing teacher cache. Using the root
requirements in a separate virtual environment with the available PyTorch
installation, it downloaded and verified all six official teachers, trained and
resumed MTLKD, and evaluated two instances for each of the 16 shared test tasks.
The same clone trained and evaluated SlotDiff. After restoring the original
MTLKD epoch default, the suite includes 37 tests covering both methods' defaults,
explicit epoch overrides, CLI help, training, and resume behavior.
These were short CPU checks of installation and integration; the GPU ablation
jobs continue separately. The commands and logs are retained locally under
`results/integration_checks/`.

The integrated root command was also checked on an NVIDIA RTX A5000 with the
full 128-dimensional, six-decoder-layer model, 50 nodes, batch 128, and POMO 50.
GPU training, checkpoint resume, and evaluation passed; the two training epochs
performed 139 optimizer updates and used about 5 GiB of GPU memory with
trajectory chunking. Commands, process-level GPU samples, and checkpoints are
under `results/gpu_integration_checks/`. This was a short operational check,
not a completed 900-epoch experiment.
