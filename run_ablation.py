"""Launch SlotDiff studies with per-mode commands, logs, and exit status.

Run inside tmux for long-lived jobs. A failed mode does not stop the other modes.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
MODES = ('normal', 'shuffled', 'stopgrad', 'noisy_slots')


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--epochs', type=positive_int, default=5000)
    parser.add_argument('--modes', choices=MODES, nargs='+', default=list(MODES))
    parser.add_argument('--jobs-per-gpu', type=positive_int, default=2)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--extra', default='', help='additional train.py flags, parsed with shlex')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if any(gpu < 0 for gpu in args.gpus) or len(set(args.gpus)) != len(args.gpus):
        parser.error('--gpus requires distinct nonnegative GPU indices')
    if len(set(args.modes)) != len(args.modes):
        parser.error('--modes must be distinct')
    run_dir = (args.run_dir or ROOT / 'results' / 'ablations' /
               datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    jobs = []
    for index, mode in enumerate(args.modes):
        gpu = args.gpus[index % len(args.gpus)]
        command = [args.python, '-u', str(ROOT / 'train.py'),
                   '--model_type', 'SlotDiffModel', '--enable_slot_diffusion',
                   '--diffusion_slot_mode', mode, '--epochs', str(args.epochs),
                   '--gpu_id', str(gpu), '--log_dir', str(run_dir / mode),
                   '--strict_aux_losses', '--log_interval', '10', *shlex.split(args.extra)]
        jobs.append(dict(mode=mode, gpu=gpu, command=command, status='pending',
                         log=str(run_dir / f'{mode}.log')))
        print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    if run_dir.exists() and any(run_dir.iterdir()):
        parser.error(f'Run directory is not empty: {run_dir}')
    run_dir.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / name for name in ('train.py', 'Trainer.py', 'utils.py', 'checkpointing.py')]
    sources += list((ROOT / 'models').glob('*.py')) + list((ROOT / 'envs').glob('*.py'))
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        destination = run_dir / 'source_snapshot' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    cwd=str(ROOT), python=args.python, epochs=args.epochs,
                    training_defaults=dict(seed=2023, training_instances_per_epoch=20000,
                                           train_batch_size=128, problem_size=50, pomo_size=50),
                    configuration_note='Commands and each training config.json include any --extra overrides.',
                    source_sha256=hashes, jobs=jobs)
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    processes = {}

    def save_status():
        temporary = run_dir / 'status.json.tmp'
        temporary.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                                             launcher_pid=os.getpid(), jobs=jobs), indent=2) + '\n')
        temporary.replace(run_dir / 'status.json')

    while any(job['status'] in ('pending', 'running') for job in jobs):
        for job in jobs:
            if job['status'] == 'running':
                process, log = processes[job['mode']]
                code = process.poll()
                if code is not None:
                    log.close()
                    checkpoints = list((run_dir / job['mode']).glob(f'*/epoch-{args.epochs}.pt'))
                    job.update(status='completed' if code == 0 and checkpoints else 'failed',
                               exit_code=code, finished_utc=datetime.now(timezone.utc).isoformat(),
                               final_checkpoints=[str(path) for path in checkpoints])
                    print(f"[{job['mode']}] {job['status']} (exit {code})", flush=True)
        for job in jobs:
            if job['status'] != 'pending':
                continue
            occupied = sum(other['status'] == 'running' and other['gpu'] == job['gpu'] for other in jobs)
            if occupied >= args.jobs_per_gpu:
                continue
            log = open(job['log'], 'w')
            process = subprocess.Popen(job['command'], cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT)
            processes[job['mode']] = (process, log)
            job.update(status='running', pid=process.pid,
                       started_utc=datetime.now(timezone.utc).isoformat())
            print(f"[{job['mode']}] PID {process.pid} GPU {job['gpu']} log {job['log']}", flush=True)
        save_status()
        if any(job['status'] == 'running' for job in jobs):
            time.sleep(15)
    save_status()
    return 1 if any(job['status'] == 'failed' for job in jobs) else 0


if __name__ == '__main__':
    raise SystemExit(main())
