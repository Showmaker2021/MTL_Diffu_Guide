"""Train the MTL-KD baseline using SlotDiff's experiment protocol."""

import argparse
import csv
import random
from pathlib import Path

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from checkpointing import capture_rng_state, restore_training_state
from .protocol import (
    DEFAULT_TEACHER_DIR, TRAIN_PROBLEMS, MTLKD_model, add_common_args,
    distillation_loss, load_teachers, make_env, pomo_chunks, positive_int, prepare_student,
    setup, student_params, student_step, write_config,
)
from utils import AverageMeter, TimeEstimator


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument('--epochs', type=positive_int, default=900,
                        help='final epoch, including epochs already in a resumed checkpoint')
    parser.add_argument('--train_episodes', type=positive_int, default=20000)
    parser.add_argument('--train_batch_size', type=positive_int, default=128)
    parser.add_argument('--pomo_chunk_size', type=positive_int, default=8,
                        help='trajectories processed at once; gradients accumulate before each decision update')
    parser.add_argument('--model_save_interval', type=positive_int, default=100)
    parser.add_argument('--log_interval', type=positive_int, default=10,
                        help='print progress every this many instance batches')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--milestones', type=positive_int, nargs='+', default=[4501])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--embedding_dim', type=positive_int, default=128)
    parser.add_argument('--decoder_layer_num', type=positive_int, default=6)
    parser.add_argument('--qkv_dim', type=positive_int, default=16)
    parser.add_argument('--head_num', type=positive_int, default=8)
    parser.add_argument('--ff_hidden_dim', type=positive_int, default=512)
    parser.add_argument('--teacher_dir', type=Path, default=DEFAULT_TEACHER_DIR)
    parser.add_argument('--teacher_size', type=positive_int, default=100,
                        help='teacher filename suffix; bundled checkpoints were trained at size 100')
    parser.add_argument('--no_download_teachers', action='store_true')
    return parser


def train_batch(model, teacher, env, data, optimizer, pomo_chunk_size=8):
    model.train()
    model.set_eval_type('softmax')
    env.load_problems(data[0].size(0), problems=data, aug_factor=1)
    reset_state, _, _ = env.reset()
    reset = prepare_student(model, reset_state)
    with torch.no_grad():
        teacher.pre_forward(reset_state)
    state, reward, done = env.pre_step()
    loss_sum, updates = 0.0, 0
    while not done:
        optimizer.zero_grad(set_to_none=True)
        selected_chunks, decision_loss = [], 0.0
        for chunk in pomo_chunks(state, env, pomo_chunk_size):
            selected, candidates, probabilities = student_step(model, reset, chunk, env)
            # Average the gradients over all POMO trajectories before updating.
            if state.selected_count > 2:
                loss = distillation_loss(teacher, chunk, selected, candidates, probabilities)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite KD loss on {env.problem}')
                weighted_loss = loss * (selected.size(1) / env.pomo_size)
                weighted_loss.backward()
                decision_loss += weighted_loss.item()
            selected_chunks.append(selected.detach())
        # Preserve upstream's per-decision KL updates and gradient clipping.
        if state.selected_count > 2:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += decision_loss
            updates += 1
        state, reward, done = env.step(torch.cat(selected_chunks, dim=1))
    score = -reward.max(dim=1).values.float().mean().item()
    return score, loss_sum / max(updates, 1), updates


def validate_resume(checkpoint, args, params, teachers):
    if checkpoint.get('model_type') != 'MTLKD':
        raise ValueError('Use an MTLKD checkpoint from this repository to resume this protocol')
    previous = checkpoint['config']
    # Epoch limit, save interval, logging location and GPU may change on resume.
    controls = ('problem', 'problem_size', 'pomo_size', 'train_episodes',
                'train_batch_size', 'pomo_chunk_size', 'seed', 'lr', 'weight_decay', 'milestones',
                'gamma', 'teacher_size')
    for key in controls:
        if previous[key] != getattr(args, key):
            raise ValueError(f'Resume setting {key} differs: {previous[key]} vs {getattr(args, key)}')
    for key, value in checkpoint['model_params'].items():
        if key != 'device' and params[key] != value:
            raise ValueError(f'Resume model setting {key} differs')
    for problem, teacher in teachers.items():
        old = previous['teacher_checkpoints'][problem]
        if old['sha256'] != teacher.checkpoint_metadata['sha256']:
            raise ValueError(f'Resume teacher checkpoint differs for {problem}')


def main(argv=None, *, args=None):
    if args is None:
        args = build_parser().parse_args(argv)
    device, log_path = setup(args)
    problems = TRAIN_PROBLEMS if args.problem == 'Train_ALL' else (args.problem,)
    params = student_params(args, device)
    teachers = load_teachers(problems, args.teacher_dir.resolve(), args.teacher_size, device,
                             download=not args.no_download_teachers)
    model = MTLKD_model(**params).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    start_epoch, total_updates = 1, 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        validate_resume(checkpoint, args, params, teachers)
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        restore_training_state(checkpoint, optimizer, scheduler)
        start_epoch = checkpoint['epoch'] + 1
        total_updates = checkpoint['optimizer_updates']
        if args.epochs < start_epoch:
            raise ValueError(f'--epochs must be >= {start_epoch} to continue this checkpoint')
        print(f'Resuming from epoch {start_epoch}, LR {optimizer.param_groups[0]["lr"]}')
    config = dict(vars(args))
    config.update(model_type='MTLKD', model_params=params,
                  train_problems=problems, device=str(device),
                  data_protocol='SlotDiff shared envs; fresh instances; one task per batch',
                  teacher_checkpoints={p: teachers[p].checkpoint_metadata for p in problems},
                  teacher_paths={p: str(args.teacher_dir.resolve() / f'{p.lower()}{args.teacher_size}.pt')
                                 for p in problems})
    write_config(log_path / 'config.json', config)
    print(f'Log path: {log_path}', flush=True)
    print(f'Training {problems}, size {args.problem_size}, POMO {args.pomo_size}, '
          f'{args.train_episodes} instances/epoch, batch {args.train_batch_size}, '
          f'final epoch {args.epochs}', flush=True)
    timer = TimeEstimator()
    timer.reset(start_epoch)
    with (log_path / 'train_log.csv').open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epoch', 'score', 'loss', 'lr', 'optimizer_updates'])
        for epoch in range(start_epoch, args.epochs + 1):
            score_meter, loss_meter = AverageMeter(), AverageMeter()
            episode, batch_index = 0, 0
            while episode < args.train_episodes:
                batch_size = min(args.train_batch_size, args.train_episodes - episode)
                problem = random.sample(problems, 1)[0]
                env = make_env(problem, args.problem_size, args.pomo_size, device)
                data = env.get_random_problems(batch_size, args.problem_size)
                score, loss, updates = train_batch(model, teachers[problem], env, data,
                                                   optimizer, args.pomo_chunk_size)
                score_meter.update(score, batch_size)
                loss_meter.update(loss, batch_size)
                total_updates += updates
                episode += batch_size
                batch_index += 1
                if batch_index == 1 or batch_index % args.log_interval == 0 or episode == args.train_episodes:
                    print(f'Epoch {epoch}: {episode}/{args.train_episodes} instances, '
                          f'task={problem}, score={score:.4f}, KD loss={loss:.6f}, '
                          f'optimizer updates={total_updates}', flush=True)
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            writer.writerow([epoch, score_meter.avg, loss_meter.avg, lr, total_updates])
            file.flush()
            elapsed, remaining = timer.get_est_string(epoch, args.epochs)
            print(f'Epoch {epoch}/{args.epochs}: score={score_meter.avg:.4f}, '
                  f'KD loss={loss_meter.avg:.6f}, LR={lr:g}, '
                  f'elapsed={elapsed}, remaining={remaining}', flush=True)
            if epoch % args.model_save_interval == 0 or epoch == args.epochs:
                checkpoint = dict(epoch=epoch, problem=args.problem, model_type='MTLKD',
                                  model_params=params, config=config,
                                  model_state_dict=model.state_dict(),
                                  optimizer_state_dict=optimizer.state_dict(),
                                  scheduler_state_dict=scheduler.state_dict(),
                                  rng_state=capture_rng_state(), optimizer_updates=total_updates)
                torch.save(checkpoint, log_path / f'epoch-{epoch}.pt')
    return log_path


if __name__ == '__main__':
    main()
