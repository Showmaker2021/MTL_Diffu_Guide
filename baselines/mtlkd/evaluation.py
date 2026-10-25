"""Evaluate MTL-KD on the same held-out instances and metrics as SlotDiff."""

import argparse
import csv
from pathlib import Path

import torch

from .protocol import (
    PROJECT_ROOT, TRAIN_PROBLEMS, ALL_PROBLEMS, MTLKD_model, add_common_args,
    evaluate_batch, make_env, positive_int, setup, write_config,
)
from utils import get_opt_sol_path, load_dataset


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, testing=True)
    parser.add_argument('--data_dir', type=Path, default=PROJECT_ROOT / 'data')
    parser.add_argument('--test_episodes', type=positive_int, default=1000)
    parser.add_argument('--test_batch_size', type=positive_int, default=100)
    parser.add_argument('--pomo_chunk_size', type=positive_int, default=8)
    parser.add_argument('--aug_factor', type=int, choices=(1, 8), default=8)
    parser.add_argument('--test_set_path', type=Path)
    parser.add_argument('--test_set_opt_sol_path', type=Path)
    return parser


def main(argv=None, *, args=None):
    parser = build_parser()
    if args is None:
        args = parser.parse_args(argv)
    if args.test_set_path and args.problem in ('ALL', 'Train_ALL'):
        parser.error('--test_set_path requires one explicit --problem')
    if args.test_set_opt_sol_path and not args.test_set_path:
        parser.error('--test_set_opt_sol_path requires --test_set_path')
    device, log_path = setup(args)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint.get('model_type') != 'MTLKD':
        raise ValueError('Use an MTLKD checkpoint from this repository for evaluation')
    params = dict(checkpoint['model_params'], device=str(device), eval_type='argmax')
    model = MTLKD_model(**params).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    write_config(log_path / 'config.json', dict(vars(args), model_params=params,
                                               checkpoint_epoch=checkpoint['epoch']))
    problems = (ALL_PROBLEMS if args.problem == 'ALL' else TRAIN_PROBLEMS
                if args.problem == 'Train_ALL' else (args.problem,))
    for problem in problems:
        env = make_env(problem, args.problem_size, args.pomo_size, device)
        path = args.test_set_path or args.data_dir / problem / f'{problem.lower()}{args.problem_size}_uniform.pkl'
        opt_path = args.test_set_opt_sol_path
        if args.test_set_path is None:
            opt_path = Path(get_opt_sol_path(str(args.data_dir / problem), problem, args.problem_size))
        # Fail on missing reference files for the standard reporting protocol.
        references = load_dataset(str(opt_path), disable_print=True) if opt_path else None
        if references is not None and len(references) < args.test_episodes:
            raise ValueError(f'{opt_path} has fewer than {args.test_episodes} reference solutions')
        all_scores, all_aug_scores = [], []
        for offset in range(0, args.test_episodes, args.test_batch_size):
            batch_size = min(args.test_batch_size, args.test_episodes - offset)
            data = env.load_dataset(str(path), offset=offset, num_samples=batch_size)
            if data[0].size(0) != batch_size:
                raise ValueError(f'{path} has fewer than {args.test_episodes} instances')
            scores, aug_scores = evaluate_batch(model, data, env, args.aug_factor,
                                                args.pomo_chunk_size)
            all_scores.extend(scores.cpu().tolist())
            all_aug_scores.extend(aug_scores.cpu().tolist())
        mean_score = sum(all_scores) / len(all_scores)
        mean_aug_score = sum(all_aug_scores) / len(all_aug_scores)
        gap = aug_gap = None
        if references is not None:
            # Mean per-instance relative gap, matching the parent Tester.py.
            costs = [solution[0] for solution in references[:args.test_episodes]]
            gap = sum((score - cost) / cost * 100 for score, cost in zip(all_scores, costs)) / len(costs)
            aug_gap = sum((score - cost) / cost * 100 for score, cost in zip(all_aug_scores, costs)) / len(costs)
        result_path = log_path / f'test_results_{problem}.csv'
        with result_path.open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'problem', 'problem_size', 'no_aug_score',
                             'no_aug_gap', 'aug_score', 'aug_gap'])
            writer.writerow([log_path.name, problem, args.problem_size, mean_score,
                             gap, mean_aug_score, aug_gap])
        with (log_path / f'instance_scores_{problem}.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['instance', 'no_aug_score', 'aug_score'])
            writer.writerows((i, score, aug) for i, (score, aug) in
                             enumerate(zip(all_scores, all_aug_scores)))
        print(f'{problem}{args.problem_size}: score={mean_score:.4f}, '
              f'gap={gap}, aug_score={mean_aug_score:.4f}, aug_gap={aug_gap}', flush=True)
        print(f'Results: {result_path}', flush=True)
    return log_path


if __name__ == '__main__':
    main()
