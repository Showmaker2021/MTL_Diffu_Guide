"""Integration checks for the adapted data, rollout and resume protocol."""

import random
from copy import deepcopy

import numpy as np
import pytest
import torch

from checkpointing import capture_rng_state, restore_training_state
from baselines.mtlkd.protocol import (
    ALL_PROBLEMS, TRAIN_PROBLEMS, MTLKD_model,
    distillation_loss, evaluate_batch, load_teachers, make_env, pomo_chunks,
    prepare_student, seed_everything, student_step,
)
from baselines.mtlkd.training import build_parser, main, train_batch
from baselines.mtlkd.teacher import SINGLEModel


@pytest.fixture(autouse=True)
def cpu_execution():
    previous_device = torch.get_default_device()
    previous_threads = torch.get_num_threads()
    torch.set_default_device('cpu')
    torch.set_num_threads(1)
    seed_everything(2023)
    yield
    torch.set_default_device(previous_device)
    torch.set_num_threads(previous_threads)


@pytest.fixture(scope='session')
def teacher_dir(tmp_path_factory):
    # Synthetic teacher weights keep these integration tests offline. Actual
    # upstream downloads and checkpoints are verified by the clean-copy smoke run.
    directory = tmp_path_factory.mktemp('synthetic_teachers')
    for problem in TRAIN_PROBLEMS:
        model = SINGLEModel(problem=problem, eval_type='argmax', norm_loc='norm_last',
                            norm='instance', embedding_dim=128, sqrt_embedding_dim=128 ** 0.5,
                            encoder_layer_num=6, decoder_layer_num=1, qkv_dim=16,
                            head_num=8, logit_clipping=10, ff_hidden_dim=512, device='cpu')
        torch.save(dict(problem=problem, epoch=0, model_state_dict=model.state_dict()),
                   directory / f'{problem.lower()}100.pt')
    return directory


def small_model():
    return MTLKD_model(embedding_dim=16, sqrt_embedding_dim=4,
                       encoder_layer_num=1, decoder_layer_num=1,
                       qkv_dim=8, head_num=2, logit_clipping=10,
                       ff_hidden_dim=32, eval_type='softmax', norm=None,
                       device='cpu')


@pytest.mark.parametrize('problem', ALL_PROBLEMS)
def test_shared_environment_rollout(problem):
    env = make_env(problem, 20, 3, 'cpu')
    data = env.get_random_problems(2, 20)
    scores, augmented = evaluate_batch(small_model(), data, env, 8)
    assert torch.isfinite(scores).all()
    assert torch.isfinite(augmented).all()
    assert (augmented <= scores + 1e-5).all()
    assert torch.isneginf(env.visited_ninf_flag[:, :, 1:]).all()
    torch.testing.assert_close(
        env.selected_node_list[:, :, 1], env.step_state.START_NODE,
    )
    # Costs are computed by the very same environment used by SlotDiff.
    rewards = -env._get_travel_distance().reshape(8, 2, env.pomo_size)
    torch.testing.assert_close(scores, -rewards[0].max(dim=1).values)
    torch.testing.assert_close(augmented, -rewards.amax(dim=(0, 2)))


def test_features_and_candidate_padding():
    env = make_env('VRPTW', 20, 3, 'cpu')
    env.load_problems(2, problems=env.get_random_problems(2, 20))
    reset_state, _, _ = env.reset()
    model = small_model()
    reset = prepare_student(model, reset_state)
    torch.testing.assert_close(reset.depot_node_feat[:, 0, :2], reset_state.depot_xy[:, 0])
    assert (reset.depot_node_feat[:, 0, 2:] == 0).all()
    torch.testing.assert_close(reset.depot_node_feat[:, 1:, 3], reset_state.node_service_time)
    torch.testing.assert_close(reset.depot_node_feat[:, 1:, 4], reset_state.node_tw_start)
    torch.testing.assert_close(reset.depot_node_feat[:, 1:, 5], reset_state.node_tw_end)
    state, _, _ = env.pre_step()
    for _ in range(4):
        selected, candidates, probabilities = student_step(model, reset, state, env)
        if candidates is not None:
            # Every probability is over a feasible real node; padded depots have no mass.
            mask = state.ninf_mask.gather(2, candidates)
            assert (probabilities[torch.isneginf(mask)] == 0).all()
            assert (probabilities[:, :, 1:][candidates[:, :, 1:] == 0] == 0).all()
        state, _, _ = env.step(selected)


@pytest.mark.parametrize('problem', TRAIN_PROBLEMS)
def test_train_with_teacher(problem, teacher_dir):
    teacher = load_teachers((problem,), teacher_dir, 100, 'cpu', download=False)[problem]
    model = small_model()
    before = model.encoder.embedding.weight.detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    env = make_env(problem, 20, 3, 'cpu')
    score, loss, updates = train_batch(model, teacher, env,
                                      env.get_random_problems(2, 20), optimizer,
                                      pomo_chunk_size=2)
    assert np.isfinite(score) and np.isfinite(loss)
    assert updates > 0
    assert not torch.equal(before, model.encoder.embedding.weight)
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_chunked_gradients_match_full_pomo_gradients(teacher_dir):
    teacher = load_teachers(('CVRP',), teacher_dir, 100, 'cpu', download=False)['CVRP']
    full_model = small_model()
    full_model.set_eval_type('argmax')
    chunk_model = deepcopy(full_model)
    env = make_env('CVRP', 20, 3, 'cpu')
    env.load_problems(2, problems=env.get_random_problems(2, 20))
    reset_state, _, _ = env.reset()
    with torch.no_grad():
        teacher.pre_forward(reset_state)
    full_reset = prepare_student(full_model, reset_state)
    chunk_reset = prepare_student(chunk_model, reset_state)
    state, _, _ = env.pre_step()
    with torch.no_grad():
        for _ in range(3):
            selected, _, _ = student_step(full_model, full_reset, state, env)
            state, _, _ = env.step(selected)
    selected, candidates, probabilities = student_step(full_model, full_reset, state, env)
    full_loss = distillation_loss(teacher, state, selected, candidates, probabilities)
    full_loss.backward()
    chunked_loss = 0.0
    for chunk in pomo_chunks(state, env, 2):
        selected, candidates, probabilities = student_step(chunk_model, chunk_reset, chunk, env)
        loss = distillation_loss(teacher, chunk, selected, candidates, probabilities)
        weighted = loss * selected.size(1) / env.pomo_size
        weighted.backward()
        chunked_loss += weighted.item()
    assert chunked_loss == pytest.approx(full_loss.item(), rel=1e-6)
    for full_param, chunk_param in zip(full_model.parameters(), chunk_model.parameters()):
        torch.testing.assert_close(full_param.grad, chunk_param.grad, rtol=1e-4, atol=1e-6)


def test_checkpoint_restores_scheduler_and_all_random_generators():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[1, 3], gamma=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
    checkpoint = dict(optimizer_state_dict=optimizer.state_dict(),
                      scheduler_state_dict=scheduler.state_dict(),
                      rng_state=capture_rng_state())
    expected_random = (random.random(), np.random.rand(), torch.rand(3))
    restored_optimizer = torch.optim.Adam(model.parameters(), lr=0.2)
    restored_scheduler = torch.optim.lr_scheduler.MultiStepLR(restored_optimizer, [99])
    restore_training_state(checkpoint, restored_optimizer, restored_scheduler)
    assert random.random() == expected_random[0]
    assert np.random.rand() == expected_random[1]
    torch.testing.assert_close(torch.rand(3), expected_random[2])
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    restored_optimizer.step()
    restored_scheduler.step()
    assert restored_optimizer.param_groups[0]['lr'] == pytest.approx(1e-6)


def test_resumed_training_matches_uninterrupted_training(tmp_path, teacher_dir):
    common = ['--no_cuda', '--teacher_dir', str(teacher_dir), '--no_download_teachers',
              '--problem', 'CVRP', '--problem_size', '20',
              '--pomo_size', '2', '--train_episodes', '3', '--train_batch_size', '2',
              '--embedding_dim', '16', '--decoder_layer_num', '1', '--head_num', '2',
              '--qkv_dim', '8', '--ff_hidden_dim', '32', '--milestones', '1',
              '--pomo_chunk_size', '1']
    first = main(common + ['--epochs', '1', '--log_dir', str(tmp_path / 'split')])
    resumed = main(common + ['--epochs', '2', '--log_dir', str(tmp_path / 'resumed'),
                             '--checkpoint', str(first / 'epoch-1.pt')])
    whole = main(common + ['--epochs', '2', '--log_dir', str(tmp_path / 'whole')])
    split_checkpoint = torch.load(resumed / 'epoch-2.pt', weights_only=False)
    full_checkpoint = torch.load(whole / 'epoch-2.pt', weights_only=False)
    assert split_checkpoint['optimizer_updates'] == full_checkpoint['optimizer_updates']
    assert split_checkpoint['scheduler_state_dict'] == full_checkpoint['scheduler_state_dict']
    for key, value in full_checkpoint['model_state_dict'].items():
        torch.testing.assert_close(split_checkpoint['model_state_dict'][key], value, rtol=0, atol=0)


def test_training_budget_defaults():
    args = build_parser().parse_args([])
    assert (args.epochs, args.problem_size, args.pomo_size) == (900, 50, 50)
    assert (args.train_episodes, args.train_batch_size, args.seed) == (20000, 128, 2023)
    assert args.milestones == [4501]
