"""URS integration: upstream policy equivalence, shared costs and real updates."""

from types import SimpleNamespace

import pytest
import torch

from Trainer import Trainer
from baselines.urs.model import Model
from models.URSModel import URSModel
from utils import get_env, seed_everything


EXPECTED_NAMES = {
    'CVRP': 'cvrp', 'OVRP': 'ocvrp', 'VRPB': 'cvrpb', 'VRPL': 'cvrpl',
    'VRPTW': 'cvrptw', 'OVRPTW': 'ocvrptw', 'OVRPB': 'ocvrpb',
    'OVRPL': 'ocvrpl', 'VRPBL': 'cvrpbl', 'VRPBTW': 'cvrpbtw',
    'VRPLTW': 'cvrpltw', 'OVRPBL': 'ocvrpbl', 'OVRPBTW': 'ocvrpbtw',
    'OVRPLTW': 'ocvrpltw', 'VRPBLTW': 'cvrpbltw', 'OVRPBLTW': 'ocvrpbltw',
}


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


def params():
    return dict(embedding_dim=16, encoder_layer_num=2, ff_hidden_dim=32,
                logit_clipping=50, eval_type='argmax', demand_max1=True, device='cpu')


@pytest.mark.parametrize('problem', EXPECTED_NAMES)
def test_shared_rollout_matches_upstream_policy(problem):
    env = get_env(problem)[0](problem_size=20, pomo_size=3, device='cpu')
    data = env.get_random_problems(2, 20)
    env.load_problems(2, problems=data, aug_factor=8)
    reset, _, _ = env.reset()
    adapted = URSModel(**params()).eval()
    reference = Model(**dict(params(), eval_type='greedy')).eval()
    reference.load_state_dict(adapted.state_dict(), strict=True)
    with torch.no_grad():
        adapted.pre_forward(reset)
        assert adapted.problem_name == EXPECTED_NAMES[problem]
        assert adapted.problem_representation.numel() == 13
        raw = adapted.last_unified_state.problems
        torch.testing.assert_close(raw[:, 1:, 2], reset.node_demand)
        torch.testing.assert_close(raw[:, 1:, 6], reset.node_tw_start)
        torch.testing.assert_close(raw[:, 1:, 7], reset.node_tw_end)
        torch.testing.assert_close(raw[:, 1:, 8], reset.node_service_time)
        reference.decoder.assign(adapted.problem_representation)
        reference.pre_forward(adapted.last_unified_state, adapted.problem_name,
                              adapted.problem_representation)
        state, reward, done = env.pre_step()
        steps = 0
        while not done:
            selected, _ = adapted(state)
            native_state = SimpleNamespace(**vars(state), batch_size=env.batch_size,
                                           pomo_size=env.pomo_size, depot_num=1)
            distance = (adapted.shared_distance.gather(1, state.current_node[:, :, None].expand(
                        -1, -1, env.problem_size + 1)) if state.current_node is not None else None)
            expected, _ = reference(native_state, distance)
            torch.testing.assert_close(selected, expected)
            assert not torch.isneginf(state.ninf_mask.gather(2, selected[:, :, None])).any()
            state, reward, done = env.step(selected)
            steps += 1
            assert steps < 1000
    assert torch.isneginf(env.visited_ninf_flag[:, :, 1:]).all()
    torch.testing.assert_close(env.selected_node_list[:, :, 1], state.START_NODE)
    torch.testing.assert_close(reward, -env._get_travel_distance())
    reward = reward.reshape(8, 2, env.pomo_size).max(dim=2).values
    assert torch.isfinite(reward).all()
    assert (reward.max(dim=0).values >= reward[0] - 1e-5).all()


@pytest.mark.parametrize('problem', ['CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW'])
def test_shared_trainer_performs_urs_adamw_update(problem, tmp_path):
    args = SimpleNamespace(problem=problem, model_type='URS', device='cpu',
                           log_path=str(tmp_path), checkpoint=None,
                           optimizer_type='AdamW', strict_aux_losses=True)
    trainer = Trainer(args, dict(problem_size=20, pomo_size=3, device='cpu'), params(),
                      dict(optimizer=dict(lr=1e-4, weight_decay=1e-6),
                           scheduler=dict(milestones=[451], gamma=0.1)),
                      dict(lambda_diffusion=0, lambda_recon=0, lambda_contrastive=0))
    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    before = trainer.model.position_embedding.weight.detach().clone()
    env = get_env(problem)[0](problem_size=20, pomo_size=3, device='cpu')
    try:
        score, loss = trainer._train_one_batch(env.get_random_problems(2, 20), env)
    finally:
        trainer.train_log_file.close()
    assert torch.isfinite(torch.tensor([score, loss])).all()
    assert not torch.equal(before, trainer.model.position_embedding.weight)
    gradients = [parameter.grad for parameter in trainer.model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.abs().sum() > 0 for gradient in gradients)
    assert set(trainer.last_loss_components) == {'rl'}


def test_single_pomo_uses_shared_forced_start():
    env = get_env('CVRP')[0](problem_size=20, pomo_size=1, device='cpu')
    env.load_problems(2, problems=env.get_random_problems(2, 20))
    reset, _, _ = env.reset()
    model = URSModel(**params()).eval()
    with torch.no_grad():
        model.pre_forward(reset)
        state, _, _ = env.pre_step()
        selected, _ = model(state)
        state, _, _ = env.step(selected)
        selected, probability = model(state)
    torch.testing.assert_close(selected, state.START_NODE)
    assert probability.shape == (2, 1)
