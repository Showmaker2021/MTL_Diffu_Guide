"""URS policy interface for the root VRP environments.

The original model, attention/bias modules, and problem encodings are vendored
under baselines/urs/. This adapter changes the state representation only;
the shared environments retain feasibility, transitions, and cost calculation.
"""

import math
from types import SimpleNamespace

import torch

from baselines.urs.model import Model
from baselines.urs.problem_rep import get_problem_representations


__all__ = ['URSModel']


def urs_problem_name(problem_bits):
    """Translate shared [C, O, B, L, TW] flags to the upstream problem name."""
    _, opened, backhaul, length, time_window = problem_bits.detach().cpu().flatten().tolist()
    return (('o' if opened else '') + 'cvrp' + ('b' if backhaul else '')
            + ('l' if length else '') + ('tw' if time_window else ''))


class URSModel(Model):
    def __init__(self, **model_params):
        params = dict(model_params)
        params['demand_max1'] = params.get('demand_max1', True)
        params['eval_type'] = 'greedy'
        super().__init__(**params)
        self.eval_type = model_params.get('eval_type', 'argmax')
        self.problem_encodings = get_problem_representations()
        self.shared_distance = None
        self.last_unified_state = None

    def set_eval_type(self, eval_type):
        self.eval_type = eval_type
        self.set_decoder_type('sampling' if self.training or eval_type == 'softmax' else 'greedy')

    def pre_forward(self, reset_state):
        self.set_eval_type(self.eval_type)
        problem = urs_problem_name(reset_state.prob_emb[0])
        batch, customers, _ = reset_state.node_xy.shape
        xy = torch.cat((reset_state.depot_xy, reset_state.node_xy), dim=1)
        # URS raw UDR fields: x,y,demand,prize,fake_prize,penalty,TW start,TW end,service.
        raw = xy.new_zeros(batch, customers + 1, 9)
        raw[:, :, :2] = xy
        raw[:, 1:, 2] = reset_state.node_demand
        raw[:, 1:, 6] = reset_state.node_tw_start
        raw[:, 1:, 7] = reset_state.node_tw_end
        raw[:, 1:, 8] = reset_state.node_service_time
        if 'tw' in problem:
            raw[:, 0, 7] = 3.0  # shared synthetic environments' depot closing time
        distance = torch.cdist(xy, xy, compute_mode='donot_use_mm_for_euclid_dist')
        unified = SimpleNamespace(problems=raw, dist=distance,
                                  log_scale=math.log2(customers), relation=None)
        representation = raw.new_tensor(self.problem_encodings[problem])
        self.shared_distance = distance
        self.last_unified_state = unified
        self.decoder.assign(representation)
        super().pre_forward(unified, problem, representation)

    def forward(self, state, selected=None):
        if selected is not None:
            raise ValueError('URS does not support forcing arbitrary decoder actions')
        # Use the same forced customer starts as the other methods, including POMO=1.
        if state.selected_count == 1:
            return state.START_NODE, self.shared_distance.new_ones(state.BATCH_IDX.shape)
        batch, pomo = state.BATCH_IDX.shape
        adapted = SimpleNamespace(**vars(state))
        adapted.batch_size = batch
        adapted.pomo_size = pomo
        adapted.depot_num = 1
        if state.current_node is None:
            distance = None
        else:
            indices = state.current_node.unsqueeze(-1).expand(-1, -1, self.shared_distance.size(-1))
            distance = self.shared_distance.gather(1, indices)
        return super().forward(adapted, distance)
