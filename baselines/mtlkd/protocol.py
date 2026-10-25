"""Adapt the upstream MTL-KD student to the SlotDiff environments.

All instance generation, feasibility masks, transitions, and route costs come
from the parent project's environments; the student and teachers stay upstream.
"""

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from .student import MTLKD_model
from .teacher import SINGLEModel
from utils import get_env, seed_everything


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEACHER_DIR = PROJECT_ROOT / "pretrained" / "mtlkd"
TRAIN_PROBLEMS = ('CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW')
ALL_PROBLEMS = (
    'CVRP', 'OVRP', 'VRPB', 'VRPL', 'VRPTW', 'OVRPTW', 'OVRPB', 'OVRPL',
    'VRPBL', 'VRPBTW', 'VRPLTW', 'OVRPBL', 'OVRPBTW', 'OVRPLTW',
    'VRPBLTW', 'OVRPBLTW',
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return number


def add_common_args(parser, *, testing=False):
    parser.add_argument('--problem', default='ALL' if testing else 'Train_ALL',
                        choices=('ALL', 'Train_ALL', *ALL_PROBLEMS) if testing
                        else ('Train_ALL', *TRAIN_PROBLEMS))
    parser.add_argument('--problem_size', type=positive_int, default=50)
    parser.add_argument('--pomo_size', type=positive_int, default=50)
    parser.add_argument('--seed', type=int, default=2024 if testing else 2023)
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--checkpoint', type=Path, required=testing)
    parser.add_argument('--log_dir', type=Path,
                        default=PROJECT_ROOT / 'results' / ('test/MTLKD' if testing else 'MTLKD'))


def setup(args):
    if args.pomo_size > args.problem_size:
        raise ValueError('--pomo_size must be <= --problem_size')
    if not args.no_cuda and torch.cuda.is_available():
        if not 0 <= args.gpu_id < torch.cuda.device_count():
            raise ValueError(f'GPU {args.gpu_id} unavailable; found {torch.cuda.device_count()} GPU(s)')
        device = torch.device('cuda', args.gpu_id)
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')
    # The shared environments create some tensors without an explicit device,
    # just as in the SlotDiff entry point.
    torch.set_default_device(device)
    seed_everything(args.seed)
    log_path = Path(args.log_dir) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    log_path.mkdir(parents=True)
    return device, log_path


def write_config(path, config):
    path.write_text(json.dumps(config, indent=2, default=str) + '\n')


def make_env(problem, problem_size, pomo_size, device):
    return get_env(problem)[0](problem_size=problem_size, pomo_size=pomo_size, device=device)


def student_params(args, device):
    # Keep MTL-KD's light encoder, heavy decoder, and lack of normalization.
    return dict(embedding_dim=args.embedding_dim,
                sqrt_embedding_dim=args.embedding_dim ** 0.5,
                encoder_layer_num=1, decoder_layer_num=args.decoder_layer_num,
                qkv_dim=args.qkv_dim, head_num=args.head_num,
                logit_clipping=getattr(args, 'logit_clipping', 10), ff_hidden_dim=args.ff_hidden_dim,
                eval_type='softmax', norm=None, device=str(device))


def load_teachers(problems, teacher_dir, teacher_size, device, *, download=True):
    paths = {problem: teacher_dir / f'{problem.lower()}{teacher_size}.pt'
             for problem in problems}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing and download and teacher_size == 100:
        from .assets import ensure_teachers
        ensure_teachers([problem for problem, path in paths.items() if not path.is_file()], teacher_dir)
        missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing teacher checkpoints: ' + ', '.join(missing))
    teachers = {}
    for problem, path in paths.items():
        params = dict(problem=problem, eval_type='argmax', norm_loc='norm_last',
                      norm='instance', embedding_dim=128, sqrt_embedding_dim=128 ** 0.5,
                      encoder_layer_num=6, decoder_layer_num=1, qkv_dim=16,
                      head_num=8, logit_clipping=10, ff_hidden_dim=512,
                      device=str(device))
        teacher = SINGLEModel(**params).to(device)
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        if checkpoint.get('problem', problem) != problem:
            raise ValueError(f'Teacher {path} was trained on {checkpoint["problem"]}, expected {problem}')
        teacher.load_state_dict(checkpoint['model_state_dict'], strict=True)
        teacher.eval().requires_grad_(False)
        hasher = hashlib.sha256()
        with path.open('rb') as file:
            for block in iter(lambda: file.read(1024 * 1024), b''):
                hasher.update(block)
        teacher.checkpoint_metadata = dict(path=str(path), sha256=hasher.hexdigest(),
                                           epoch=checkpoint.get('epoch'),
                                           training_size=teacher_size)
        teachers[problem] = teacher
        print(f'Teacher {problem}: {path} (epoch {checkpoint.get("epoch", "unknown")})')
    return teachers


def prepare_student(model, reset_state):
    # Upstream input: x, y, demand, service time, TW start, TW end;
    # depot non-coordinate features are zero.
    customer = torch.cat((reset_state.node_xy,
                          reset_state.node_demand.unsqueeze(-1),
                          reset_state.node_service_time.unsqueeze(-1),
                          reset_state.node_tw_start.unsqueeze(-1),
                          reset_state.node_tw_end.unsqueeze(-1)), dim=-1)
    depot = torch.cat((reset_state.depot_xy,
                       customer.new_zeros(customer.size(0), 1, 4)), dim=-1)
    reset = SimpleNamespace(depot_node_feat=torch.cat((depot, customer), dim=1))
    model.test = not model.training
    if model.test:
        model.encoded_nodes = model.encoder(reset.depot_node_feat)
    return reset


def student_step(model, reset, state, env):
    # Match SlotDiff's depot move and forced, distinct POMO customer starts.
    if state.selected_count <= 1:
        selected = (torch.zeros_like(state.BATCH_IDX) if state.selected_count == 0
                    else state.START_NODE)
        return selected, None, None
    adapted = SimpleNamespace(**vars(state))
    adapted.visited_ninf_flag = (state.visited_ninf_flag if hasattr(state, 'visited_ninf_flag')
                                else env.visited_ninf_flag)
    adapted.route_limit = (env.route_limit.reshape(env.batch_size)
                           if hasattr(env, 'route_limit')
                           else env.depot_node_demand.new_zeros(env.batch_size))
    selected, candidates, _, probabilities = model(reset, adapted, returnProbs=True)
    return selected, candidates, probabilities


def pomo_chunks(state, env, chunk_size):
    """Slice trajectories, preserving instance indices and local POMO indices."""
    batch_size, pomo_size = state.BATCH_IDX.shape
    for start in range(0, pomo_size, chunk_size):
        end = min(start + chunk_size, pomo_size)
        fields = {}
        for name, value in vars(state).items():
            if isinstance(value, torch.Tensor) and value.ndim >= 2 and value.shape[:2] == (batch_size, pomo_size):
                value = value[:, start:end]
            fields[name] = value
        fields['POMO_IDX'] = torch.arange(end - start, device=state.BATCH_IDX.device)[None, :].expand(batch_size, -1)
        fields['visited_ninf_flag'] = env.visited_ninf_flag[:, start:end]
        yield SimpleNamespace(**fields)


def distillation_loss(teacher, state, selected, candidates, student_probs):
    with torch.no_grad():
        _, _, teacher_probs = teacher(state, selected=selected, returnProbs=True)
        target = teacher_probs.gather(2, candidates).float()
        # Zero indices after the first candidate are padding, not extra depots.
        target[:, :, 1:] = target[:, :, 1:].masked_fill(candidates[:, :, 1:] == 0, 1e-9)
    target = target + 1e-9
    prediction = student_probs + 1e-9
    return (target * (target.log() - prediction.log())).sum(dim=2).mean()


@torch.no_grad()
def evaluate_batch(model, data, env, aug_factor, pomo_chunk_size=8):
    # Shared dataset loaders use torch.Tensor(...), which can create CPU tensors
    # even when factory functions default to CUDA.
    data = tuple(tensor.to(env.device) for tensor in data)
    batch_size = data[0].size(0)
    model.eval()
    model.set_eval_type('argmax')
    env.load_problems(batch_size, problems=data, aug_factor=aug_factor)
    reset_state, _, _ = env.reset()
    reset = prepare_student(model, reset_state)
    state, reward, done = env.pre_step()
    while not done:
        selected = torch.cat([student_step(model, reset, chunk, env)[0]
                              for chunk in pomo_chunks(state, env, pomo_chunk_size)], dim=1)
        state, reward, done = env.step(selected)
    rewards = reward.reshape(aug_factor, batch_size, env.pomo_size).max(dim=2).values
    return -rewards[0].float(), -rewards.max(dim=0).values.float()
