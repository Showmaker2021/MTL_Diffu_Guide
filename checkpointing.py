"""Training state shared by SlotDiff and its experiment baselines."""

import random

import numpy as np
import torch


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_training_state(checkpoint, optimizer, scheduler):
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # Older checkpoints have no RNG state: they resume successfully, but do not
    # reproduce the random sequence of an uninterrupted run.
    state = checkpoint.get('rng_state')
    if state is not None:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'].cpu())
        if torch.cuda.is_available() and 'cuda' in state:
            for index, cuda_state in enumerate(state['cuda']):
                if index < torch.cuda.device_count():
                    torch.cuda.set_rng_state(cuda_state.cpu(), index)
