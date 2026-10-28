"""Exercise the public CLI defaults without launching long experiments."""

from pathlib import Path
import runpy
import sys

import pytest

import Trainer
from baselines.mtlkd import training


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('override, expected', [(None, 900), (300, 300), (5000, 5000)])
def test_root_mtlkd_epoch_default_and_overrides(monkeypatch, tmp_path, override, expected):
    captured = []
    monkeypatch.setattr(training, 'main', lambda *, args: captured.append(args))
    arguments = ['train.py', '--model_type', 'MTLKD', '--no_cuda', '--log_dir', str(tmp_path)]
    if override is not None:
        arguments += ['--epochs', str(override)]
    monkeypatch.setattr(sys, 'argv', arguments)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
    assert exit_info.value.code == 0
    assert captured[0].epochs == expected


@pytest.mark.parametrize('override, expected', [(None, 5000), (2000, 2000)])
def test_root_slotdiff_epoch_default_and_overrides(monkeypatch, tmp_path, override, expected):
    captured = []

    class CaptureTrainer:
        def __init__(self, *, args, **kwargs):
            captured.append(args)

        def run(self):
            pass

    monkeypatch.setattr(Trainer, 'Trainer', CaptureTrainer)
    arguments = ['train.py', '--model_type', 'SlotDiffModel', '--no_cuda', '--log_dir', str(tmp_path)]
    if override is not None:
        arguments += ['--epochs', str(override)]
    monkeypatch.setattr(sys, 'argv', arguments)
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
    assert captured[0].epochs == expected


def test_internal_mtlkd_parser_matches_original_default():
    assert training.build_parser().parse_args([]).epochs == 900
    assert training.build_parser().parse_args(['--epochs', '300']).epochs == 300


@pytest.mark.parametrize('override, expected', [(None, 500), (300, 300), (5000, 5000)])
def test_root_urs_defaults_and_epoch_overrides(monkeypatch, tmp_path, override, expected):
    captured = []

    class CaptureTrainer:
        def __init__(self, *, args, model_params, trainer_params, **kwargs):
            captured.append((args, model_params, trainer_params))

        def run(self):
            pass

    monkeypatch.setattr(Trainer, 'Trainer', CaptureTrainer)
    arguments = ['train.py', '--model_type', 'URS', '--no_cuda', '--log_dir', str(tmp_path)]
    if override is not None:
        arguments += ['--epochs', str(override)]
    monkeypatch.setattr(sys, 'argv', arguments)
    runpy.run_path(str(ROOT / 'train.py'), run_name='__main__')
    args, model_params, trainer_params = captured[0]
    assert args.epochs == expected
    assert args.optimizer_type == 'AdamW' and args.milestones == [451]
    assert model_params['encoder_layer_num'] == 12 and model_params['logit_clipping'] == 50
    assert model_params['demand_max1']
    assert all(trainer_params[name] == 0 for name in ('lambda_diffusion', 'lambda_recon', 'lambda_contrastive'))


@pytest.mark.parametrize('entrypoint', ['train.py', 'test.py'])
def test_root_help_commands_succeed(monkeypatch, capsys, entrypoint):
    monkeypatch.setattr(sys, 'argv', [entrypoint, '--help'])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(ROOT / entrypoint), run_name='__main__')
    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert 'MTLKD' in output
    if entrypoint == 'train.py':
        assert '900 for MTLKD' in output
