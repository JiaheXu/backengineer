import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from finetune import build_parser, run_training, full_resolution_disparity, supervised_loss, _save_checkpoint
from torch_network import ZedDepthNetwork

ROOT = Path(__file__).resolve().parent


def test_failed_checkpoint_write_preserves_previous_weights(tmp_path, monkeypatch):
    destination = tmp_path / 'last.pt'
    destination.write_bytes(b'previous checkpoint')
    def interrupted_save(state, path):
        path.write_bytes(b'incomplete')
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', interrupted_save)
    with pytest.raises(OSError, match='disk full'):
        _save_checkpoint({}, destination)
    assert destination.read_bytes() == b'previous checkpoint'
    assert list(tmp_path.iterdir()) == [destination]


def test_disparity_upsampling_scales_horizontal_pixel_units():
    prediction = torch.full((1, 1, 2, 3), 2., requires_grad=True)
    full = full_resolution_disparity(prediction, (4, 6))
    assert full.shape == (1, 1, 4, 6)
    torch.testing.assert_close(full, torch.full_like(full, 4.))
    full.mean().backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()


def test_masked_loss_ignores_invalid_disparity():
    prediction = torch.tensor([[[[1., 7., 2.]]]], requires_grad=True)
    target = torch.tensor([[[[2., float('nan'), 0.]]]])
    valid = torch.tensor([[[[True, False, False]]]])
    loss, stats = supervised_loss(prediction, target, valid)
    assert float(loss.detach()) == .5
    assert stats['valid_pixels'] == 1 and stats['absolute_error_sum'] == 1.
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[[-1., 0., 0.]]]]))
    loss, stats = supervised_loss(prediction, target, torch.zeros_like(valid))
    assert loss is None and stats['valid_pixels'] == 0


def test_quantized_graph_is_rejected_before_training(tmp_path):
    args = build_parser().parse_args(['--train-csv', 'missing.csv', '--val-csv', 'missing.csv',
        '--model', str(ROOT / 'neural_depth_5.6.onnx'), '--output-dir', str(tmp_path / 'run')])
    with pytest.raises(ValueError, match='quantiz'):
        run_training(args)


def test_one_epoch_updates_weights_and_saves_usable_checkpoints(tmp_path):
    torch.set_num_threads(2)
    rng = np.random.default_rng(7)
    left = rng.integers(0, 256, (128, 256, 3), dtype=np.uint8)
    Image.fromarray(left).save(tmp_path / 'left.png')
    Image.fromarray(np.roll(left, -4, axis=1)).save(tmp_path / 'right.png')
    np.save(tmp_path / 'disparity.npy', np.full((128, 256), 4., dtype=np.float32))
    for split in ['train', 'val']:
        with (tmp_path / f'{split}.csv').open('w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['left', 'right', 'disparity'])
            writer.writerow(['left.png', 'right.png', 'disparity.npy'])
    args = build_parser().parse_args(['--train-csv', str(tmp_path / 'train.csv'),
        '--val-csv', str(tmp_path / 'val.csv'), '--model', str(ROOT / 'neural_depth_5.3.onnx'),
        '--output-dir', str(tmp_path / 'run'), '--epochs', '1', '--batch-size', '1',
        '--crop-height', '128', '--crop-width', '256', '--device', 'cpu', '--threads', '2'])
    result = run_training(args)
    assert result['epochs_completed'] == 1
    state = torch.load(tmp_path / 'run' / 'last.pt', weights_only=True)
    original = ZedDepthNetwork(ROOT / 'neural_depth_5.3.onnx')
    assert any(not torch.equal(value, original.state_dict()[name]) for name, value in state.items())
    original.load_state_dict(state, strict=True)
    assert (tmp_path / 'run' / 'best.pt').exists()
    assert (tmp_path / 'run' / 'training_state.pt').exists()
    records = [json.loads(line) for line in (tmp_path / 'run' / 'metrics.jsonl').read_text().splitlines()]
    assert len(records) == 1 and records[0]['validation']['valid_pixels'] > 0
    # Resume the actual optimizer/model state for one additional epoch.
    args.resume = str(tmp_path / 'run' / 'training_state.pt')
    args.epochs = 2
    result = run_training(args)
    assert result['epochs_completed'] == 2
    assert len((tmp_path / 'run' / 'metrics.jsonl').read_text().splitlines()) == 2
