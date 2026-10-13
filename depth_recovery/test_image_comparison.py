from pathlib import Path
import hashlib

import numpy as np
import pytest
import torch
from PIL import Image

from compare_images import load_stereo_images, output_metrics, compare_images
from convert_weights import convert_weights

ROOT = Path(__file__).resolve().parent


def test_npz_conversion_preserves_names_dtypes_and_values(tmp_path):
    source = tmp_path / 'weights.npz'
    original = {'weight': np.array([[1., 2.]], dtype=np.float16),
                'shape': np.array([2, 3], dtype=np.int64)}
    np.savez(source, **original)
    target = tmp_path / 'weights.pt'
    info = convert_weights(source, target)
    state = torch.load(target, weights_only=True)
    assert info['tensors'] == 2
    assert set(state) == set(original)
    for name, value in state.items():
        assert value.numpy().dtype == original[name].dtype
        assert value.numpy().tobytes() == original[name].tobytes()
    with pytest.raises(FileExistsError):
        convert_weights(source, target)


def test_onnx_checkpoint_loads_strictly(tmp_path):
    from torch_network import ZedDepthNetwork
    path = ROOT / 'neural_depth_5.3.onnx'
    target = tmp_path / 'network.pt'
    convert_weights(path, target)
    model = ZedDepthNetwork(path)
    model.load_state_dict(torch.load(target, weights_only=True), strict=True)


def test_rgb_loading_scales_channels_and_pads_both_images(tmp_path):
    rgb = np.zeros((2, 3, 3), dtype=np.uint8)
    rgb[:] = [255, 128, 0]
    left_path, right_path = tmp_path / 'left.png', tmp_path / 'right.png'
    Image.fromarray(rgb).save(left_path)
    Image.fromarray(rgb // 2).save(right_path)
    left, right, metadata = load_stereo_images(left_path, right_path, np.float32)
    assert left.shape == right.shape == (1, 3, 128, 128)
    np.testing.assert_array_equal(left[0, :, 0, 0], np.array([1., np.float32(128)*np.float32(1/255), 0.]))
    np.testing.assert_array_equal(left[0, :, -1, -1], left[0, :, 0, 0])
    assert metadata['original_hw'] == [2, 3]
    assert metadata['padding_bottom_right'] == [126, 125]
    Image.fromarray(np.zeros((3, 3, 3), dtype=np.uint8)).save(right_path)
    with pytest.raises(ValueError, match='same'):
        load_stereo_images(left_path, right_path, np.float32)


def test_difference_metrics_do_not_hide_nonfinite_values():
    expected = np.array([0., 2.], dtype=np.float32)
    actual = np.array([0., 3.], dtype=np.float32)
    metrics = output_metrics(expected, actual, atol=0.1, rtol=0.1)
    assert metrics['mean_abs_error'] == .5
    assert metrics['max_abs_error'] == 1.
    assert metrics['fraction_outside_tolerance'] == .5
    assert not metrics['allclose']
    metrics = output_metrics(expected, np.array([0., np.nan]), atol=1., rtol=1.)
    assert not metrics['all_finite'] and not metrics['allclose']


def test_images_compare_independent_outputs_and_save_artifacts(tmp_path):
    rgb = np.random.default_rng(123).integers(0, 256, (128, 256, 3), dtype=np.uint8)
    left_path, right_path = tmp_path / 'left.png', tmp_path / 'right.png'
    Image.fromarray(rgb).save(left_path)
    Image.fromarray(np.roll(rgb, -4, axis=1)).save(right_path)
    output = tmp_path / 'comparison'
    source = ROOT / 'neural_depth_5.3.onnx'
    checkpoint = tmp_path / 'converted.pth'
    convert_weights(source, checkpoint)
    report = compare_images(source, left_path, right_path, checkpoint=checkpoint,
                            output_dir=output, threads=2, atol=.003, rtol=.003)
    assert report['checkpoint'] == str(checkpoint.resolve())
    assert report['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert report['reference_backend'] == 'onnxruntime_standard_nodes_numpy_custom_ops'
    assert set(report['outputs']) == {'disparity', 'disparity_right', 'confidence'}
    assert report['allclose']
    with np.load(output / 'onnx_outputs.npz') as onnx_values, np.load(output / 'pytorch_outputs.npz') as torch_values:
        assert onnx_values['disparity'].shape == torch_values['disparity'].shape == (1, 1, 64, 128)
    assert (output / 'comparison.json').exists()
    assert (output / 'disparity_difference.png').exists()
    assert (output / 'analysis.pdf').read_bytes().startswith(b'%PDF-')
    for backend in ['onnx', 'pytorch']:
        for name in ['disparity', 'disparity_right', 'confidence']:
            assert (output / f'{backend}_{name}.png').exists()


def test_changed_pytorch_checkpoint_is_detected(tmp_path):
    source = ROOT / 'neural_depth_5.3.onnx'
    checkpoint = tmp_path / 'changed.pth'
    convert_weights(source, checkpoint)
    state = torch.load(checkpoint, weights_only=True)
    state['x1d5903d0a30a08'].zero_()  # Shared first feature convolution.
    torch.save(state, checkpoint)
    rgb = np.random.default_rng(23).integers(0, 256, (128, 256, 3), dtype=np.uint8)
    left, right = tmp_path / 'left.png', tmp_path / 'right.png'
    Image.fromarray(rgb).save(left)
    Image.fromarray(np.roll(rgb, -3, axis=1)).save(right)
    report = compare_images(source, left, right, checkpoint=checkpoint,
                            output_dir=tmp_path / 'changed_outputs')
    assert not report['allclose']
    assert not report['outputs']['disparity']['allclose']
