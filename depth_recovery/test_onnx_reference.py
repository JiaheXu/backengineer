"""Independent reference checks; no SDK/CUDA equivalence is implied."""
from pathlib import Path
import numpy as np
import pytest
from backengineer.depth_recovery.onnx_reference import (
    OnnxDepthReference, build_gwc_volume, build_offset_volume,
    census_xnor_cost_volume, quantize_linear,
)


def test_gwc_manual():
    left = np.array([[[[1, 2, 3]], [[2, 4, 6]]]], dtype=np.float32)
    right = np.array([[[[2, 3, 4]], [[1, 2, 3]]]], dtype=np.float32)
    np.testing.assert_array_equal(build_gwc_volume(left, right, 4, 1),
                                  [[[[[2, 7, 15]], [[0, 4, 10.5]], [[0, 0, 6]], [[0, 0, 0]]]]])


def test_offset_manual_partial_border():
    left = np.ones((1, 1, 1, 3), np.float32)
    right = np.array([[[[2, 4, 8]]]], np.float32)
    disparity = np.full((1, 1, 1, 3), .5, np.float32)
    np.testing.assert_array_equal(build_offset_volume(left, right, disparity, 2, 1),
                                  [[[[3, 6, 4]], [[1, 3, 6]], [[0, 1, 3]]]])


def test_census_manual():
    left = np.ones((1, 32, 1, 2), np.float32)
    right = left.copy()
    right[:, :8, :, 0] = 0
    right[:, :, :, 1] = -1
    np.testing.assert_array_equal(census_xnor_cost_volume(left, right, 3, 2, 32),
                                  [[[[[0, -1]], [[0, 0]], [[0, 0]]],
                                    [[[1, -1]], [[0, 1]], [[0, 0]]]]])


def test_half_quantization_nearest_even():
    x = np.array([.5, 1.5, 2.5, -1.5, 500], np.float16)
    np.testing.assert_array_equal(quantize_linear(x, np.array(1, np.float16), np.array(0, np.int8)),
                                  [0, 2, 2, -2, 127])


@pytest.mark.parametrize('version', ['5.3', 'light_5.3', '5.4', '5.6'])
def test_full_graph_independent(version, tmp_path):
    import torch
    from backengineer.depth_recovery.convert_weights import convert_weights
    from backengineer.depth_recovery.torch_network import ZedDepthNetwork
    torch.set_num_threads(2)
    path = Path(__file__).with_name(f'neural_depth_{version}.onnx')
    rng = np.random.default_rng(17)
    dtype = np.float16 if version == '5.6' else np.float32
    left = rng.random((1, 3, 128, 256), dtype=np.float32).astype(dtype)
    right = np.roll(left, -3, axis=3).copy()
    outputs = OnnxDepthReference(path).run(left, right)
    checkpoint = tmp_path / (path.stem + '.pth')
    convert_weights(path, checkpoint)
    model = ZedDepthNetwork(path).eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    with torch.no_grad():
        expected = model(torch.from_numpy(left), torch.from_numpy(right))
    assert set(outputs) == {'disparity', 'disparity_right', 'confidence'}
    for name, actual in outputs.items():
        wanted = expected[name].numpy()
        assert actual.shape == wanted.shape
        assert np.isfinite(actual).all()
        error = np.abs(actual.astype(np.float32) - wanted.astype(np.float32))
        print(f'{version} {name}: max_abs={error.max():.7g} mean_abs={error.mean():.7g}')
        if version != '5.6':
            np.testing.assert_allclose(actual, wanted, rtol=2e-4, atol=2e-3)


def test_stereo_validation():
    path = Path(__file__).with_name('neural_depth_light_5.3.onnx')
    runner = OnnxDepthReference(path)
    left = np.zeros((1, 3, 128, 256), np.float32)
    with pytest.raises(ValueError, match='matching NCHW'):
        runner.run(left, left[:, :, :, :-1])
    with pytest.raises(ValueError, match='graph inputs require'):
        runner.run(left.astype(np.float16), left.astype(np.float16))
