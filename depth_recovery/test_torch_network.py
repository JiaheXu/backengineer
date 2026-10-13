from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from torch_network import ZedDepthNetwork

ROOT = Path(__file__).resolve().parent


def test_weights_are_registered_without_changes():
    net = ZedDepthNetwork(ROOT / 'neural_depth_5.3.onnx')
    with np.load(ROOT / 'neural_depth_5.3.weights.npz') as original:
        assert set(net.state_dict()) == set(original.files)
        for name, tensor in net.state_dict().items():
            assert tensor.numpy().tobytes() == original[name].tobytes()
    assert sum(p.numel() for p in net.parameters()) > 7_000_000


def test_invalid_stereo_inputs_are_rejected():
    net = ZedDepthNetwork()
    with pytest.raises(ValueError, match='same'):
        net(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 96))
    with pytest.raises(ValueError, match='3 channels'):
        net(torch.zeros(1, 1, 64, 64), torch.zeros(1, 1, 64, 64))


def test_quantization_rounds_to_even_and_saturates():
    x = torch.tensor([-200., -1.9, -1.5, -0.5, 0.5, 1.5, 1.9, 200.], dtype=torch.float16)
    result = ZedDepthNetwork._run('QuantizeLinear', [x, torch.tensor(1., dtype=torch.float16),
                                                  torch.tensor(0, dtype=torch.int8)], {})
    torch.testing.assert_close(result, torch.tensor([-128, -2, -2, 0, 0, 2, 2, 127], dtype=torch.int8))


def test_regular_forward_matches_trace_and_backpropagates():
    torch.set_num_threads(2)
    net = ZedDepthNetwork().eval()
    left = torch.rand(1, 3, 128, 256, requires_grad=True)
    right = torch.rand_like(left, requires_grad=True)
    result = net(left, right)
    with torch.no_grad():
        traced, _ = net(left, right, return_intermediates=True)
    for key, value in result.items():
        torch.testing.assert_close(value, traced[key])
        assert value.shape == (1, 1, 64, 128)
    result['disparity'].mean().backward()
    assert left.grad is not None and torch.isfinite(left.grad).all()
    assert right.grad is not None and torch.isfinite(right.grad).all()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 70 and all(torch.isfinite(g).all() for g in grads)


def test_batched_stereo():
    torch.set_num_threads(2)
    net = ZedDepthNetwork().eval()
    with torch.no_grad():
        result = net(torch.rand(2, 3, 128, 256), torch.rand(2, 3, 128, 256))
    assert all(value.shape == (2, 1, 64, 128) for value in result.values())
    # The original ONNX graph exposes this as a zero-filled placeholder.
    assert torch.count_nonzero(result['disparity_right']) == 0


@pytest.mark.parametrize('name', ['neural_depth_5.3', 'neural_depth_light_5.3',
                                  'neural_depth_5.4', 'neural_depth_5.6'])
def test_standard_graph_matches_onnxruntime(name, tmp_path):
    """Inject custom/Q outputs; independently verify quantization and other layers."""
    torch.set_num_threads(2)
    torch.manual_seed(17)
    path = ROOT / (name + '.onnx')
    net = ZedDepthNetwork(path).eval()
    from convert_weights import convert_weights
    checkpoint = tmp_path / (name + '.pth')
    convert_weights(path, checkpoint)
    net.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    dtype = torch.float16 if name.endswith('5.6') else torch.float32
    left = torch.rand(1, 3, 128, 256, dtype=dtype)
    right = torch.rand_like(left)
    with torch.no_grad():
        result, trace = net(left, right, return_intermediates=True)
    assert set(result) == {'disparity', 'disparity_right', 'confidence'}
    assert all(torch.isfinite(t).all() for t in result.values())

    graph = onnx.load(str(path))
    quantizers = [n for n in graph.graph.node if n.op_type == 'QuantizeLinear']
    for node in quantizers:
        data, scale, zero = [trace[key].detach().numpy() for key in node.input]
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if scale.ndim == 1:
            shape = [1] * data.ndim
            shape[attrs.get('axis', 1)] = scale.size
            scale, zero = scale.reshape(shape), zero.reshape(shape) if zero.size > 1 else zero
        limits = np.iinfo(zero.dtype)
        expected = np.clip(np.rint(data.astype(np.float32) / scale.astype(np.float32)) + zero,
                           limits.min, limits.max).astype(zero.dtype)
        np.testing.assert_array_equal(trace[node.output[0]].numpy(), expected)
    custom = [n for n in graph.graph.node if n.domain == 'sl' or
              n.op_type in ('CensusXnorCostVolume', 'QuantizeLinear')]
    standard = [n for n in graph.graph.node if n not in custom]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if dtype == torch.float16:
        # Half rounding accumulates between nodes; compare each op with identical
        # inputs rather than accepting a broad tolerance for the whole graph.
        for node in standard:
            inputs = list(dict.fromkeys(key for key in node.input if key))
            feed = {key: trace[key].detach().numpy() for key in inputs}
            info = lambda key, value: onnx.helper.make_tensor_value_info(
                key, onnx.helper.np_dtype_to_tensor_dtype(value.dtype), value.shape)
            subgraph = onnx.helper.make_graph([node], node.name,
                [info(key, value) for key, value in feed.items()],
                [info(key, trace[key].detach().numpy()) for key in node.output])
            submodel = onnx.helper.make_model(subgraph, opset_imports=graph.opset_import,
                                               ir_version=graph.ir_version)
            session = ort.InferenceSession(submodel.SerializeToString(), options,
                                           providers=['CPUExecutionProvider'])
            for key, expected in zip(node.output, session.run(None, feed)):
                actual = trace[key].detach().numpy()
                if np.issubdtype(expected.dtype, np.integer):
                    np.testing.assert_array_equal(actual, expected, err_msg=f'{node.op_type} {key}')
                else:
                    np.testing.assert_allclose(actual, expected, atol=0.005, rtol=0.005,
                                               err_msg=f'{node.op_type} {key}')
        return
    del graph.graph.node[:]
    graph.graph.node.extend(standard)
    feed = {'left': left.numpy(), 'right': right.numpy()}
    for node in custom:
        for output in node.output:
            array = trace[output].numpy()
            graph.graph.input.append(onnx.helper.make_tensor_value_info(
                output, onnx.helper.np_dtype_to_tensor_dtype(array.dtype), array.shape))
            feed[output] = array
    # Include all standard intermediate results to expose errors hidden by final activations.
    del graph.graph.output[:]
    outputs = [output for node in standard for output in node.output]
    for output in outputs:
        value = trace[output].numpy()
        graph.graph.output.append(onnx.helper.make_tensor_value_info(
            output, onnx.helper.np_dtype_to_tensor_dtype(value.dtype), value.shape))
    session = ort.InferenceSession(graph.SerializeToString(), options, providers=['CPUExecutionProvider'])
    expected = session.run(outputs, feed)
    for key, value in zip(outputs, expected):
        actual = trace[key].numpy()
        if np.issubdtype(value.dtype, np.integer):
            np.testing.assert_array_equal(actual, value, err_msg=key)
        else:
            atol, rtol = (0.05, 0.05) if dtype == torch.float16 else (0.003, 0.003)
            np.testing.assert_allclose(actual, value, atol=atol, rtol=rtol, err_msg=key)
