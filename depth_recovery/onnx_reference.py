"""CPU ONNX Runtime reference with independent NumPy custom boundaries.

The extracted source graph is never modified on disk. Standard operators run
in graph segments; only recovered custom operators and FP16 QuantizeLinear
run in NumPy. No PyTorch tensors or intermediate results enter this runner.
"""
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper
import onnxruntime as ort


def _shape(left, right, groups):
    if left.ndim != 4 or left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError('features must have matching NCHW shape and dtype')
    if not np.issubdtype(left.dtype, np.floating) or min(left.shape) < 1:
        raise ValueError('features must be floating with positive dimensions')
    if groups < 1 or left.shape[1] % groups:
        raise ValueError('groups must divide channels')
    return left.shape


def build_gwc_volume(left, right, max_disp, num_groups):
    batch, channels, height, width = _shape(left, right, num_groups)
    if max_disp < 1:
        raise ValueError('max_disp must be positive')
    output = np.zeros((batch, num_groups, max_disp, height, width), left.dtype)
    for d in range(min(max_disp, width)):
        product = left[..., d:] * right[..., :width-d]
        output[:, :, d, :, d:] = product.reshape(
            batch, num_groups, channels // num_groups, height, width-d).mean(axis=2)
    return output


def build_offset_volume(left, right, disparity, max_offset_disp, num_groups):
    batch, channels, height, width = _shape(left, right, num_groups)
    if max_offset_disp < 1:
        raise ValueError('max_offset_disp must be positive')
    if disparity.shape == (batch, height, width):
        disparity = disparity[:, None]
    if disparity.shape != (batch, 1, height, width) or not np.issubdtype(disparity.dtype, np.floating):
        raise ValueError('disparity must be floating N1HW or NHW')
    x = np.arange(width, dtype=disparity.dtype).reshape(1, 1, 1, width)
    planes = []
    for offset in range(1-max_offset_disp, max_offset_disp):
        displacement = disparity + offset
        integer = np.floor(displacement)
        fraction = displacement - integer
        index = (x - integer).astype(np.int64)
        sampled = np.zeros_like(right)
        for tap, weight in ((index, 1-fraction), (index-1, fraction)):
            values = np.take_along_axis(right, np.broadcast_to(np.clip(tap, 0, width-1), right.shape), axis=3)
            sampled = sampled + values * weight * ((tap >= 0) & (tap < width))
        planes.append((left * sampled).reshape(batch, num_groups, channels // num_groups,
                                                height, width).mean(axis=2))
    return np.concatenate(planes, axis=1)


def census_xnor_cost_volume(left, right, max_disp, groups, census_bits):
    batch, channels, height, width = _shape(left, right, groups)
    bits = census_bits // groups
    if census_bits != channels or census_bits % 32 or (bits < 32 and 32 % bits) or (bits > 32 and bits % 32):
        raise ValueError('unsupported recovered Census packing')
    if max_disp < 1:
        raise ValueError('max_disp must be positive')
    output = np.zeros((batch, groups, max_disp, height, width), left.dtype)
    for d in range(min(max_disp, width)):
        mismatch = (left[..., d:] > 0) != (right[..., :width-d] > 0)
        counts = mismatch.reshape(batch, groups, bits, height, width-d).sum(axis=2).astype(left.dtype)
        output[:, :, d, :, d:] = 1 - counts * (2 / bits)
    return output


def quantize_linear(x, scale, zero=None, axis=1):
    """ONNX nearest-even integer quantization, dividing half inputs in FP32."""
    if zero is None:
        zero = np.array(0, np.uint8)
    if scale.ndim == 1:
        shape = [1] * x.ndim
        shape[axis] = scale.size
        scale = scale.reshape(shape)
        if zero.size > 1:
            zero = zero.reshape(shape)
    precision = np.float32 if x.dtype == np.float16 else x.dtype
    limits = np.iinfo(zero.dtype)
    result = np.rint(x.astype(precision) / scale.astype(precision)) + zero
    return np.clip(result, limits.min, limits.max).astype(zero.dtype)


_CUSTOM = {'build_gwc_volume', 'build_offset_volume', 'CensusXnorCostVolume'}


class OnnxDepthReference:
    """Run recovered NCHW stereo graphs on CPU, with native graph input dtype."""

    def __init__(self, model_path, threads=2):
        self.model = onnx.load(str(Path(model_path)))
        graph = self.model.graph
        self.output_names = tuple(v.name for v in graph.output)
        self.initializers = {v.name: numpy_helper.to_array(v) for v in graph.initializer}
        # Seed custom output ranks so ONNX can infer standard operators downstream.
        input_type = graph.input[0].type.tensor_type.elem_type
        for node in graph.node:
            if node.op_type in _CUSTOM:
                rank = 4 if node.op_type == 'build_offset_volume' else 5
                graph.value_info.append(helper.make_tensor_value_info(node.output[0], input_type, [None] * rank))
        inferred = onnx.shape_inference.infer_shapes(self.model)
        self.metadata = {v.name: v for v in (*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output)}
        self.metadata.update({v.name: helper.make_tensor_value_info(v.name, v.data_type, list(v.dims)) for v in graph.initializer})
        self.options = ort.SessionOptions()
        self.options.intra_op_num_threads = threads
        self.options.inter_op_num_threads = 1
        self.options.log_severity_level = 3
        self.steps = []
        pending = []
        nodes = list(graph.node)
        for index, node in enumerate(nodes):
            half_quantize = (node.op_type == 'QuantizeLinear' and
                            self.metadata[node.input[0]].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16)
            if node.op_type in _CUSTOM or half_quantize:
                if pending:
                    self._add_segment(pending, nodes[index:])
                    pending = []
                attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
                self.steps.append(('numpy', node, attrs))
            else:
                pending.append(node)
        if pending:
            self._add_segment(pending, [])
        self.uses = Counter(name for step in self.steps for name in self._inputs(step))

    def _add_segment(self, nodes, remaining):
        produced = {name for node in nodes for name in node.output}
        consumed = {name for node in nodes for name in node.input if name}
        needed = set(self.output_names) | {name for node in remaining for name in node.input if name}
        inputs = sorted(consumed - produced - self.initializers.keys())
        outputs = sorted(produced & needed)
        initializers = [v for v in self.model.graph.initializer if v.name in consumed]
        graph = helper.make_graph(nodes, 'independent_reference_segment',
                                  [self.metadata[name] for name in inputs],
                                  [self.metadata[name] for name in outputs], initializers)
        model = helper.make_model(graph, opset_imports=list(self.model.opset_import), ir_version=self.model.ir_version)
        session = ort.InferenceSession(model.SerializeToString(), self.options, providers=['CPUExecutionProvider'])
        self.steps.append(('ort', session, inputs, outputs))

    @staticmethod
    def _inputs(step):
        return [name for name in (step[1].input if step[0] == 'numpy' else step[2]) if name]

    def run(self, left, right):
        if left.shape != right.shape or left.ndim != 4 or left.shape[1] != 3:
            raise ValueError('stereo inputs must have matching NCHW shape with 3 channels')
        dtype = helper.tensor_dtype_to_np_dtype(self.model.graph.input[0].type.tensor_type.elem_type)
        if left.dtype != dtype or right.dtype != dtype:
            raise ValueError(f'graph inputs require {dtype}')
        values = dict(self.initializers, left=left, right=right)
        uses = self.uses.copy()
        for step in self.steps:
            inputs = self._inputs(step)
            if step[0] == 'ort':
                _, session, names, outputs = step
                results = session.run(outputs, {name: values[name] for name in names})
            else:
                _, node, attrs = step
                args = [values[name] if name else None for name in node.input]
                outputs = node.output
                if node.op_type == 'QuantizeLinear':
                    results = [quantize_linear(*args, axis=attrs.get('axis', 1))]
                elif node.op_type == 'CensusXnorCostVolume':
                    results = [census_xnor_cost_volume(*args, **attrs)]
                else:
                    results = [globals()[node.op_type](*args, **attrs)]
            values.update(zip(outputs, results))
            for name in inputs:
                uses[name] -= 1
                if uses[name] == 0 and name not in self.output_names:
                    values.pop(name, None)
        return {name: values[name] for name in self.output_names}
