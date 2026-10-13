"""PyTorch execution of the recovered ZED inference graphs and shipped weights."""
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .torch_custom_ops import build_gwc_volume, build_offset_volume, census_xnor_cost_volume
except ImportError:
    from torch_custom_ops import build_gwc_volume, build_offset_volume, census_xnor_cost_volume


class ZedDepthNetwork(nn.Module):
    """Return disparity, disparity_right and confidence from NCHW stereo tensors.

    Input tensors must already have the SDK network's preprocessing applied.
    This module executes the extracted inference topology; it does not reproduce
    camera calibration, temporal processing, or the original training pipeline.
    """

    def __init__(self, model_path=None):
        super().__init__()
        path = Path(model_path) if model_path else Path(__file__).with_name('neural_depth_5.3.onnx')
        graph = onnx.load(str(path)).graph
        self.model_path = str(path.resolve())
        self.output_names = tuple(v.name for v in graph.output)
        self.initializer_names = tuple(t.name for t in graph.initializer)
        producers = {out: node for node in graph.node for out in node.output}

        def weight_source(name):
            while name in producers and producers[name].op_type in ('QuantizeLinear', 'DequantizeLinear', 'Cast'):
                name = producers[name].input[0]
            return name

        learned = set()
        for node in graph.node:
            if node.op_type in ('Conv', 'ConvTranspose', 'BatchNormalization'):
                learned.update(weight_source(name) for name in node.input[1:3])
        for tensor in graph.initializer:
            value = torch.from_numpy(np.array(onnx.numpy_helper.to_array(tensor), copy=True))
            if tensor.name in learned and value.is_floating_point():
                self.register_parameter(tensor.name, nn.Parameter(value))
            else:
                self.register_buffer(tensor.name, value)
        self.nodes = []
        for node in graph.node:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            attrs = {k: v.decode() if isinstance(v, bytes) else v for k, v in attrs.items()}
            self.nodes.append((node.name, node.op_type, tuple(node.input), tuple(node.output), attrs))
        self.uses = Counter(name for node in graph.node for name in node.input if name)

    def forward(self, left, right, *, return_intermediates=False):
        if left.shape != right.shape:
            raise ValueError('Stereo inputs must have the same shape')
        if left.ndim != 4 or left.shape[1] != 3:
            raise ValueError('Stereo inputs must be NCHW tensors with 3 channels')
        if not left.is_floating_point() or left.device != right.device or left.dtype != right.dtype:
            raise ValueError('Stereo inputs must have matching floating dtype and device')
        values = {name: getattr(self, name) for name in self.initializer_names}
        values.update(left=left, right=right)
        if left.device != next(self.buffers()).device:
            raise ValueError('Move model and inputs to the same device')
        uses = self.uses.copy()
        for index, (name, op, inputs, outputs, attrs) in enumerate(self.nodes):
            args = [values[x] if x else None for x in inputs]
            try:
                result = self._run(op, args, attrs)
            except Exception as error:
                raise RuntimeError(f'Node {index} {name} ({op}): {error}') from error
            if len(outputs) != 1:
                raise NotImplementedError(f'Multiple outputs for {op}')
            values[outputs[0]] = result
            if not return_intermediates:
                for key in inputs:
                    if key:
                        uses[key] -= 1
                        if uses[key] == 0 and key not in self.output_names:
                            values.pop(key, None)
        result = {name: values[name] for name in self.output_names}
        return (result, values) if return_intermediates else result

    @staticmethod
    def _run(op, xs, a):
        x = xs[0]
        ints = lambda t: tuple(int(v) for v in t.reshape(-1).tolist())
        if op == 'Conv' or op == 'ConvTranspose':
            rank = x.ndim - 2
            pads = a.get('pads', [0] * (rank * 2))
            if pads[:rank] != pads[rank:]:
                raise NotImplementedError('Asymmetric convolution padding')
            fn = getattr(F, ('conv_transpose' if op == 'ConvTranspose' else 'conv') + str(rank) + 'd')
            kwargs = dict(stride=a.get('strides', 1), padding=tuple(pads[:rank]),
                          dilation=a.get('dilations', 1), groups=a.get('group', 1))
            if op == 'ConvTranspose':
                kwargs['output_padding'] = a.get('output_padding', 0)
            return fn(x, xs[1], xs[2] if len(xs) > 2 else None, **kwargs)
        if op == 'BatchNormalization':
            if a.get('training_mode', 0):
                raise NotImplementedError('Only shipped inference BatchNormalization is supported')
            shape = [1, -1] + [1] * (x.ndim - 2)
            scale, bias, mean, var = [t.reshape(shape) for t in xs[1:5]]
            return (x - mean) / torch.sqrt(var + a.get('epsilon', 1e-5)) * scale + bias
        if op == 'Relu': return F.relu(x)
        if op == 'LeakyRelu': return F.leaky_relu(x, a.get('alpha', 0.01))
        if op == 'Sigmoid': return torch.sigmoid(x)
        if op == 'Softmax': return F.softmax(x, dim=a.get('axis', -1))
        if op == 'LogSoftmax': return F.log_softmax(x, dim=a.get('axis', -1))
        if op == 'Add': return x + xs[1]
        if op == 'Mul': return x * xs[1]
        if op == 'Div':
            return x / xs[1] if x.is_floating_point() else torch.div(x, xs[1], rounding_mode='trunc')
        if op == 'Neg': return -x
        if op == 'Concat': return torch.cat(xs, dim=a['axis'])
        if op == 'Shape': return torch.tensor(x.shape[a.get('start', 0):a.get('end', x.ndim)], device=x.device)
        if op == 'Gather':
            axis = a.get('axis', 0) % x.ndim
            index = xs[1].long()
            index = torch.where(index < 0, index + x.shape[axis], index)
            return torch.index_select(x, axis, index.flatten()).reshape(
                tuple(x.shape[:axis]) + tuple(index.shape) + tuple(x.shape[axis + 1:]))
        if op == 'Unsqueeze':
            axes = ints(xs[1]) if len(xs) > 1 else tuple(a['axes'])
            rank = x.ndim + len(axes)
            for axis in sorted(v % rank for v in axes): x = x.unsqueeze(axis)
            return x
        if op == 'Squeeze':
            axes = ints(xs[1]) if len(xs) > 1 else a.get('axes')
            if axes is None: return x.squeeze()
            for axis in sorted((v % x.ndim for v in axes), reverse=True):
                if x.shape[axis] != 1: raise ValueError('Squeeze axis must have size one')
                x = x.squeeze(axis)
            return x
        if op == 'Reshape':
            shape = ints(xs[1])
            if not a.get('allowzero', 0): shape = tuple(x.shape[i] if v == 0 else v for i, v in enumerate(shape))
            return x.reshape(shape)
        if op == 'Transpose': return x.permute(a.get('perm', list(reversed(range(x.ndim)))))
        if op == 'Slice':
            starts, ends = ints(xs[1]), ints(xs[2])
            axes = ints(xs[3]) if len(xs) > 3 else tuple(range(len(starts)))
            steps = ints(xs[4]) if len(xs) > 4 else (1,) * len(starts)
            for start, end, axis, step in zip(starts, ends, axes, steps):
                start, end, step = slice(start, end, step).indices(x.shape[axis])
                indices = torch.arange(start, end, step, device=x.device) if len(range(start, end, step)) else torch.empty(0, dtype=torch.long, device=x.device)
                x = torch.index_select(x, axis, indices)
            return x
        if op == 'Cast':
            dtype = torch.from_numpy(np.empty((), dtype=onnx.helper.tensor_dtype_to_np_dtype(a['to']))).dtype
            return x.to(dtype=dtype)
        if op == 'ReduceSum':
            dims = ints(xs[1]) if len(xs) > 1 else a.get('axes')
            return torch.sum(x, dim=dims, keepdim=bool(a.get('keepdims', 1)))
        if op == 'ReduceL2':
            return torch.sqrt(torch.sum(x.square(), dim=tuple(a['axes']), keepdim=bool(a.get('keepdims', 1))))
        if op == 'Clip': return torch.clamp(x, min=xs[1] if len(xs) > 1 else None, max=xs[2] if len(xs) > 2 else None)
        if op == 'Expand': return x.expand(torch.broadcast_shapes(x.shape, ints(xs[1])))
        if op == 'Tile': return x.repeat(ints(xs[1]))
        if op == 'ConstantOfShape':
            value = np.array(onnx.numpy_helper.to_array(a['value']), copy=True)
            fill = torch.from_numpy(value).to(x.device)
            return torch.full(ints(x), fill.item(), dtype=fill.dtype, device=x.device)
        if op == 'MaxPool':
            rank = x.ndim - 2
            pads = a.get('pads', [0] * (rank * 2))
            if pads[:rank] != pads[rank:]: raise NotImplementedError('Asymmetric pool padding')
            return getattr(F, f'max_pool{rank}d')(x, a['kernel_shape'], a.get('strides'), tuple(pads[:rank]),
                                                a.get('dilations', 1), bool(a.get('ceil_mode', 0)))
        if op == 'Resize':
            mode = a.get('mode', 'nearest')
            coordinate = a.get('coordinate_transformation_mode', 'half_pixel')
            if mode != 'linear' or coordinate not in ('half_pixel', 'align_corners'):
                raise NotImplementedError(f'Resize {mode}/{coordinate}')
            mode = {3: 'linear', 4: 'bilinear', 5: 'trilinear'}[x.ndim]
            options = {'mode': mode, 'align_corners': coordinate == 'align_corners'}
            if len(xs) > 3 and xs[3] is not None and xs[3].numel():
                options['size'] = ints(xs[3])[2:]
            else:
                options['scale_factor'] = tuple(float(v) for v in xs[2].tolist())[2:]
                options['recompute_scale_factor'] = False
            return F.interpolate(x, **options)
        if op in ('QuantizeLinear', 'DequantizeLinear'):
            scale = xs[1]
            zero = xs[2] if len(xs) > 2 else torch.zeros((), dtype=torch.uint8, device=x.device)
            if scale.ndim == 1:
                shape = [1] * x.ndim
                shape[a.get('axis', 1)] = scale.numel()
                scale, zero = scale.reshape(shape), zero.reshape(shape) if zero.numel() > 1 else zero
            if op == 'DequantizeLinear': return (x.to(scale.dtype) - zero.to(scale.dtype)) * scale
            limit = torch.iinfo(zero.dtype)
            # Divide FP16 values in FP32 so half rounding cannot alter integer ties.
            precision = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
            return torch.clamp(torch.round(x.to(precision) / scale.to(precision)) + zero,
                               limit.min, limit.max).to(zero.dtype)
        if op == 'build_gwc_volume': return build_gwc_volume(x, xs[1], a['max_disp'], a['num_groups'])
        if op == 'build_offset_volume': return build_offset_volume(x, xs[1], xs[2], a['max_offset_disp'], a['num_groups'])
        if op == 'CensusXnorCostVolume':
            return census_xnor_cost_volume(x, xs[1], a['max_disp'], a['groups'], a['census_bits'])
        raise NotImplementedError(f'Unsupported recovered operator {op}')
