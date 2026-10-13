"""Compare reconstructed ONNX and PyTorch depth outputs from two RGB images."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import onnx
from PIL import Image
import torch

try:
    from .onnx_reference import OnnxDepthReference
    from .torch_network import ZedDepthNetwork
except ImportError:
    from onnx_reference import OnnxDepthReference
    from torch_network import ZedDepthNetwork


def load_stereo_images(left_path, right_path, dtype, *, height=None, width=None):
    if (height is None) != (width is None):
        raise ValueError('Provide both --height and --width to resize the stereo pair')
    if height is not None and (height < 1 or width < 1):
        raise ValueError('Resize dimensions must be positive')
    with Image.open(left_path) as image:
        left = image.convert('RGB')
    with Image.open(right_path) as image:
        right = image.convert('RGB')
    if left.size != right.size:
        raise ValueError('Left and right RGB images must have the same dimensions')
    original_hw = [left.height, left.width]
    if height is not None:
        left = left.resize((width, height), Image.Resampling.BILINEAR)
        right = right.resize((width, height), Image.Resampling.BILINEAR)
    image_hw = [left.height, left.width]
    pad_h, pad_w = (-left.height) % 128, (-left.width) % 128

    def tensor(image):
        # Matches the recovered color kernel: RGB CHW, FP32 multiply by 1/255,
        # then cast to the declared model input precision.
        array = np.asarray(image).astype(np.float32) * np.float32(1 / 255)
        array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        return np.ascontiguousarray(array.transpose(2, 0, 1)[None], dtype=dtype)

    metadata = {'original_hw': original_hw, 'resized_hw': image_hw,
                'network_hw': [image_hw[0] + pad_h, image_hw[1] + pad_w],
                'padding_bottom_right': [pad_h, pad_w], 'padding_mode': 'edge',
                'color_order': 'RGB', 'scaling': 'uint8 -> float32 * (1/255) -> model dtype',
                'dtype': str(np.dtype(dtype))}
    return tensor(left), tensor(right), metadata


def output_metrics(reference, actual, *, atol, rtol):
    if reference.shape != actual.shape:
        raise ValueError(f'Output shapes differ: {reference.shape} versus {actual.shape}')
    if atol < 0 or rtol < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError('Tolerances must be finite and nonnegative')
    reference, actual = np.asarray(reference, dtype=np.float64), np.asarray(actual, dtype=np.float64)
    finite = np.isfinite(reference) & np.isfinite(actual)
    difference = np.abs(reference - actual)
    close = finite & (difference <= atol + rtol * np.abs(reference))
    valid = difference[finite]
    return {'shape': list(reference.shape), 'all_finite': bool(finite.all()),
            'nonfinite_elements': int((~finite).sum()),
            'mean_abs_error': float(valid.mean()) if valid.size else None,
            'max_abs_error': float(valid.max()) if valid.size else None,
            'rmse': float(np.sqrt(np.mean(valid ** 2))) if valid.size else None,
            'p99_abs_error': float(np.percentile(valid, 99)) if valid.size else None,
            'fraction_outside_tolerance': float((~close).mean()), 'allclose': bool(close.all())}


def compare_images(model_path, left_path, right_path, *, checkpoint=None, output_dir='depth_comparison',
                   height=None, width=None, threads=2, atol=.003, rtol=.003):
    if threads < 1:
        raise ValueError('threads must be positive')
    if atol < 0 or rtol < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError('Tolerances must be finite and nonnegative')
    model_path = Path(model_path).resolve()
    graph = onnx.load(str(model_path))
    input_type = graph.graph.input[0].type.tensor_type.elem_type
    dtype = {onnx.TensorProto.FLOAT: np.float32, onnx.TensorProto.FLOAT16: np.float16}.get(input_type)
    if dtype is None:
        raise ValueError('Expected a recovered FLOAT or FLOAT16 stereo depth model')
    left, right, preprocessing = load_stereo_images(left_path, right_path, dtype, height=height, width=width)
    torch.set_num_threads(threads)
    model = ZedDepthNetwork(model_path).eval()
    checkpoint_sha256 = None
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
        checkpoint_sha256 = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    start = perf_counter()
    with torch.no_grad():
        result = model(torch.from_numpy(left), torch.from_numpy(right))
    torch_seconds = perf_counter() - start
    torch_outputs = {key: value.cpu().numpy() for key, value in result.items()}
    start = perf_counter()
    reference = OnnxDepthReference(model_path, threads=threads)
    onnx_outputs = reference.run(left, right)
    reference_seconds = perf_counter() - start
    if set(onnx_outputs) != set(torch_outputs):
        raise ValueError('ONNX and PyTorch output names differ')
    outputs = {key: output_metrics(onnx_outputs[key], torch_outputs[key], atol=atol, rtol=rtol)
               for key in onnx_outputs}
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / 'onnx_outputs.npz', **onnx_outputs)
    np.savez_compressed(destination / 'pytorch_outputs.npz', **torch_outputs)
    # Preserve the raw network coordinates, including padding, in both exports.
    difference = np.abs(onnx_outputs['disparity'].astype(np.float32) - torch_outputs['disparity'].astype(np.float32))[0, 0]
    finite = np.isfinite(difference)
    maximum = float(difference[finite].max()) if finite.any() else 0.
    pixels = np.zeros(difference.shape, dtype=np.uint8)
    if maximum:
        pixels[finite] = np.clip(difference[finite] / maximum * 255, 0, 255).astype(np.uint8)
    pixels[~finite] = 255
    Image.fromarray(pixels).save(destination / 'disparity_difference.png')
    report = {'model': str(model_path), 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'checkpoint': str(Path(checkpoint).resolve()) if checkpoint else None,
              'checkpoint_sha256': checkpoint_sha256,
              'left_image': str(Path(left_path).resolve()), 'right_image': str(Path(right_path).resolve()),
              'preprocessing': preprocessing,
              'reference_backend': 'onnxruntime_standard_nodes_numpy_custom_ops',
              'reference_is_sdk_cuda': False, 'pytorch_intermediates_injected': False,
              'fp16_quantization_reference': 'NumPy ONNX nearest-even specification',
              'comparison_domain': 'full network outputs including padded image area',
              'atol': atol, 'rtol': rtol, 'outputs': outputs,
              'allclose': all(value['allclose'] for value in outputs.values()),
              'pytorch_forward_seconds': torch_seconds,
              'onnx_reference_setup_and_run_seconds': reference_seconds,
              'difference_image_scale': 'absolute disparity difference, scaled to maximum (white)',
              'difference_image_max_abs_error': maximum, 'output_directory': str(destination)}
    (destination / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    try:
        from .analyze_outputs import generate_report
    except ImportError:
        from analyze_outputs import generate_report
    report['analysis_pdf'] = str(generate_report(destination))
    (destination / 'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx', default=str(Path(__file__).with_name('neural_depth_5.3.onnx')))
    parser.add_argument('--weights', help='Optional converted PyTorch state_dict checkpoint')
    parser.add_argument('--left', required=True, help='Left RGB image')
    parser.add_argument('--right', required=True, help='Right RGB image')
    parser.add_argument('--output-dir', default='depth_comparison')
    parser.add_argument('--height', type=int, help='Optional resize height; also requires --width')
    parser.add_argument('--width', type=int, help='Optional resize width; also requires --height')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--atol', type=float, default=.003)
    parser.add_argument('--rtol', type=float, default=.003)
    args = parser.parse_args()
    try:
        report = compare_images(args.onnx, args.left, args.right, checkpoint=args.weights,
            output_dir=args.output_dir, height=args.height, width=args.width,
            threads=args.threads, atol=args.atol, rtol=args.rtol)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f'Comparison failed: {error}\n')
    for name, value in report['outputs'].items():
        print(f"{name}: MAE={value['mean_abs_error']} max={value['max_abs_error']} "
              f"RMSE={value['rmse']} allclose={value['allclose']}")
    print(f"Report and outputs: {report['output_directory']}")
    print(f"Analysis PDF: {report['analysis_pdf']}")
    print('Reference: ONNX Runtime standard layers + independent NumPy custom operators.')
    if not report['allclose']:
        parser.exit(2, 'Outputs differ beyond tolerance; see comparison.json.\n')


if __name__ == '__main__':
    main()
