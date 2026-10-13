"""Supervised disparity fine-tuning of recovered differentiable ZED depth graphs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import tempfile
from time import perf_counter

import numpy as np
import onnx
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from .stereo_dataset import StereoDisparityDataset
    from .torch_network import ZedDepthNetwork
except ImportError:
    from stereo_dataset import StereoDisparityDataset
    from torch_network import ZedDepthNetwork


def full_resolution_disparity(prediction, image_hw):
    """Convert output-grid disparity into input-image pixels, preserving gradients."""
    scale = image_hw[1] / prediction.shape[-1]
    return F.interpolate(prediction, size=image_hw, mode='bilinear', align_corners=False) * scale


def _save_checkpoint(state, destination):
    # Preserve the previous checkpoint if serialization is interrupted.
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.tmp', delete=False) as file:
        temporary = Path(file.name)
    try:
        torch.save(state, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def supervised_loss(prediction, target, valid, max_disparity=None):
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError('Prediction, target and validity shapes must agree')
    mask = valid.bool() & torch.isfinite(target) & (target > 0)
    if max_disparity is not None:
        mask &= target < max_disparity
    count = int(mask.sum().item())
    if count == 0:
        return None, {'valid_pixels': 0, 'loss_sum': 0., 'absolute_error_sum': 0., 'bad_pixels': 0}
    selected, labels = prediction[mask], target[mask]
    if not torch.isfinite(selected).all():
        raise FloatingPointError('Nonfinite disparity prediction on valid training pixels')
    loss = F.smooth_l1_loss(selected, labels, beta=1.)
    error = (selected.detach().double() - labels.detach().double()).abs()
    return loss, {'valid_pixels': count, 'loss_sum': float(loss.detach()) * count,
                  'absolute_error_sum': float(error.sum()),
                  'bad_pixels': int(((error > 3.) & (error > .05 * labels.detach())).sum())}


def _epoch(model, loader, device, *, optimizer=None, grad_clip=1., max_disparity=None):
    model.train(optimizer is not None)
    totals = {'valid_pixels': 0, 'loss_sum': 0., 'absolute_error_sum': 0., 'bad_pixels': 0}
    skipped, updates = 0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            left, right, target, valid = [batch[key].to(device) for key in ['left', 'right', 'disparity', 'valid']]
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = full_resolution_disparity(model(left, right)['disparity'], target.shape[-2:])
            loss, statistics = supervised_loss(prediction, target, valid, max_disparity)
            if loss is None:
                skipped += 1
                continue
            if optimizer is not None:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
                optimizer.step()
                updates += 1
            for key in totals:
                totals[key] += statistics[key]
    if totals['valid_pixels'] == 0:
        raise ValueError('Epoch has no valid disparity pixels; check labels, masks, crop and maximum disparity')
    count = totals['valid_pixels']
    return {'loss': totals['loss_sum'] / count, 'epe_pixels': totals['absolute_error_sum'] / count,
            'd1_percent': 100. * totals['bad_pixels'] / count, 'valid_pixels': count,
            'skipped_batches': skipped, 'optimizer_updates': updates}


def run_training(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.threads < 1:
        raise ValueError('Epochs, batch size and threads must be positive; workers must be nonnegative')
    if not np.isfinite([args.learning_rate, args.weight_decay, args.grad_clip]).all() or args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError('Learning rate and gradient clip must be positive and finite; weight decay nonnegative')
    if args.max_disparity is not None and (not np.isfinite(args.max_disparity) or args.max_disparity <= 0):
        raise ValueError('Maximum disparity must be positive and finite')
    model_path = Path(args.model).resolve()
    graph = onnx.load(str(model_path))
    forbidden = {'QuantizeLinear', 'DequantizeLinear', 'CensusXnorCostVolume'}
    if graph.graph.input[0].type.tensor_type.elem_type != onnx.TensorProto.FLOAT or any(n.op_type in forbidden for n in graph.graph.node):
        raise ValueError('Fine-tuning supports FLOAT 5.3 / Light 5.3 graphs; quantized/sign-bit graphs need the original QAT training design')
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Output directory is not empty; choose a new run directory or use --resume')
    if args.resume and args.weights:
        raise ValueError('Use --weights for initialization or --resume for optimizer continuation, not both')
    if args.resume and Path(args.resume).resolve().parent != output:
        raise ValueError('Resume in the original checkpoint directory using --output-dir')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu or run on a CUDA-enabled host')
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_data = StereoDisparityDataset(args.train_csv, (args.crop_height, args.crop_width), True, args.png_scale)
    val_data = StereoDisparityDataset(args.val_csv, (args.crop_height, args.crop_width), False, args.png_scale)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = ZedDepthNetwork(model_path).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    settings = {name: getattr(args, name) for name in ['batch_size', 'learning_rate', 'weight_decay',
        'crop_height', 'crop_width', 'png_scale', 'grad_clip', 'max_disparity', 'seed']}
    for key in ['train_csv', 'val_csv']:
        path = Path(getattr(args, key)).resolve()
        settings[key] = str(path)
        settings[key + '_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    start_epoch, best_epe = 0, float('inf')
    if args.resume:
        state = torch.load(args.resume, map_location='cpu', weights_only=True)
        if state['model_sha256'] != model_hash or state['settings'] != settings:
            raise ValueError('Resume graph, data manifests or training settings differ from the saved run')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        start_epoch, best_epe = state['epoch'], state['best_epe']
        torch.set_rng_state(state['torch_rng'])
        if device.type == 'cuda' and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if args.epochs <= start_epoch:
            raise ValueError('--epochs is the total target epoch count and must exceed the resumed epoch')
    output.mkdir(parents=True, exist_ok=True)
    configuration = {'model': str(model_path), 'model_sha256': model_hash, 'device': str(device),
                     'settings': settings, 'train_samples': len(train_data), 'validation_samples': len(val_data),
                     'loss': 'masked smooth L1 in input-image pixel units',
                     'batch_normalization': 'exported inference/folded behavior is preserved',
                     'confidence_supervision': False}
    (output / 'config.json').write_text(json.dumps(configuration, indent=2) + '\n')
    for epoch in range(start_epoch, args.epochs):
        start = perf_counter()
        training = _epoch(model, train_loader, device, optimizer=optimizer,
                          grad_clip=args.grad_clip, max_disparity=args.max_disparity)
        validation = _epoch(model, val_loader, device, max_disparity=args.max_disparity)
        weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        _save_checkpoint(weights, output / 'last.pt')
        if validation['epe_pixels'] < best_epe:
            best_epe = validation['epe_pixels']
            _save_checkpoint(weights, output / 'best.pt')
        _save_checkpoint({'model': weights, 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1,
                    'best_epe': best_epe, 'model_sha256': model_hash, 'settings': settings,
                    'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None},
                   output / 'training_state.pt')
        record = {'epoch': epoch + 1, 'training': training, 'validation': validation,
                  'best_validation_epe_pixels': best_epe, 'seconds': perf_counter() - start}
        with (output / 'metrics.jsonl').open('a') as file:
            file.write(json.dumps(record, allow_nan=False) + '\n')
        print(f"Epoch {epoch + 1}/{args.epochs}: train loss={training['loss']:.5f}, "
              f"validation EPE={validation['epe_pixels']:.5f}px, D1={validation['d1_percent']:.3f}%", flush=True)
    return {'epochs_completed': args.epochs, 'best_validation_epe_pixels': best_epe, 'output_directory': str(output)}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-csv', required=True)
    parser.add_argument('--val-csv', required=True)
    parser.add_argument('--model', default=str(Path(__file__).with_name('neural_depth_5.3.onnx')))
    parser.add_argument('--weights', help='Optional initial state_dict; otherwise use shipped ONNX weights')
    parser.add_argument('--resume', help='training_state.pt from the original output directory')
    parser.add_argument('--output-dir', default='finetune_run')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--grad-clip', type=float, default=1.)
    parser.add_argument('--crop-height', type=int, default=256)
    parser.add_argument('--crop-width', type=int, default=512)
    parser.add_argument('--png-scale', type=float, default=256.)
    parser.add_argument('--max-disparity', type=float, help='Optional exclusive upper label threshold in input pixels')
    parser.add_argument('--device', default='auto', help='auto, cpu, cuda, or cuda:N')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--seed', type=int, default=17)
    return parser


if __name__ == '__main__':
    parser = build_parser()
    try:
        print(json.dumps(run_training(parser.parse_args()), indent=2))
    except (OSError, ValueError, RuntimeError, ArithmeticError) as error:
        parser.exit(1, f'Fine-tuning failed: {error}\n')
