"""Rectified RGB pairs with left disparity in full-resolution pixel units.

Validation uses the same centered crop size as training by default, allowing
batch collation. Set training=False, crop_size=None to validate whole images,
padded to multiples of 128; differently sized images then need batch_size=1.
No resizing, horizontal flips, or ImageNet normalization are performed.
"""

import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _read_pfm(path):
    with path.open('rb') as stream:
        if stream.readline().strip() != b'Pf':
            raise ValueError(f'{path}: disparity PFM must be single-channel Pf')
        try:
            width, height = map(int, stream.readline().split())
            scale = float(stream.readline().strip())
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{path}: invalid PFM header') from exc
        if width <= 0 or height <= 0 or not math.isfinite(scale) or scale == 0:
            raise ValueError(f'{path}: invalid PFM dimensions or scale')
        raw = stream.read()
        if len(raw) != width * height * 4:
            raise ValueError(f'{path}: PFM payload does not match shape {height, width}')
        values = np.frombuffer(raw, dtype='<f4' if scale < 0 else '>f4')
        return values.reshape(height, width)[::-1].astype(np.float32) * abs(scale)


class StereoDisparityDataset(Dataset):
    """CSV columns: left,right,disparity[,valid]; paths relative to the CSV.

    Disparity accepts numeric NPY, single-channel PFM, or uint16 PNG (divided
    by png_scale, 256 for KITTI). Optional valid accepts bool NPY or grayscale
    images with positive pixels. Returned RGB is float32 CHW scaled by 1/255;
    disparity is float32 1HW, valid is bool 1HW. Invalid source targets become
    zero. Cropped pixels whose right correspondence is outside the crop are
    masked, while their disparity retains its original pixel units.
    """

    def __init__(self, manifest_path, crop_size=(256, 512), training=True, png_scale=256.0):
        self.manifest_path = Path(manifest_path).resolve()
        self.training = training
        if crop_size is None:
            if training:
                raise ValueError('crop_size=None requires training=False')
        elif (not isinstance(crop_size, (tuple, list)) or len(crop_size) != 2 or any(
                not isinstance(d, int) or isinstance(d, bool) or d <= 0 or d % 128
                for d in crop_size)):
            raise ValueError('crop_size dimensions must be positive integer multiples of 128')
        self.crop_size = crop_size
        if not math.isfinite(png_scale) or png_scale <= 0:
            raise ValueError('png_scale must be positive and finite')
        self.png_scale = png_scale
        self.samples = []
        with self.manifest_path.open(newline='') as stream:
            rows = csv.DictReader(stream)
            if not rows.fieldnames or not {'left', 'right', 'disparity'} <= set(rows.fieldnames):
                raise ValueError(f'{self.manifest_path}: CSV requires left,right,disparity columns')
            for line, row in enumerate(rows, start=2):
                if any(not row.get(key) or not row[key].strip() for key in ('left', 'right', 'disparity')):
                    raise ValueError(f'{self.manifest_path}:{line}: missing left, right, or disparity path')
                sample = {}
                for key in ('left', 'right', 'disparity', 'valid'):
                    value = (row.get(key) or '').strip()
                    if value:
                        path = self.manifest_path.parent / value
                        if not path.is_file():
                            raise FileNotFoundError(f'{self.manifest_path}:{line}: {key} file not found: {path}')
                        sample[key] = path
                self.samples.append(sample)
        if not self.samples:
            raise ValueError(f'{self.manifest_path}: no samples in empty manifest')

    def __len__(self):
        return len(self.samples)

    def _disparity(self, path):
        suffix = path.suffix.lower()
        if suffix == '.npy':
            values = np.load(path, allow_pickle=False)
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise ValueError(f'{path}: disparity NPY must contain real numeric values')
            return values.astype(np.float32)
        if suffix == '.pfm':
            return _read_pfm(path)
        if suffix == '.png':
            with Image.open(path) as image:
                # Pillow decodes 16-bit grayscale PNG to mode I on some versions.
                if image.mode not in ('I;16', 'I;16B', 'I;16L', 'I'):
                    raise ValueError(f'{path}: disparity PNG must be 16-bit grayscale')
                values = np.array(image)
            return values.astype(np.float32) / self.png_scale
        raise ValueError(f'{path}: unsupported disparity format; use NPY, PFM, or PNG')

    def __getitem__(self, index):
        sample = self.samples[index]
        images = []
        for key in ('left', 'right'):
            with Image.open(sample[key]) as image:
                images.append(np.array(image.convert('RGB')))
        left, right = images
        disparity = self._disparity(sample['disparity'])
        h, w = left.shape[:2]
        if right.shape != left.shape or disparity.shape != (h, w):
            raise ValueError(f'{sample["left"]}: images and disparity must have the same spatial shape; '
                             f'left={left.shape}, right={right.shape}, disparity={disparity.shape}')
        valid = np.isfinite(disparity) & (disparity > 0)
        if 'valid' in sample:
            path = sample['valid']
            if path.suffix.lower() == '.npy':
                mask = np.load(path, allow_pickle=False)
                if mask.dtype != np.bool_:
                    raise ValueError(f'{path}: valid NPY mask must have bool dtype')
            else:
                with Image.open(path) as image:
                    mask = np.array(image)
                if mask.ndim != 2:
                    raise ValueError(f'{path}: valid mask must be grayscale')
                mask = mask > 0
            if mask.shape != (h, w):
                raise ValueError(f'{path}: valid mask shape {mask.shape} differs from image shape {(h, w)}')
            valid &= mask
        disparity = np.where(valid, disparity, 0).astype(np.float32)
        target_h, target_w = self.crop_size or (((h + 127) // 128) * 128, ((w + 127) // 128) * 128)
        pad_h, pad_w = max(0, target_h - h), max(0, target_w - w)
        if pad_h or pad_w:
            left = np.pad(left, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            right = np.pad(right, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
            disparity = np.pad(disparity, ((0, pad_h), (0, pad_w)))
            valid = np.pad(valid, ((0, pad_h), (0, pad_w)))
        max_y, max_x = left.shape[0] - target_h, left.shape[1] - target_w
        if self.training:
            y = int(torch.randint(max_y + 1, (1,)).item())
            x = int(torch.randint(max_x + 1, (1,)).item())
        else:
            y, x = max_y // 2, max_x // 2
        selection = np.s_[y:y + target_h, x:x + target_w]
        disparity, valid = disparity[selection], valid[selection].copy()
        valid &= np.arange(target_w)[None, :] - disparity >= 0
        scale = np.float32(1 / 255)
        return {
            'left': torch.from_numpy(left[selection].transpose(2, 0, 1).astype(np.float32) * scale),
            'right': torch.from_numpy(right[selection].transpose(2, 0, 1).astype(np.float32) * scale),
            'disparity': torch.from_numpy(np.ascontiguousarray(disparity[None])),
            'valid': torch.from_numpy(valid[None]),
        }
