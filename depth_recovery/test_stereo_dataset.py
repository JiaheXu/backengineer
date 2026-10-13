import csv

import numpy as np
import pytest
import torch
from PIL import Image

from stereo_dataset import StereoDisparityDataset


def make_sample(root, disparity, valid=None):
    h, w = disparity.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = np.arange(w, dtype=np.uint8)
    rgb[..., 1] = np.arange(h, dtype=np.uint8)[:, None]
    rgb[..., 2] = 255
    Image.fromarray(rgb).save(root / 'left.png')
    Image.fromarray(rgb // 2).save(root / 'right.png')
    np.save(root / 'disp.npy', disparity)
    if valid is not None:
        np.save(root / 'valid.npy', valid)
    manifest = root / 'data.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['left', 'right', 'disparity', 'valid'])
        writer.writerow(['left.png', 'right.png', 'disp.npy', 'valid.npy' if valid is not None else ''])
    return manifest, rgb


def test_rgb_alignment_padding_and_invalid_targets(tmp_path):
    disparity = np.array([[1, 1, np.nan, 2], [np.inf, -1, 1, 0]], dtype=np.float32)
    manifest, rgb = make_sample(tmp_path, disparity)
    item = StereoDisparityDataset(manifest, crop_size=None, training=False)[0]
    assert len(StereoDisparityDataset(manifest)) == 1
    assert item['left'].shape == item['right'].shape == (3, 128, 128)
    assert item['left'].dtype == item['disparity'].dtype == torch.float32
    assert item['valid'].dtype == torch.bool
    np.testing.assert_array_equal(item['left'][:, :2, :4], rgb.transpose(2, 0, 1).astype(np.float32) * np.float32(1/255))
    np.testing.assert_array_equal(item['right'][:, :2, :4], (rgb // 2).transpose(2, 0, 1).astype(np.float32) * np.float32(1/255))
    assert torch.equal(item['left'][:, -1, -1], item['left'][:, 1, 3])
    assert item['valid'][0, :2, :4].tolist() == [[False, True, False, True], [False, False, True, False]]
    assert item['disparity'][0, :2, :4].tolist() == [[1, 1, 0, 2], [0, 0, 1, 0]]
    assert not item['valid'][:, 2:].any()
    assert not item['valid'][:, :, 4:].any()
    assert not item['disparity'][:, 2:].any()


def test_centered_crop_preserves_units_and_masks_missing_correspondence(tmp_path):
    manifest, rgb = make_sample(tmp_path, np.full((132, 140), 4, dtype=np.float32))
    dataset = StereoDisparityDataset(manifest, crop_size=(128, 128), training=False)
    first, second = dataset[0], dataset[0]
    assert torch.equal(first['left'], second['left'])
    np.testing.assert_array_equal(first['left'][:, 0, 0], rgb[2, 6].astype(np.float32) * np.float32(1/255))
    assert (first['disparity'] == 4).all()
    assert not first['valid'][..., :4].any()
    assert first['valid'][..., 4:].all()


def test_training_crop_uses_same_random_origin(tmp_path, monkeypatch):
    manifest, rgb = make_sample(tmp_path, np.ones((132, 140), dtype=np.float32))
    origins = iter([3, 9])
    monkeypatch.setattr(torch, 'randint', lambda *args, **kwargs: torch.tensor([next(origins)]))
    item = StereoDisparityDataset(manifest, crop_size=(128, 128))[0]
    np.testing.assert_array_equal(item['left'][:, 0, 0], rgb[3, 9].astype(np.float32) * np.float32(1/255))
    np.testing.assert_array_equal(item['right'][:, 0, 0], (rgb[3, 9] // 2).astype(np.float32) * np.float32(1/255))


@pytest.mark.parametrize('endian,scale', [('<', -1.0), ('>', 1.0)])
def test_pfm_byte_order_and_bottom_to_top_rows(tmp_path, endian, scale):
    disparity = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    manifest, _ = make_sample(tmp_path, disparity)
    pfm = tmp_path / 'disp.pfm'
    pfm.write_bytes(f'Pf\n3 2\n{scale}\n'.encode() + disparity[::-1].astype(endian + 'f4').tobytes())
    manifest.write_text('left,right,disparity\nleft.png,right.png,disp.pfm\n')
    item = StereoDisparityDataset(manifest, crop_size=None, training=False)[0]
    np.testing.assert_array_equal(item['disparity'][0, :2, :3], disparity)


def test_png_scale_and_grayscale_valid_mask(tmp_path):
    manifest, _ = make_sample(tmp_path, np.ones((2, 4), dtype=np.float32))
    Image.fromarray(np.full((2, 4), 512, dtype=np.uint16)).save(tmp_path / 'disp.png')
    mask = np.array([[255, 255, 0, 255], [0, 255, 255, 255]], dtype=np.uint8)
    Image.fromarray(mask).save(tmp_path / 'mask.png')
    manifest.write_text('left,right,disparity,valid\nleft.png,right.png,disp.png,mask.png\n')
    item = StereoDisparityDataset(manifest, crop_size=None, training=False)[0]
    assert item['disparity'][0, 0, 3] == 2
    assert not item['valid'][0, 0, 2]
    assert item['disparity'][0, 0, 2] == 0
    assert item['valid'][0, 1, 2]
    custom = StereoDisparityDataset(manifest, crop_size=None, training=False, png_scale=512)[0]
    assert custom['disparity'][0, 0, 3] == 1


def test_bool_npy_mask_and_shape_validation(tmp_path):
    mask = np.ones((2, 4), dtype=bool)
    mask[1, 3] = False
    manifest, _ = make_sample(tmp_path, np.ones((2, 4), dtype=np.float32), mask)
    item = StereoDisparityDataset(manifest)[0]
    assert not item['valid'][0, 1, 3]
    np.save(tmp_path / 'valid.npy', np.ones((1, 4), dtype=bool))
    with pytest.raises(ValueError, match='shape'):
        StereoDisparityDataset(manifest)[0]
    Image.new('RGB', (4, 3)).save(tmp_path / 'right.png')
    with pytest.raises(ValueError, match='shape|same'):
        StereoDisparityDataset(manifest)[0]


@pytest.mark.parametrize('crop', [(0, 128), (128, 129), (-128, 128), 128, (128,), ('128', 128)])
def test_crop_validation(tmp_path, crop):
    manifest, _ = make_sample(tmp_path, np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match='128'):
        StereoDisparityDataset(manifest, crop_size=crop)


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_scale_validation(tmp_path, scale):
    manifest, _ = make_sample(tmp_path, np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match='png_scale'):
        StereoDisparityDataset(manifest, png_scale=scale)


def test_manifest_errors_and_label_shape(tmp_path):
    manifest, _ = make_sample(tmp_path, np.ones((2, 4), dtype=np.float32))
    np.save(tmp_path / 'disp.npy', np.ones((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match='shape'):
        StereoDisparityDataset(manifest)[0]
    manifest.write_text('left,right\nleft.png,right.png\n')
    with pytest.raises(ValueError, match='disparity'):
        StereoDisparityDataset(manifest)
    manifest.write_text('left,right,disparity\n')
    with pytest.raises(ValueError, match='empty|samples'):
        StereoDisparityDataset(manifest)


def test_rejects_unsafe_npy_and_8bit_disparity_png(tmp_path):
    manifest, _ = make_sample(tmp_path, np.ones((2, 4), dtype=np.float32))
    np.save(tmp_path / 'disp.npy', np.full((2, 4), '1', dtype=object))
    with pytest.raises(ValueError, match='pickle'):
        StereoDisparityDataset(manifest)[0]
    Image.fromarray(np.ones((2, 4), dtype=np.uint8)).save(tmp_path / 'disp.png')
    manifest.write_text('left,right,disparity\nleft.png,right.png,disp.png\n')
    with pytest.raises(ValueError, match='16-bit'):
        StereoDisparityDataset(manifest)[0]


def test_missing_files_and_missing_row_values_are_informative(tmp_path):
    manifest = tmp_path / 'data.csv'
    manifest.write_text('left,right,disparity\nleft.png,right.png,disp.npy\n')
    with pytest.raises(FileNotFoundError, match='left.*left.png'):
        StereoDisparityDataset(manifest)
    manifest.write_text('left,right,disparity\nleft.png,right.png,\n')
    with pytest.raises(ValueError, match='missing'):
        StereoDisparityDataset(manifest)
