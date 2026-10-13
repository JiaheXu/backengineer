"""PyTorch translations of custom depth operators recovered from libsl_ai PTX.

See evidence/custom_ops_notes.txt for instruction locations and fidelity limits.
Inputs use NCHW. GWC and Census return NGDHW; offset returns N(DG)HW.
"""
import torch
import torch.nn.functional as F


def _feature_shape(left, right, groups):
    if left.ndim != 4 or left.shape != right.shape:
        raise ValueError('left and right must have matching NCHW shapes')
    if left.dtype != right.dtype or left.device != right.device:
        raise ValueError('left and right must have matching dtype and device')
    if not left.is_floating_point():
        raise ValueError('features must be floating point')
    if groups < 1 or left.shape[1] % groups:
        raise ValueError('groups must be positive and divide feature channels')
    if any(size < 1 for size in left.shape):
        raise ValueError('feature dimensions must be positive')
    return left.shape


def build_gwc_volume(left, right, max_disp, num_groups):
    """Channel-group mean of left[x] * right[x-d], with zero invalid columns."""
    batch, channels, height, width = _feature_shape(left, right, num_groups)
    if max_disp < 1:
        raise ValueError('max_disp must be positive')
    planes = []
    for disparity in range(max_disp):
        if disparity < width:
            product = left[..., disparity:] * right[..., :width-disparity]
            plane = product.reshape(batch, num_groups, channels // num_groups,
                                    height, width-disparity).mean(2)
            planes.append(F.pad(plane, (disparity, 0)))
        else:
            planes.append(left[:, :num_groups] * 0)
    return torch.stack(planes, dim=2)


def build_offset_volume(left, right, disparity, max_offset_disp, num_groups):
    """Correlate at disparity + offsets [1-radius, radius-1].

    Right features are linearly interpolated with zero outside each row.
    Channels are ordered offset first, then feature group.
    """
    batch, channels, height, width = _feature_shape(left, right, num_groups)
    if max_offset_disp < 1:
        raise ValueError('max_offset_disp must be positive')
    if disparity.shape == (batch, height, width):
        disparity = disparity.unsqueeze(1)
    if disparity.shape != (batch, 1, height, width):
        raise ValueError('disparity must have shape N1HW or NHW')
    if disparity.device != left.device or not disparity.is_floating_point():
        raise ValueError('disparity must be floating point on the feature device')
    x = torch.arange(width, dtype=disparity.dtype, device=left.device).view(1, 1, 1, width)
    planes = []
    for offset in range(1-max_offset_disp, max_offset_disp):
        displacement = disparity + offset
        integer = torch.floor(displacement)
        fraction = displacement - integer
        index = (x - integer).to(torch.long)
        sampled = torch.zeros_like(right)
        for tap, weight in ((index, 1-fraction), (index-1, fraction)):
            values = right.gather(3, tap.clamp(0, width-1).expand(-1, channels, -1, -1))
            sampled = sampled + values * weight * ((tap >= 0) & (tap < width))
        plane = (left * sampled).reshape(batch, num_groups, channels // num_groups,
                                        height, width).mean(2)
        planes.append(plane)
    return torch.cat(planes, dim=1)


def census_xnor_cost_volume(left, right, max_disp, groups, census_bits):
    """Sign correlation: 1 - 2 * sign mismatches / bits per group.

    The recovered operator packs channels strictly greater than zero; it does
    not compare spatial neighbors. Sign packing is nondifferentiable.
    """
    batch, channels, height, width = _feature_shape(left, right, groups)
    if census_bits != channels or census_bits % 32:
        raise ValueError('recovered Census requires census_bits == channels, a multiple of 32')
    bits_per_group = census_bits // groups
    if bits_per_group < 32 and 32 % bits_per_group:
        raise ValueError('groups smaller than one packed word must divide 32')
    if bits_per_group > 32 and bits_per_group % 32:
        raise ValueError('groups larger than one packed word must contain whole words')
    if max_disp < 1:
        raise ValueError('max_disp must be positive')
    left_sign, right_sign = left > 0, right > 0
    planes = []
    for disparity in range(max_disp):
        if disparity < width:
            mismatch = (left_sign[..., disparity:] != right_sign[..., :width-disparity])
            counts = mismatch.reshape(batch, groups, bits_per_group, height,
                                      width-disparity).sum(2)
            plane = (1 - counts.to(left.dtype) * (2 / bits_per_group))
            planes.append(F.pad(plane, (disparity, 0)))
        else:
            planes.append(torch.zeros((batch, groups, height, width), dtype=left.dtype,
                                      device=left.device))
    return torch.stack(planes, dim=2)
