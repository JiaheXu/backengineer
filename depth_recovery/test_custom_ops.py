"""Numeric cases derived from the recovered PTX, with no CUDA dependency."""
import unittest

import torch

from backengineer.depth_recovery.torch_custom_ops import (
    build_gwc_volume, build_offset_volume, census_xnor_cost_volume,
)


class CustomOpsTest(unittest.TestCase):
    def test_gwc_channel_mean_shift_and_layout(self):
        left = torch.tensor([[[[1., 2., 3.]], [[3., 4., 5.]],
                              [[2., 2., 2.]], [[4., 4., 4.]]]])
        right = torch.tensor([[[[10., 20., 30.]], [[2., 4., 6.]],
                               [[3., 6., 9.]], [[5., 10., 15.]]]])
        actual = build_gwc_volume(left, right, 4, 2)
        expected = torch.tensor([[[[[8., 28., 60.]], [[0., 14., 40.]],
                                    [[0., 0., 20.]], [[0., 0., 0.]]],
                                  [[[13., 26., 39.]], [[0., 13., 26.]],
                                   [[0., 0., 13.]], [[0., 0., 0.]]]]])
        torch.testing.assert_close(actual, expected)

    def test_offset_fractional_disparity_and_partial_border(self):
        left = torch.ones(1, 1, 1, 3)
        right = torch.tensor([[[[10., 20., 40.]]]])
        disparity = torch.full((1, 1, 1, 3), .5)
        actual = build_offset_volume(left, right, disparity, 2, 1)
        # Offsets -1, 0, +1; each invalid interpolation tap contributes zero.
        expected = torch.tensor([[[[15., 30., 20.]],
                                  [[5., 15., 30.]], [[0., 5., 15.]]]])
        torch.testing.assert_close(actual, expected)

    def test_offset_offset_major_group_minor_layout(self):
        left = torch.tensor([[[[1., 2.]], [[3., 4.]]]])
        right = torch.tensor([[[[10., 20.]], [[100., 200.]]]])
        actual = build_offset_volume(left, right, torch.zeros(1, 1, 1, 2), 2, 2)
        expected = torch.tensor([[[[20., 0.]], [[600., 0.]],
                                  [[10., 40.]], [[300., 800.]],
                                  [[0., 20.]], [[0., 400.]]]])
        torch.testing.assert_close(actual, expected)

    def test_census_sign_threshold_scale_groups_and_border(self):
        left = torch.ones(1, 32, 1, 2)
        right = torch.ones_like(left)
        right[:, :16] = -1
        right[:, 16, :, :] = 0  # Strict >0; zero has the negative sign bit.
        actual = census_xnor_cost_volume(left, right, 3, 2, 32)
        expected = torch.tensor([[[[[-1., -1.]], [[0., -1.]], [[0., 0.]]],
                                  [[[.875, .875]], [[0., .875]], [[0., 0.]]]]])
        torch.testing.assert_close(actual, expected)
        self.assertFalse(actual.requires_grad)

    def test_census_multiple_packed_words_and_batch(self):
        left = torch.ones(2, 64, 1, 1)
        right = left.clone()
        right[1, 31:33] = -1
        actual = census_xnor_cost_volume(left, right, 1, 1, 64)
        torch.testing.assert_close(actual.flatten(), torch.tensor([1., .9375]))

    def test_feature_and_disparity_gradients(self):
        left = torch.rand(1, 4, 1, 3, dtype=torch.float64, requires_grad=True)
        right = torch.rand_like(left, requires_grad=True)
        disp = torch.full((1, 1, 1, 3), .3, dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(
            lambda l, r: build_gwc_volume(l, r, 2, 2), (left, right)))
        self.assertTrue(torch.autograd.gradcheck(
            lambda l, r, d: build_offset_volume(l, r, d, 2, 2), (left, right, disp)))


if __name__ == '__main__':
    unittest.main()
