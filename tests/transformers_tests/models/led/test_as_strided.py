import unittest

import numpy as np
import torch

import mindspore as ms

from mindone.transformers.models.led.modeling_led import as_strided


class TestAsStrided(unittest.TestCase):
    def setUp(self):
        # Set seeds for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        ms.set_seed(42)

    def assert_tensor_equal(self, ms_tensor, torch_tensor, msg=None):
        """Helper to compare MindSpore and PyTorch tensors"""
        ms_np = ms_tensor.asnumpy()
        torch_np = torch_tensor.detach().cpu().numpy()
        np.testing.assert_allclose(ms_np, torch_np, rtol=1e-5, atol=1e-5, err_msg=msg)

    def test_basic_striding(self):
        """Test basic striding operations"""
        # Create identical input tensors
        torch_input = torch.tensor(
            [[0.9039, 0.6291, 1.0795], [0.1586, 2.1939, -0.4900], [-0.1909, -0.7503, 1.9355]], dtype=torch.float32
        )
        ms_input = ms.Tensor(torch_input.numpy())

        # Test case 1: Basic 2x2 window with stride (1,2)
        size, stride = (2, 2), (1, 2)
        torch_output = torch.as_strided(torch_input, size=size, stride=stride)
        ms_output = as_strided(ms_input, size=size, stride=stride)
        self.assert_tensor_equal(ms_output, torch_output, "Basic 2x2 window test failed")

    def test_different_shapes(self):
        """Test with different input shapes and strides"""
        # Create 4x4 input
        torch_input = torch.randn(4, 4)
        ms_input = ms.Tensor(torch_input.numpy())

        test_cases = [
            ((2, 2), (2, 2)),  # Non-overlapping blocks
            ((2, 2), (1, 1)),  # Overlapping blocks
            ((3, 3), (1, 1)),  # Larger window
            ((1, 4), (1, 1)),  # Row-wise sliding
            ((1, 4), (0, 0)),
        ]

        for size, stride in test_cases:
            torch_output = torch.as_strided(torch_input, size=size, stride=stride)
            ms_output = as_strided(ms_input, size=size, stride=stride)
            self.assert_tensor_equal(ms_output, torch_output, f"Failed for size={size}, stride={stride}")

    def test_with_storage_offset(self):
        """Test with different storage offsets"""
        torch_input = torch.randn(5, 5)
        ms_input = ms.Tensor(torch_input.numpy())

        test_cases = [
            ((2, 2), (1, 1), 1),  # Small offset
            ((2, 2), (2, 2), 2),  # Larger offset
            ((3, 3), (1, 1), 3),  # Different window size
        ]

        for size, stride, offset in test_cases:
            torch_output = torch.as_strided(torch_input, size=size, stride=stride, storage_offset=offset)
            ms_output = as_strided(ms_input, size=size, stride=stride, storage_offset=offset)
            self.assert_tensor_equal(
                ms_output, torch_output, f"Failed for size={size}, stride={stride}, offset={offset}"
            )

    def test_edge_cases(self):
        """Test edge cases"""
        torch_input = torch.randn(3, 3)
        ms_input = ms.Tensor(torch_input.numpy())

        # Test cases that should work
        edge_cases = [
            ((1, 1), (1, 1)),  # Minimal window
            ((3, 3), (1, 1)),  # Full size window
            ((2, 2), (2, 2)),  # Non-overlapping
        ]

        for size, stride in edge_cases:
            torch_output = torch.as_strided(torch_input, size=size, stride=stride)
            ms_output = as_strided(ms_input, size=size, stride=stride)
            self.assert_tensor_equal(ms_output, torch_output, f"Edge case failed for size={size}, stride={stride}")


if __name__ == "__main__":
    unittest.main()
