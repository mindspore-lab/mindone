import unittest

import numpy as np
import torch

import mindspore


class TestEinsum(unittest.TestCase):
    def setUp(self):
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        mindspore.set_seed(42)

    def test_sliding_chunks_query_key_matmul_einsum(self):
        # Test parameters
        batch_size = 2
        num_heads = 4
        seq_len = 16  # Must be multiple of (window_overlap * 2)
        head_dim = 32
        window_overlap = 4

        # Create random input tensors
        query_np = np.random.randn(batch_size, seq_len, num_heads, head_dim)
        key_np = np.random.randn(batch_size, seq_len, num_heads, head_dim)

        # Convert to PyTorch tensors
        query_torch = torch.from_numpy(query_np).float()
        key_torch = torch.from_numpy(key_np).float()

        # Convert to MindSpore tensors
        query_ms = mindspore.Tensor(query_np, dtype=mindspore.float32)
        key_ms = mindspore.Tensor(key_np, dtype=mindspore.float32)

        # Reshape tensors as done in the original code
        query_torch = query_torch.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key_torch = key_torch.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        query_ms = query_ms.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key_ms = key_ms.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        # Chunk the tensors (simplified chunking for test)
        chunks = seq_len // (window_overlap * 2)
        query_chunks_torch = query_torch.view(batch_size * num_heads, chunks, 2 * window_overlap, head_dim)
        key_chunks_torch = key_torch.view(batch_size * num_heads, chunks, 2 * window_overlap, head_dim)

        query_chunks_ms = query_ms.view(batch_size * num_heads, chunks, 2 * window_overlap, head_dim)
        key_chunks_ms = key_ms.view(batch_size * num_heads, chunks, 2 * window_overlap, head_dim)

        np.testing.assert_allclose(query_chunks_torch.detach().numpy(), query_chunks_ms.asnumpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(key_chunks_torch.detach().numpy(), key_chunks_ms.asnumpy(), rtol=1e-5, atol=1e-5)

        torch_output = torch.einsum("bcxd,bcyd->bcxy", query_chunks_torch, key_chunks_torch)
        ms_output = mindspore.mint.einsum("bcxd,bcyd->bcxy", query_chunks_ms, key_chunks_ms)

        # Convert outputs to numpy for comparison
        torch_output_np = torch_output.detach().numpy()
        ms_output_np = ms_output.asnumpy()

        # Compare outputs
        np.testing.assert_allclose(torch_output_np, ms_output_np, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
