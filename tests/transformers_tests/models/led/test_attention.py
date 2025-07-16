import math
import unittest

import numpy as np
import torch
from transformers import LEDConfig
from transformers.models.led.modeling_led import LEDEncoderSelfAttention as TorchLEDEncoderSelfAttention

import mindspore
from mindspore import Tensor

from mindone.transformers.models.led.modeling_led import LEDEncoderSelfAttention as MSLEDEncoderSelfAttention
from mindone.transformers.models.led.modeling_led import as_strided


class TestLEDEncoderSelfAttention(unittest.TestCase):
    def setUp(self):
        # Common config for both implementations
        self.config = LEDConfig(
            hidden_size=768,
            num_attention_heads=4,
            attention_probs_dropout_prob=0.0,  # Set to 0 for deterministic comparison
            attention_window=[16, 16, 16, 16, 16, 16],  # Assuming 6 layers
        )
        self.layer_id = 0

        # Initialize both models
        self.ms_attn = MSLEDEncoderSelfAttention(self.config, self.layer_id)
        self.torch_attn = TorchLEDEncoderSelfAttention(self.config, self.layer_id)

        # Set both models to eval mode
        self.ms_attn.set_train(False)
        self.torch_attn.eval()
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        mindspore.set_seed(42)

        self.copy_weights()

    def load_inputs(self):
        # Create input tensors
        hidden_states = np.load("layer0_ms/hidden_states.npy")
        attention_mask = np.load("layer0_ms/attention_mask.npy")
        is_index_masked = np.load("layer0_ms/is_index_masked.npy")
        is_index_global_attn = np.load("layer0_ms/is_index_global_attn.npy")
        return hidden_states, attention_mask, is_index_masked, is_index_global_attn

    def get_qkv(self):
        hidden_states = self.load_inputs()[0]
        # Convert to respective framework tensors
        ms_hidden_states = Tensor(hidden_states, dtype=mindspore.float32)
        torch_hidden_states = torch.tensor(hidden_states, dtype=torch.float32)

        ms_hidden_states = ms_hidden_states.transpose(0, 1)
        torch_hidden_states = torch_hidden_states.transpose(0, 1)

        # project hidden states
        ms_query_vectors = self.ms_attn.query(ms_hidden_states)
        ms_key_vectors = self.ms_attn.key(ms_hidden_states)
        ms_value_vectors = self.ms_attn.value(ms_hidden_states)

        torch_query_vectors = self.torch_attn.query(torch_hidden_states)
        torch_key_vectors = self.torch_attn.key(torch_hidden_states)
        torch_value_vectors = self.torch_attn.value(torch_hidden_states)

        seq_len, batch_size, _ = ms_hidden_states.shape

        # normalize query
        ms_query_vectors /= math.sqrt(self.ms_attn.head_dim)

        ms_query_vectors = ms_query_vectors.view(
            seq_len, batch_size, self.ms_attn.num_heads, self.ms_attn.head_dim
        ).transpose(0, 1)
        ms_key_vectors = ms_key_vectors.view(
            seq_len, batch_size, self.ms_attn.num_heads, self.ms_attn.head_dim
        ).transpose(0, 1)

        torch_query_vectors /= math.sqrt(self.torch_attn.head_dim)

        torch_query_vectors = torch_query_vectors.view(
            seq_len, batch_size, self.torch_attn.num_heads, self.torch_attn.head_dim
        ).transpose(0, 1)
        torch_key_vectors = torch_key_vectors.view(
            seq_len, batch_size, self.torch_attn.num_heads, self.torch_attn.head_dim
        ).transpose(0, 1)

        return (
            ms_query_vectors,
            ms_key_vectors,
            ms_value_vectors,
            torch_query_vectors,
            torch_key_vectors,
            torch_value_vectors,
        )

    def copy_weights(self):
        """Copy weights from PyTorch model to MindSpore model"""

        # Helper function to convert torch tensor to mindspore parameter
        def torch_to_ms_tensor(torch_tensor):
            return Tensor(torch_tensor.detach().numpy(), dtype=mindspore.float32)

        # Copy weights for query, key, value
        self.ms_attn.query.weight.set_data(torch_to_ms_tensor(self.torch_attn.query.weight))
        self.ms_attn.key.weight.set_data(torch_to_ms_tensor(self.torch_attn.key.weight))
        self.ms_attn.value.weight.set_data(torch_to_ms_tensor(self.torch_attn.value.weight))

        # Copy bias for query, key, value
        self.ms_attn.query.bias.set_data(torch_to_ms_tensor(self.torch_attn.query.bias))
        self.ms_attn.key.bias.set_data(torch_to_ms_tensor(self.torch_attn.key.bias))
        self.ms_attn.value.bias.set_data(torch_to_ms_tensor(self.torch_attn.value.bias))

        # Copy weights for global attention
        self.ms_attn.query_global.weight.set_data(torch_to_ms_tensor(self.torch_attn.query_global.weight))
        self.ms_attn.key_global.weight.set_data(torch_to_ms_tensor(self.torch_attn.key_global.weight))
        self.ms_attn.value_global.weight.set_data(torch_to_ms_tensor(self.torch_attn.value_global.weight))

        # Copy bias for global attention
        self.ms_attn.query_global.bias.set_data(torch_to_ms_tensor(self.torch_attn.query_global.bias))
        self.ms_attn.key_global.bias.set_data(torch_to_ms_tensor(self.torch_attn.key_global.bias))
        self.ms_attn.value_global.bias.set_data(torch_to_ms_tensor(self.torch_attn.value_global.bias))

    def test_forward_pass(self):
        hidden_states, attention_mask, is_index_masked, is_index_global_attn = self.load_inputs()
        # Convert to respective framework tensors
        ms_hidden_states = Tensor(hidden_states, dtype=mindspore.float32)
        torch_hidden_states = torch.tensor(hidden_states, dtype=torch.float32)

        # Create attention masks
        ms_attention_mask = Tensor(attention_mask, dtype=mindspore.float32)
        torch_attention_mask = torch.tensor(attention_mask, dtype=torch.float32)

        torch_is_index_masked = torch.tensor(is_index_masked).bool()
        ms_is_index_masked = Tensor(is_index_masked).bool()
        torch_is_index_global_attn = torch.tensor(is_index_global_attn).bool()
        ms_is_index_global_attn = Tensor(is_index_global_attn).bool()

        # Forward pass
        ms_output = self.ms_attn(
            ms_hidden_states,
            attention_mask=ms_attention_mask,
            layer_head_mask=None,
            is_index_masked=ms_is_index_masked,
            is_index_global_attn=ms_is_index_global_attn,
            is_global_attn=False,
            output_attentions=False,
        )

        torch_output = self.torch_attn(
            torch_hidden_states,
            attention_mask=torch_attention_mask,
            layer_head_mask=None,
            is_index_masked=torch_is_index_masked,
            is_index_global_attn=torch_is_index_global_attn,
            is_global_attn=False,
            output_attentions=False,
        )

        # Convert outputs to numpy for comparison
        ms_output_np = ms_output[0].asnumpy()
        torch_output_np = torch_output[0].detach().numpy()

        # Compare outputs
        np.testing.assert_allclose(
            ms_output_np,
            torch_output_np,
            rtol=1e-4,
            atol=1e-4,
            err_msg="MindSpore and PyTorch implementations produce different outputs",
        )

    def test_projection(self):
        (
            ms_query_vectors,
            ms_key_vectors,
            ms_value_vectors,
            torch_query_vectors,
            torch_key_vectors,
            torch_value_vectors,
        ) = self.get_qkv()
        # Copy weights to ensure both models have identical parameters

        # compare query, key, value
        np.testing.assert_allclose(
            ms_query_vectors.asnumpy(), torch_query_vectors.detach().numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(ms_key_vectors.asnumpy(), torch_key_vectors.detach().numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            ms_value_vectors.asnumpy(), torch_value_vectors.detach().numpy(), rtol=1e-4, atol=1e-4
        )

    def test_attention_scores(self):
        (
            ms_query_vectors,
            ms_key_vectors,
            _,
            torch_query_vectors,
            torch_key_vectors,
            _,
        ) = self.get_qkv()

        ms_attn_scores = self.ms_attn._sliding_chunks_query_key_matmul(
            ms_query_vectors, ms_key_vectors, self.ms_attn.one_sided_attn_window_size
        )
        torch_attn_scores = self.torch_attn._sliding_chunks_query_key_matmul(
            torch_query_vectors, torch_key_vectors, self.torch_attn.one_sided_attn_window_size
        )

        np.testing.assert_allclose(ms_attn_scores.asnumpy(), torch_attn_scores.detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_sliding_chunk(self):
        (
            ms_query_vectors,
            ms_key_vectors,
            _,
            torch_query_vectors,
            torch_key_vectors,
            _,
        ) = self.get_qkv()

        batch_size, seq_len, num_heads, head_dim = ms_query_vectors.shape
        window_overlap = self.ms_attn.one_sided_attn_window_size
        assert self.ms_attn.one_sided_attn_window_size == self.torch_attn.one_sided_attn_window_size

        ms_query_vectors = ms_query_vectors.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        ms_key_vectors = ms_key_vectors.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        ms_query_vectors = self.ms_attn._chunk(
            ms_query_vectors, window_overlap, getattr(self.ms_attn.config, "onnx_export", False)
        )
        ms_key_vectors = self.ms_attn._chunk(
            ms_key_vectors, window_overlap, getattr(self.ms_attn.config, "onnx_export", False)
        )

        torch_query_vectors = torch_query_vectors.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        torch_key_vectors = torch_key_vectors.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        torch_query_vectors = self.torch_attn._chunk(
            torch_query_vectors, window_overlap, getattr(self.torch_attn.config, "onnx_export", False)
        )
        torch_key_vectors = self.torch_attn._chunk(
            torch_key_vectors, window_overlap, getattr(self.torch_attn.config, "onnx_export", False)
        )

        np.testing.assert_allclose(
            ms_query_vectors.asnumpy(), torch_query_vectors.detach().numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(ms_key_vectors.asnumpy(), torch_key_vectors.detach().numpy(), rtol=1e-4, atol=1e-4)

    def test_astrided(self):
        (
            ms_query_vectors,
            _,
            _,
            torch_query_vectors,
            _,
            _,
        ) = self.get_qkv()

        batch_size, seq_len, num_heads, head_dim = ms_query_vectors.shape
        window_overlap = self.ms_attn.one_sided_attn_window_size
        assert self.ms_attn.one_sided_attn_window_size == self.torch_attn.one_sided_attn_window_size

        ms_query_vectors = ms_query_vectors.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)

        ms_query_vectors = ms_query_vectors.view(
            ms_query_vectors.shape[0],
            mindspore.mint.div(ms_query_vectors.shape[1], (window_overlap * 2), rounding_mode="trunc").item(),
            window_overlap * 2,
            ms_query_vectors.shape[2],
        )

        ms_chunk_size = list(ms_query_vectors.shape)
        ms_chunk_size[1] = ms_chunk_size[1] * 2 - 1

        ms_chunk_stride = list(ms_query_vectors.stride())
        ms_chunk_stride[1] = ms_chunk_stride[1] // 2

        torch_query_vectors = torch_query_vectors.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        torch_query_vectors = torch_query_vectors.view(
            torch_query_vectors.shape[0],
            torch.div(torch_query_vectors.shape[1], (window_overlap * 2), rounding_mode="trunc"),
            window_overlap * 2,
            torch_query_vectors.shape[2],
        )

        torch_chunk_size = list(torch_query_vectors.shape)
        torch_chunk_size[1] = torch_chunk_size[1] * 2 - 1
        torch_chunk_stride = list(torch_query_vectors.stride())
        torch_chunk_stride[1] = torch_chunk_stride[1] // 2

        np.testing.assert_allclose(
            ms_query_vectors.asnumpy(), torch_query_vectors.detach().numpy(), rtol=1e-4, atol=1e-4
        )
        assert torch_chunk_size == ms_chunk_size
        assert torch_chunk_stride == ms_chunk_stride

        ms_query_vectors = as_strided(ms_query_vectors, size=ms_chunk_size, stride=ms_chunk_stride)
        torch_query_vectors = torch.as_strided(torch_query_vectors, size=torch_chunk_size, stride=torch_chunk_stride)

        np.testing.assert_allclose(
            ms_query_vectors.asnumpy(), torch_query_vectors.detach().numpy(), rtol=1e-4, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
