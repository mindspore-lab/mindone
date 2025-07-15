import unittest

import numpy as np
import torch
from transformers import LEDConfig
from transformers.models.led.modeling_led import LEDEncoderSelfAttention as TorchLEDEncoderSelfAttention

import mindspore
from mindspore import Tensor

from mindone.transformers.models.led.modeling_led import LEDEncoderSelfAttention as MSLEDEncoderSelfAttention


class TestLEDEncoderSelfAttention(unittest.TestCase):
    def setUp(self):
        # Common config for both implementations
        self.config = LEDConfig(
            hidden_size=32,
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
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        mindspore.set_seed(42)

        # Create input tensors
        batch_size, seq_len = 2, 32
        hidden_states = np.random.randn(batch_size, seq_len, self.config.hidden_size)

        # Convert to respective framework tensors
        ms_hidden_states = Tensor(hidden_states, dtype=mindspore.float32)
        torch_hidden_states = torch.tensor(hidden_states, dtype=torch.float32)

        # Create attention masks
        attention_mask = np.ones((batch_size, seq_len))
        ms_attention_mask = Tensor(attention_mask, dtype=mindspore.float32)
        torch_attention_mask = torch.tensor(attention_mask, dtype=torch.float32)

        torch_is_index_masked = torch.tensor(
            [False] * (seq_len // 2) + [True] * (seq_len // 2), dtype=torch.bool
        ).expand(batch_size, seq_len)
        ms_is_index_masked = Tensor(
            [False] * (seq_len // 2) + [True] * (seq_len // 2), dtype=mindspore.bool_
        ).broadcast_to((batch_size, seq_len))

        torch_is_index_global_attn = torch.tensor([False] * seq_len, dtype=torch.bool).expand(batch_size, seq_len)
        ms_is_index_global_attn = Tensor([False] * seq_len, dtype=mindspore.bool_).broadcast_to((batch_size, seq_len))

        # Copy weights to ensure both models have identical parameters
        self.copy_weights()

        # Forward pass
        ms_output = self.ms_attn(
            ms_hidden_states,
            attention_mask=ms_attention_mask,
            layer_head_mask=None,
            is_index_masked=ms_is_index_masked,
            is_index_global_attn=ms_is_index_global_attn,
            output_attentions=False,
        )

        torch_output = self.torch_attn(
            torch_hidden_states,
            attention_mask=torch_attention_mask,
            layer_head_mask=None,
            is_index_masked=torch_is_index_masked,
            is_index_global_attn=torch_is_index_global_attn,
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
        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        mindspore.set_seed(42)

        # Create input tensors
        batch_size, seq_len = 2, 32
        hidden_states = np.random.randn(batch_size, seq_len, self.config.hidden_size)

        # Convert to respective framework tensors
        ms_hidden_states = Tensor(hidden_states, dtype=mindspore.float32)
        torch_hidden_states = torch.tensor(hidden_states, dtype=torch.float32)

        # Copy weights to ensure both models have identical parameters
        self.copy_weights()

        ms_hidden_states = ms_hidden_states.transpose(0, 1)
        torch_hidden_states = torch_hidden_states.transpose(0, 1)
        np.testing.assert_allclose(
            ms_hidden_states.asnumpy(), torch_hidden_states.detach().numpy(), rtol=1e-4, atol=1e-4
        )

        # project hidden states
        ms_query_vectors = self.ms_attn.query(ms_hidden_states)
        ms_key_vectors = self.ms_attn.key(ms_hidden_states)
        ms_value_vectors = self.ms_attn.value(ms_hidden_states)

        torch_query_vectors = self.torch_attn.query(torch_hidden_states)
        torch_key_vectors = self.torch_attn.key(torch_hidden_states)
        torch_value_vectors = self.torch_attn.value(torch_hidden_states)

        # compare query, key, value
        np.testing.assert_allclose(
            ms_query_vectors.asnumpy(), torch_query_vectors.detach().numpy(), rtol=1e-4, atol=1e-4
        )
        np.testing.assert_allclose(ms_key_vectors.asnumpy(), torch_key_vectors.detach().numpy(), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            ms_value_vectors.asnumpy(), torch_value_vectors.detach().numpy(), rtol=1e-4, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
