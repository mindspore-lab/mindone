import os
from math import sqrt
from typing import Optional

import numpy as np
import pytest
import torch
from transformers.integrations.flash_attention import flash_attention_forward as flash_attention_forward_transformers

from mindspore import grad, jit, mint, set_seed, tensor

from mindone.transformers.integrations.flash_attention import flash_attention_forward
from tests.modeling_test_utils import MS_DTYPE_MAPPING, PT_DTYPE_MAPPING

DTYPE_AND_THRESHOLDS = {"fp32": 1e-6, "fp16": 5e-4, "bf16": 5e-3}

flash_attention_forward_jit = jit(flash_attention_forward, jit_level="O1")


class MockConfig:
    def __init__(self, attn_implementation):
        self._attn_implementation = attn_implementation


class MockAttentionModule:
    def __init__(self, is_causal, attn_implementation):
        self.is_causal = is_causal
        self.config = MockConfig(attn_implementation)


@pytest.fixture(scope="module")
def q_k_v_target() -> dict[str, tuple]:
    # B, H, S, D
    set_seed(42)
    q = np.random.uniform(size=(2, 8, 256, 32)).astype(np.float32)
    k = np.random.uniform(size=(2, 8, 256, 32)).astype(np.float32)
    v = np.random.uniform(size=(2, 8, 256, 32)).astype(np.float32)
    target = np.random.uniform(size=(2, 256, 8, 32)).astype(np.float32)
    return {
        "ms": (tensor(q), tensor(k), tensor(v), tensor(target)),
        "pt": (torch.tensor(q), torch.tensor(k), torch.tensor(v), torch.tensor(target)),
    }


def cast_inputs(q, k, v, target, dtype, device: Optional[str] = None):
    q, k, v, target = q.to(dtype), k.to(dtype), v.to(dtype), target.to(dtype)
    if device is not None:
        q, k, v, target = q.to(device), k.to(device), v.to(device), target.to(device)
    return q, k, v, target


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])  # FlashAttention doesn't support fp32
@pytest.mark.parametrize("causal", [False, True], ids=["non-causal", "causal"])
@pytest.mark.parametrize("jit_compile", [False, True], ids=["eager", "jit"])
def test_fa_attention_forward(q_k_v_target, dtype: str, causal: bool, jit_compile):
    module = MockAttentionModule(is_causal=causal, attn_implementation="flash_attention_2")

    if torch.cuda.is_available():  # CUDA device
        if jit_compile:
            pytest.skip("Generate PyTorch outputs only for eager mode.")
        q_pt, k_pt, v_pt, _ = cast_inputs(*q_k_v_target["pt"], dtype=PT_DTYPE_MAPPING[dtype], device="cuda")
        output_pt = flash_attention_forward_transformers(module, q_pt, k_pt, v_pt, None)[0]

        torch.save(output_pt.cpu(), f"fa_attn_fwd_{dtype}_{causal}.pt")
        assert os.path.exists(f"fa_attn_fwd_{dtype}_{causal}.pt"), "Failed to save PyTorch outputs."

    else:  # Ascend device
        q, k, v, _ = cast_inputs(*q_k_v_target["ms"], dtype=MS_DTYPE_MAPPING[dtype])
        scaling = 1 / sqrt(q.shape[-1])

        fa_forward = flash_attention_forward_jit if jit_compile else flash_attention_forward
        output = fa_forward(module, q, k, v, None, scaling=scaling)[0]
        output_pt = torch.load(f"fa_attn_fwd_{dtype}_{causal}.pt")

        assert output.shape == output_pt.shape, f"Shape mismatch: {output.shape} vs {output_pt.shape}"
        assert not output.isnan().any(), "Output contains NaNs."
        assert np.allclose(
            output.numpy().astype(np.float32), output_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
        )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["non-causal", "causal"])
@pytest.mark.parametrize("jit_compile", [False, True], ids=["eager", "jit"])
def test_fa_attention_backward(q_k_v_target, dtype: str, causal: bool, jit_compile):
    module = MockAttentionModule(is_causal=causal, attn_implementation="flash_attention_2")

    if torch.cuda.is_available():  # CUDA device
        if jit_compile:
            pytest.skip("Generate PyTorch outputs only for eager mode.")
        q_pt, k_pt, v_pt, target = cast_inputs(*q_k_v_target["pt"], dtype=PT_DTYPE_MAPPING[dtype], device="cuda")
        q_pt.requires_grad, k_pt.requires_grad, v_pt.requires_grad = (True,) * 3

        output_pt = flash_attention_forward_transformers(module, q_pt, k_pt, v_pt, None)[0]
        loss = torch.nn.functional.mse_loss(output_pt, target)
        loss.backward()

        grad_out_pt = torch.stack([q_pt.grad, k_pt.grad, v_pt.grad], dim=0)
        torch.save(grad_out_pt.cpu(), f"fa_attn_bwd_{dtype}_{causal}.pt")
        assert os.path.exists(f"fa_attn_bwd_{dtype}_{causal}.pt"), "Failed to save PyTorch outputs."

    else:  # Ascend device
        q, k, v, target = cast_inputs(*q_k_v_target["ms"], dtype=MS_DTYPE_MAPPING[dtype])
        scaling = 1 / sqrt(q.shape[-1])

        fa_forward = flash_attention_forward_jit if jit_compile else flash_attention_forward

        def _forward(q_, k_, v_, target_):
            output = fa_forward(module, q_, k_, v_, None, scaling=scaling)[0]
            return mint.nn.functional.mse_loss(output, target_)

        grad_out = grad(_forward, grad_position=(0, 1, 2))(q, k, v, target)
        grad_out = mint.stack(grad_out, dim=0)

        grad_out_pt = torch.load(f"fa_attn_bwd_{dtype}_{causal}.pt")

        assert grad_out.shape == grad_out_pt.shape, f"Shape mismatch: {grad_out.shape} vs {grad_out_pt.shape}"
        assert not grad_out.isnan().any(), "Output contains NaNs."
        assert np.allclose(
            grad_out.numpy().astype(np.float32), grad_out_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
        )
