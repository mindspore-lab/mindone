import numpy as np
import pytest
from torch import float32 as torch_float32
from torch import tensor as ptensor
from transformers.integrations.sdpa_attention import sdpa_attention_forward as sdpa_attention_forward_transformers

from mindspore import jit, set_seed, tensor

from mindone.transformers.integrations.sdpa_attention import sdpa_attention_forward
from tests.modeling_test_utils import MS_DTYPE_MAPPING, PT_DTYPE_MAPPING

DTYPE_AND_THRESHOLDS = {"fp32": 1e-6, "fp16": 5e-3, "bf16": 5e-2}

graph_sdpa_attention_forward = jit(sdpa_attention_forward, jit_level="O1")


class MockAttentionModule:
    def __init__(self, is_causal):
        self.is_causal = is_causal
        self.training = False


@pytest.fixture(scope="module")
def q_k_v_mask() -> dict[str, tuple]:
    # B, H, S, D
    set_seed(42)
    q = np.random.randn(2, 8, 256, 32).astype(np.float32)
    k = np.random.randn(2, 8, 384, 32).astype(np.float32)
    v = np.random.randn(2, 8, 384, 32).astype(np.float32)
    attention_mask = np.random.randint(0, 2, (q.shape[0], 1, q.shape[2], k.shape[2]), dtype=bool)
    return {
        "ms": (tensor(q), tensor(k), tensor(v), tensor(attention_mask)),
        "pt": (ptensor(q), ptensor(k), ptensor(v), ptensor(attention_mask)),
    }


def cast_inputs(q, k, v, attention_mask, dtype):
    return q.to(dtype), k.to(dtype), v.to(dtype), attention_mask  # no casting needed for attention_mask


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("use_mask", [True, False], ids=["with_mask", "without_mask"])
@pytest.mark.parametrize("mode", ["PyNative", "Graph"])
def test_sdpa_attention_forward(q_k_v_mask, use_mask: bool, dtype: str, mode: str):
    sdpa_attention_forward_mindone = graph_sdpa_attention_forward if mode == "Graph" else sdpa_attention_forward

    module = MockAttentionModule(is_causal=False)

    q, k, v, attn_mask = cast_inputs(*q_k_v_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt, attn_mask_pt = cast_inputs(*q_k_v_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype])

    output = sdpa_attention_forward_mindone(
        module, q, k, v, attention_mask=attn_mask if use_mask else None, is_causal=False
    )[0]
    output_pt = sdpa_attention_forward_transformers(
        module, q_pt, k_pt, v_pt, attention_mask=attn_mask_pt if use_mask else None, is_causal=False
    )[0]

    assert output.shape == output_pt.shape, f"Shape mismatch: {output.shape} vs {output_pt.shape}"
    assert np.allclose(
        output.numpy().astype(np.float32), output_pt.to(torch_float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("mode", ["PyNative", "Graph"])
def test_sdpa_attention_forward_causal(q_k_v_mask, dtype: str, mode: str):
    sdpa_attention_forward_mindone = graph_sdpa_attention_forward if mode == "Graph" else sdpa_attention_forward

    module = MockAttentionModule(is_causal=True)

    q, k, v, _ = cast_inputs(*q_k_v_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt, _ = cast_inputs(*q_k_v_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype])

    output = sdpa_attention_forward_mindone(module, q, k, v, attention_mask=None)[0]
    output_pt = sdpa_attention_forward_transformers(module, q_pt, k_pt, v_pt, attention_mask=None)[0]

    assert output.shape == output_pt.shape, f"Shape mismatch: {output.shape} vs {output_pt.shape}"
    assert np.allclose(
        output.numpy().astype(np.float32), output_pt.to(torch_float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )
