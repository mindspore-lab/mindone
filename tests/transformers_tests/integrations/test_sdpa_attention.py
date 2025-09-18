import numpy as np
import pytest
import torch
from transformers.integrations.sdpa_attention import sdpa_attention_forward as sdpa_attention_forward_transformers

from mindspore import grad, mint, set_seed, tensor

from mindone.transformers.integrations.sdpa_attention import sdpa_attention_forward
from tests.modeling_test_utils import MS_DTYPE_MAPPING, PT_DTYPE_MAPPING

DTYPE_AND_THRESHOLDS = {"fp32": 1e-6, "fp16": 2e-3, "bf16": 2e-2}


class MockAttentionModule:
    def __init__(self, is_causal):
        self.is_causal = is_causal
        self.training = False


@pytest.fixture(scope="module")
def q_k_v_target_mask() -> dict[str, tuple]:
    # B, H, S, D
    set_seed(42)
    q = np.random.uniform(size=(2, 8, 256, 32)).astype(np.float32)
    k = np.random.uniform(size=(2, 8, 256, 32)).astype(np.float32)
    v = np.random.uniform(size=(2, 8, 256, 32)).astype(np.float32)
    target = np.random.uniform(size=(2, 256, 8, 32)).astype(np.float32)
    attention_mask = np.random.randint(0, 2, (q.shape[0], 1, q.shape[2], k.shape[2]), dtype=bool)
    return {
        "ms": (tensor(q), tensor(k), tensor(v), tensor(target), tensor(attention_mask)),
        "pt": (torch.tensor(q), torch.tensor(k), torch.tensor(v), torch.tensor(target), torch.tensor(attention_mask)),
    }


@pytest.fixture(scope="module")
def q_k_v_target_float_mask() -> dict[str, tuple]:
    # B, H, S, D
    set_seed(42)
    q = np.random.uniform(size=(2, 8, 256, 32)).astype(np.float32)
    k = np.random.uniform(size=(2, 8, 256, 32)).astype(np.float32)
    v = np.random.uniform(size=(2, 8, 256, 32)).astype(np.float32)
    target = np.random.uniform(size=(2, 256, 8, 32)).astype(np.float32)
    attention_mask = np.random.randint(0, 2, (q.shape[0], 1, q.shape[2], k.shape[2]))
    attention_mask = np.where(attention_mask == 1, float("-inf"), attention_mask).astype(np.float32)
    return {
        "ms": (tensor(q), tensor(k), tensor(v), tensor(target), tensor(attention_mask)),
        "pt": (torch.tensor(q), torch.tensor(k), torch.tensor(v), torch.tensor(target), torch.tensor(attention_mask)),
    }


def cast_inputs(q, k, v, target, attention_mask, dtype):
    return (
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        target.to(dtype),
        attention_mask,  # no casting needed for attention_mask
    )


def cast_inputs_float_attn_mask(q, k, v, target, attention_mask, dtype):
    return (
        q.to(dtype),
        k.to(dtype),
        v.to(dtype),
        target.to(dtype),
        attention_mask.to(dtype),
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("use_mask", [True, False], ids=["with_mask", "without_mask"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_forward(q_k_v_target_mask, use_mask: bool, dtype: str, jit: bool):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=False)

    q, k, v, _, attn_mask = cast_inputs(*q_k_v_target_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt, _, attn_mask_pt = cast_inputs(*q_k_v_target_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype])

    output = sdpa_attention_forward(module, q, k, v, attention_mask=attn_mask if use_mask else None, is_causal=False)[0]
    output_pt = sdpa_attention_forward_transformers(
        module, q_pt, k_pt, v_pt, attention_mask=attn_mask_pt if use_mask else None, is_causal=False
    )[0]

    assert output.shape == output_pt.shape, f"Shape mismatch: {output.shape} vs {output_pt.shape}"
    assert not output.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        output.numpy().astype(np.float32), output_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("use_mask", [True, False], ids=["with_mask", "without_mask"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_backward(q_k_v_target_mask, use_mask: bool, dtype: str, jit: bool):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=False)

    # MindONE
    q, k, v, target, attn_mask = cast_inputs(*q_k_v_target_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])

    def _forward(q_, k_, v_, target_):
        output = sdpa_attention_forward(
            module, q_, k_, v_, attention_mask=attn_mask if use_mask else None, is_causal=False
        )[0]
        return mint.nn.functional.mse_loss(output, target_)

    grad_out = grad(_forward, grad_position=(0, 1, 2))(q, k, v, target)
    grad_out = mint.stack(grad_out, dim=0)

    # Transformers
    q_pt, k_pt, v_pt, target_pt, attn_mask_pt = cast_inputs(*q_k_v_target_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt = q_pt.clone(), k_pt.clone(), v_pt.clone()
    q_pt.requires_grad, k_pt.requires_grad, v_pt.requires_grad = (True,) * 3

    output_pt = sdpa_attention_forward_transformers(
        module, q_pt, k_pt, v_pt, attention_mask=attn_mask_pt if use_mask else None, is_causal=False
    )[0]
    loss = torch.nn.functional.mse_loss(output_pt, target_pt)
    loss.backward()
    grad_out_pt = torch.stack([q_pt.grad, k_pt.grad, v_pt.grad], dim=0)

    assert grad_out.shape == grad_out_pt.shape, f"Shape mismatch: {grad_out.shape} vs {grad_out_pt.shape}"
    assert not grad_out.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        grad_out.numpy().astype(np.float32), grad_out_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_causal_forward(q_k_v_target_mask, dtype: str, jit: bool):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=True)

    q, k, v, *_ = cast_inputs(*q_k_v_target_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt, *_ = cast_inputs(*q_k_v_target_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype])

    output = sdpa_attention_forward(module, q, k, v, attention_mask=None)[0]
    output_pt = sdpa_attention_forward_transformers(module, q_pt, k_pt, v_pt, attention_mask=None)[0]

    assert output.shape == output_pt.shape, f"Shape mismatch: {output.shape} vs {output_pt.shape}"
    assert not output.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        output.numpy().astype(np.float32), output_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_causal_backward(q_k_v_target_mask, dtype: str, jit: bool):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=True)

    # MindONE
    q, k, v, target, _ = cast_inputs(*q_k_v_target_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])

    def _forward(q_, k_, v_, target_):
        output = sdpa_attention_forward(module, q_, k_, v_, attention_mask=None)[0]
        return mint.nn.functional.mse_loss(output, target_)

    grad_out = grad(_forward, grad_position=(0, 1, 2))(q, k, v, target)
    grad_out = mint.stack(grad_out, dim=0)

    # Transformers
    q_pt, k_pt, v_pt, target_pt, _ = cast_inputs(*q_k_v_target_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt = q_pt.clone(), k_pt.clone(), v_pt.clone()
    q_pt.requires_grad, k_pt.requires_grad, v_pt.requires_grad = (True,) * 3

    output_pt = sdpa_attention_forward_transformers(module, q_pt, k_pt, v_pt, attention_mask=None)[0]
    loss = torch.nn.functional.mse_loss(output_pt, target_pt)
    loss.backward()
    grad_out_pt = torch.stack([q_pt.grad, k_pt.grad, v_pt.grad], dim=0)

    assert grad_out.shape == grad_out_pt.shape, f"Shape mismatch: {grad_out.shape} vs {grad_out_pt.shape}"
    assert not grad_out.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        grad_out.numpy().astype(np.float32), grad_out_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("use_mask", [True, False], ids=["with_mask", "without_mask"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_float_attn_mask_forward(q_k_v_target_float_mask, use_mask: bool, dtype: str, jit: bool):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=False)

    q, k, v, _, attn_mask = cast_inputs_float_attn_mask(*q_k_v_target_float_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt, _, attn_mask_pt = cast_inputs_float_attn_mask(
        *q_k_v_target_float_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype]
    )

    output = sdpa_attention_forward(module, q, k, v, attention_mask=attn_mask if use_mask else None, is_causal=False)[0]
    output_pt = sdpa_attention_forward_transformers(
        module, q_pt, k_pt, v_pt, attention_mask=attn_mask_pt if use_mask else None, is_causal=False
    )[0]

    assert output.shape == output_pt.shape, f"Shape mismatch: {output.shape} vs {output_pt.shape}"
    assert not output.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        output.numpy().astype(np.float32), output_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("use_mask", [True, False], ids=["with_mask", "without_mask"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_float_attn_mask_backward(q_k_v_target_float_mask, use_mask: bool, dtype: str, jit: bool):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=False)

    # MindONE
    q, k, v, target, attn_mask = cast_inputs_float_attn_mask(
        *q_k_v_target_float_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype]
    )

    def _forward(q_, k_, v_, target_):
        output = sdpa_attention_forward(
            module, q_, k_, v_, attention_mask=attn_mask if use_mask else None, is_causal=False
        )[0]
        return mint.nn.functional.mse_loss(output, target_)

    grad_out = grad(_forward, grad_position=(0, 1, 2))(q, k, v, target)
    grad_out = mint.stack(grad_out, dim=0)

    # Transformers
    q_pt, k_pt, v_pt, target_pt, attn_mask_pt = cast_inputs_float_attn_mask(
        *q_k_v_target_float_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype]
    )
    q_pt, k_pt, v_pt = q_pt.clone(), k_pt.clone(), v_pt.clone()
    q_pt.requires_grad, k_pt.requires_grad, v_pt.requires_grad = (True,) * 3

    output_pt = sdpa_attention_forward_transformers(
        module, q_pt, k_pt, v_pt, attention_mask=attn_mask_pt if use_mask else None, is_causal=False
    )[0]
    loss = torch.nn.functional.mse_loss(output_pt, target_pt)
    loss.backward()
    grad_out_pt = torch.stack([q_pt.grad, k_pt.grad, v_pt.grad], dim=0)

    assert grad_out.shape == grad_out_pt.shape, f"Shape mismatch: {grad_out.shape} vs {grad_out_pt.shape}"
    assert not grad_out.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        grad_out.numpy().astype(np.float32), grad_out_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("use_mask", [True, False], ids=["with_mask", "without_mask"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_causal_with_attn_mask_forward(q_k_v_target_mask, use_mask: bool, dtype: str, jit: bool):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=True)

    q, k, v, _, attn_mask = cast_inputs(*q_k_v_target_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt, _, attn_mask_pt = cast_inputs(*q_k_v_target_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype])

    output = sdpa_attention_forward(module, q, k, v, attention_mask=attn_mask if use_mask else None, is_causal=True)[0]
    output_pt = sdpa_attention_forward_transformers(
        module, q_pt, k_pt, v_pt, attention_mask=attn_mask_pt if use_mask else None, is_causal=True
    )[0]

    assert output.shape == output_pt.shape, f"Shape mismatch: {output.shape} vs {output_pt.shape}"
    assert not output.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        output.numpy().astype(np.float32), output_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("use_mask", [True, False], ids=["with_mask", "without_mask"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_causal_with_attn_mask_backward(q_k_v_target_mask, use_mask: bool, dtype: str, jit: bool):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=True)

    # MindONE
    q, k, v, target, attn_mask = cast_inputs(*q_k_v_target_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])

    def _forward(q_, k_, v_, target_):
        output = sdpa_attention_forward(
            module, q_, k_, v_, attention_mask=attn_mask if use_mask else None, is_causal=True
        )[0]
        return mint.nn.functional.mse_loss(output, target_)

    grad_out = grad(_forward, grad_position=(0, 1, 2))(q, k, v, target)
    grad_out = mint.stack(grad_out, dim=0)

    # Transformers
    q_pt, k_pt, v_pt, target_pt, attn_mask_pt = cast_inputs(*q_k_v_target_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt = q_pt.clone(), k_pt.clone(), v_pt.clone()
    q_pt.requires_grad, k_pt.requires_grad, v_pt.requires_grad = (True,) * 3

    output_pt = sdpa_attention_forward_transformers(
        module, q_pt, k_pt, v_pt, attention_mask=attn_mask_pt if use_mask else None, is_causal=True
    )[0]
    loss = torch.nn.functional.mse_loss(output_pt, target_pt)
    loss.backward()
    grad_out_pt = torch.stack([q_pt.grad, k_pt.grad, v_pt.grad], dim=0)

    assert grad_out.shape == grad_out_pt.shape, f"Shape mismatch: {grad_out.shape} vs {grad_out_pt.shape}"
    assert not grad_out.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        grad_out.numpy().astype(np.float32), grad_out_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("use_mask", [True, False], ids=["with_mask", "without_mask"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_causal_with_float_attn_mask_forward(
    q_k_v_target_float_mask, use_mask: bool, dtype: str, jit: bool
):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=True)

    q, k, v, _, attn_mask = cast_inputs_float_attn_mask(*q_k_v_target_float_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype])
    q_pt, k_pt, v_pt, _, attn_mask_pt = cast_inputs_float_attn_mask(
        *q_k_v_target_float_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype]
    )

    output = sdpa_attention_forward(module, q, k, v, attention_mask=attn_mask if use_mask else None, is_causal=True)[0]
    output_pt = sdpa_attention_forward_transformers(
        module, q_pt, k_pt, v_pt, attention_mask=attn_mask_pt if use_mask else None, is_causal=True
    )[0]

    assert output.shape == output_pt.shape, f"Shape mismatch: {output.shape} vs {output_pt.shape}"
    assert not output.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        output.numpy().astype(np.float32), output_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )


@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("use_mask", [True, False], ids=["with_mask", "without_mask"])
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_sdpa_attention_causal_with_float_attn_mask_backward(
    q_k_v_target_float_mask, use_mask: bool, dtype: str, jit: bool
):
    if jit:
        pytest.skip("`sdpa_attention_forward` can't be compiled with jit.")

    module = MockAttentionModule(is_causal=True)

    # MindONE
    q, k, v, target, attn_mask = cast_inputs_float_attn_mask(
        *q_k_v_target_float_mask["ms"], dtype=MS_DTYPE_MAPPING[dtype]
    )

    def _forward(q_, k_, v_, target_):
        output = sdpa_attention_forward(
            module, q_, k_, v_, attention_mask=attn_mask if use_mask else None, is_causal=True
        )[0]
        return mint.nn.functional.mse_loss(output, target_)

    grad_out = grad(_forward, grad_position=(0, 1, 2))(q, k, v, target)
    grad_out = mint.stack(grad_out, dim=0)

    # Transformers
    q_pt, k_pt, v_pt, target_pt, attn_mask_pt = cast_inputs_float_attn_mask(
        *q_k_v_target_float_mask["pt"], dtype=PT_DTYPE_MAPPING[dtype]
    )
    q_pt, k_pt, v_pt = q_pt.clone(), k_pt.clone(), v_pt.clone()
    q_pt.requires_grad, k_pt.requires_grad, v_pt.requires_grad = (True,) * 3

    output_pt = sdpa_attention_forward_transformers(
        module, q_pt, k_pt, v_pt, attention_mask=attn_mask_pt if use_mask else None, is_causal=True
    )[0]
    loss = torch.nn.functional.mse_loss(output_pt, target_pt)
    loss.backward()
    grad_out_pt = torch.stack([q_pt.grad, k_pt.grad, v_pt.grad], dim=0)

    assert grad_out.shape == grad_out_pt.shape, f"Shape mismatch: {grad_out.shape} vs {grad_out_pt.shape}"
    assert not grad_out.isnan().any(), "Output contains NaNs."
    assert np.allclose(
        grad_out.numpy().astype(np.float32), grad_out_pt.to(torch.float32).numpy(), atol=DTYPE_AND_THRESHOLDS[dtype]
    )
