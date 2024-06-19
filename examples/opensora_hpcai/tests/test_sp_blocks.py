import numpy as np
import pytest

import mindspore as ms

ms.set_context(mode=ms.GRAPH_MODE)
ms.set_seed(0)


def test_llama_rms_norm():
    from opensora.models.layers.blocks import LlamaRMSNorm, SeqParallelLlamaRMSNorm

    net1 = LlamaRMSNorm(128)
    net2 = SeqParallelLlamaRMSNorm(128)

    x = ms.Tensor(np.random.rand(4, 64, 128), dtype=ms.float32)
    y1 = net1(x)
    y2 = net2(x)
    assert y1.dtype == y2.dtype
    np.testing.assert_allclose(y1.asnumpy(), y2.asnumpy(), atol=1e-5)


@pytest.mark.parametrize("enable_flash_attention", [False, True])
@pytest.mark.parametrize("flash_attention_dtype", [ms.bfloat16, ms.float16])
def test_multihead_cross_attention(enable_flash_attention, flash_attention_dtype):
    from opensora.models.layers.blocks import MultiHeadCrossAttention, SeqParallelMultiHeadCrossAttention

    net1 = MultiHeadCrossAttention(
        1152, 16, enable_flash_attention=enable_flash_attention, flash_attention_dtype=flash_attention_dtype
    )
    net2 = SeqParallelMultiHeadCrossAttention(
        1152, 16, enable_flash_attention=enable_flash_attention, flash_attention_dtype=flash_attention_dtype
    )

    x1 = ms.Tensor(np.random.rand(2, 6480, 1152), dtype=ms.float32)
    x2 = ms.Tensor(np.random.rand(1, 400, 1152), dtype=ms.float32)
    mask = ms.Tensor(np.random.rand(2, 200) > 0.5, dtype=ms.uint8)

    net2.q_linear.weight.set_data(net1.q_linear.weight)
    net2.q_linear.bias.set_data(net1.q_linear.bias)
    k_weight, v_weight = net1.kv_linear.weight.chunk(2)
    k_bias, v_bias = net1.kv_linear.bias.chunk(2)
    net2.k_linear.weight.set_data(k_weight)
    net2.v_linear.weight.set_data(v_weight)
    net2.k_linear.bias.set_data(k_bias)
    net2.v_linear.bias.set_data(v_bias)
    net2.proj.weight.set_data(net1.proj.weight)
    net2.proj.bias.set_data(net1.proj.bias)

    y1 = net1(x1, x2, mask=mask)
    y2 = net2(x1, x2, mask=mask)

    assert y1.dtype == y2.dtype
    if enable_flash_attention:
        if flash_attention_dtype == ms.bfloat16:
            np.testing.assert_allclose(y1.asnumpy(), y2.asnumpy(), atol=1e-2)
        else:
            np.testing.assert_allclose(y1.asnumpy(), y2.asnumpy(), atol=1e-3)
    else:
        np.testing.assert_allclose(y1.asnumpy(), y2.asnumpy(), atol=1e-5)


@pytest.mark.parametrize("enable_flash_attention", [False, True])
@pytest.mark.parametrize("flash_attention_dtype", [ms.bfloat16, ms.float16])
def test_self_attention(enable_flash_attention, flash_attention_dtype):
    from opensora.models.layers.blocks import SelfAttention, SeqParallelSelfAttention
    from opensora.models.layers.rotary_embedding import RotaryEmbedding, RotaryEmbeddingSP

    rope = RotaryEmbedding(dim=1152 // 16).rotate_queries_or_keys
    net1 = SelfAttention(
        1152,
        16,
        qkv_bias=True,
        qk_norm=True,
        enable_flash_attention=enable_flash_attention,
        flash_attention_dtype=flash_attention_dtype,
        rope=rope,
    )

    rope = lambda: RotaryEmbeddingSP(dim=1152 // 16)
    net2 = SeqParallelSelfAttention(
        1152,
        16,
        qkv_bias=True,
        qk_norm=True,
        enable_flash_attention=enable_flash_attention,
        flash_attention_dtype=flash_attention_dtype,
        rope=rope,
    )

    q_weight, k_weight, v_weight = net1.qkv.weight.chunk(3)
    q_bias, k_bias, v_bias = net1.qkv.bias.chunk(3)
    net2.q_linear.weight.set_data(q_weight)
    net2.k_linear.weight.set_data(k_weight)
    net2.v_linear.weight.set_data(v_weight)
    net2.q_linear.bias.set_data(q_bias)
    net2.k_linear.bias.set_data(k_bias)
    net2.v_linear.bias.set_data(v_bias)
    net2.proj.weight.set_data(net1.proj.weight)
    net2.proj.bias.set_data(net1.proj.bias)

    x = ms.Tensor(np.random.rand(2, 6480, 1152), dtype=ms.float32)

    y1 = net1(x)
    y2 = net2(x)

    assert y1.dtype == y2.dtype
    if enable_flash_attention:
        if flash_attention_dtype == ms.bfloat16:
            np.testing.assert_allclose(y1.asnumpy(), y2.asnumpy(), atol=1e-2)
        else:
            np.testing.assert_allclose(y1.asnumpy(), y2.asnumpy(), atol=1e-3)
    else:
        np.testing.assert_allclose(y1.asnumpy(), y2.asnumpy(), atol=1e-5)


def test_t2i_final_layer():
    from opensora.models.layers.blocks import SeqParallelT2IFinalLayer, T2IFinalLayer

    net1 = T2IFinalLayer(1152, 4, 8)
    net2 = SeqParallelT2IFinalLayer(1152, 4, 8)

    x = ms.Tensor(np.random.rand(2, 6480, 1152), dtype=ms.float32)
    t = ms.Tensor(np.random.rand(2, 1152), dtype=ms.float32)
    frame_mask = ms.Tensor(np.random.rand(2, 16) > 0.5, dtype=ms.bool_)
    t0 = ms.Tensor(np.random.rand(2, 1152), dtype=ms.float32)
    T = 16
    S = 405

    net2.linear.weight.set_data(net1.linear.weight)
    net2.linear.bias.set_data(net1.linear.bias)
    net2.scale_shift_table.set_data(net1.scale_shift_table[None])

    y1 = net1(x, t, frame_mask, t0, T, S)
    y2 = net2(x, t, frame_mask, t0, T, S)
    assert y1.dtype == y2.dtype
    np.testing.assert_allclose(y1.asnumpy(), y2.asnumpy(), atol=1e-5)


def test_mlp():
    from opensora.models.layers.blocks import Mlp, SeqParallelMLP

    net1 = Mlp(1152, 4608)
    net2 = SeqParallelMLP(1152, 4608)

    x = ms.Tensor(np.random.rand(2, 6480, 1152), dtype=ms.float32)

    net2.fc1.weight.set_data(net1.fc1.weight)
    net2.fc1.bias.set_data(net1.fc1.bias)
    net2.fc2.weight.set_data(net1.fc2.weight)
    net2.fc2.bias.set_data(net1.fc2.bias)

    y1 = net1(x)
    y2 = net2(x)
    assert y1.dtype == y2.dtype
    np.testing.assert_allclose(y1.asnumpy(), y2.asnumpy(), atol=1e-5)
