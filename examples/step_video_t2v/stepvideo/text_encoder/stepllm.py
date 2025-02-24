# Copyright 2025 StepFun Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# ==============================================================================
import os
from typing import Optional

from stepvideo.modules.normalization import RMSNorm
from stepvideo.text_encoder.flashattention import StepAttention
from stepvideo.text_encoder.tokenizer import LLaMaEmbedding, Wrapped_StepChatTokenizer
from transformers.modeling_utils import PretrainedConfig

import mindspore as ms
from mindspore import Tensor, mint, nn, ops

from mindone.transformers import MSPreTrainedModel as PreTrainedModel


def safediv(n, d):
    q, r = divmod(n, d)
    assert r == 0
    return q


class MultiQueryAttention(nn.Cell):
    def __init__(self, cfg, layer_id=None):
        super().__init__()

        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.max_seq_len = cfg.seq_length
        self.use_flash_attention = cfg.use_flash_attn
        assert self.use_flash_attention, "FlashAttention is required!"

        self.n_groups = cfg.num_attention_groups
        self.tp_size = 1
        self.n_local_heads = cfg.num_attention_heads
        self.n_local_groups = self.n_groups

        self.wqkv = mint.nn.Linear(
            cfg.hidden_size,
            cfg.hidden_size + self.head_dim * 2 * self.n_groups,
            bias=False,
        )
        self.wo = mint.nn.Linear(
            cfg.hidden_size,
            cfg.hidden_size,
            bias=False,
        )

        assert self.use_flash_attention, "non-Flash attention not supported yet."
        self.core_attention = StepAttention()

        self.layer_id = layer_id

    def construct(
        self,
        x: Tensor,
        mask: Optional[Tensor],
        cu_seqlens: Optional[Tensor],
        max_seq_len: Optional[Tensor],
    ):
        # import pdb;pdb.set_trace()

        seqlen, bsz, dim = x.shape
        xqkv = self.wqkv(x)

        xq, xkv = mint.split(
            xqkv,
            (dim // self.tp_size, self.head_dim * 2 * self.n_groups // self.tp_size),
            dim=-1,
        )

        # gather on 1st dimension
        xq = xq.view(seqlen, bsz, self.n_local_heads, self.head_dim)
        xkv = xkv.view(seqlen, bsz, self.n_local_groups, 2 * self.head_dim)
        xk, xv = xkv.chunk(2, -1)

        # rotary embedding + flash attn
        # xq = rearrange(xq, "s b h d -> b s h d")
        # xk = rearrange(xk, "s b h d -> b s h d")
        # xv = rearrange(xv, "s b h d -> b s h d")
        xq = mint.swapaxes(xq, 0, 1)
        xk = mint.swapaxes(xk, 0, 1)
        xv = mint.swapaxes(xv, 0, 1)

        q_per_kv = self.n_local_heads // self.n_local_groups
        if q_per_kv > 1:
            b, s, h, d = xk.shape
            if h == 1:
                xk = mint.broadcast_to(xk, (b, s, q_per_kv, d))
                xv = mint.broadcast_to(xv, (b, s, q_per_kv, d))
            else:
                """To cover the cases where h > 1, we have
                the following implementation, which is equivalent to:
                    xk = xk.repeat_interleave(q_per_kv, dim=-2)
                    xv = xv.repeat_interleave(q_per_kv, dim=-2)
                but can avoid calling aten::item() that involves cpu.
                """
                # idx = torch.arange(q_per_kv * h).reshape(q_per_kv, -1).permute(1, 0).flatten()
                # xk = torch.index_select(xk.repeat(1, 1, q_per_kv, 1), 2, idx).contiguous()
                # xv = torch.index_select(xv.repeat(1, 1, q_per_kv, 1), 2, idx).contiguous()

                xk = xk.repeat_interleave(q_per_kv, dim=-2)
                xv = xv.repeat_interleave(q_per_kv, dim=-2)

        if self.use_flash_attention:
            output = self.core_attention(xq, xk, xv, cu_seqlens=cu_seqlens, max_seq_len=max_seq_len)
            # reduce-scatter only support first dimention now
            # output = rearrange(output, "b s h d -> s b (h d)").contiguous()
            b, s, h, d = output.shape
            output = mint.swapaxes(output, 0, 1).view(s, b, h * d)
        else:
            # unuse branch

            # xq, xk, xv = [
            #     rearrange(x, "b s ... -> s b ...").contiguous()
            #     for x in (xq, xk, xv)
            # ]
            # output = self.core_attention(xq, xk, xv, mask)
            output = None

        output = self.wo(output)
        return output


class FeedForward(nn.Cell):
    def __init__(
        self,
        cfg,
        dim: int,
        hidden_dim: int,
        layer_id: int,
        multiple_of: int = 256,
    ):
        super().__init__()

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = mint.nn.Linear(
            dim,
            2 * hidden_dim,
            bias=False,
        )
        self.w2 = mint.nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def swiglu(self, x):
        x = mint.chunk(x, 2, dim=-1)
        return mint.nn.functional.silu(x[0]) * x[1]

    def construct(self, x):
        x = self.swiglu(self.w1(x))
        output = self.w2(x)
        return output


class TransformerBlock(nn.Cell):
    def __init__(self, cfg, layer_id: int):
        super().__init__()

        self.n_heads = cfg.num_attention_heads
        self.dim = cfg.hidden_size
        self.head_dim = cfg.hidden_size // cfg.num_attention_heads
        self.attention = MultiQueryAttention(
            cfg,
            layer_id=layer_id,
        )

        self.feed_forward = FeedForward(
            cfg,
            dim=cfg.hidden_size,
            hidden_dim=cfg.ffn_hidden_size,
            layer_id=layer_id,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
        )
        self.ffn_norm = RMSNorm(
            cfg.hidden_size,
            eps=cfg.layernorm_epsilon,
        )

    def construct(
        self,
        x: Tensor,
        mask: Optional[Tensor],
        cu_seqlens: Optional[Tensor],
        max_seq_len: Optional[Tensor],
    ):
        residual = self.attention(self.attention_norm(x), mask, cu_seqlens, max_seq_len)
        h = x + residual
        ffn_res = self.feed_forward(self.ffn_norm(h))
        out = h + ffn_res

        # import pdb;pdb.set_trace()

        return out


class Transformer(nn.Cell):
    def __init__(
        self,
        config,
        max_seq_size=8192,
    ):
        super().__init__()
        self.num_layers = config.num_layers
        self.layers = self._build_layers(config)

    def _build_layers(self, config):
        layers = []
        for layer_id in range(self.num_layers):
            layers.append(
                TransformerBlock(
                    config,
                    layer_id=layer_id + 1,
                )
            )
        return nn.CellList(layers)

    def construct(
        self,
        hidden_states,
        attention_mask,
        cu_seqlens=None,
        max_seq_len=None,
    ):
        # if max_seq_len is not None and not isinstance(max_seq_len, Tensor):
        #     max_seq_len = Tensor(max_seq_len, dtype=ms.int32)

        for lid, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask,
                cu_seqlens,
                max_seq_len,
            )
        return hidden_states


class Step1Model(PreTrainedModel):
    config_class = PretrainedConfig

    # @with_empty_init
    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.tok_embeddings = LLaMaEmbedding(config)
        self.transformer = Transformer(config)

    def construct(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        hidden_states = self.tok_embeddings(input_ids)

        hidden_states = self.transformer(
            hidden_states,
            attention_mask,
        )
        return hidden_states


class STEP1TextEncoder(nn.Cell):
    def __init__(self, model_dir, max_length=320):
        super(STEP1TextEncoder, self).__init__()
        self.max_length = max_length
        self.text_tokenizer = Wrapped_StepChatTokenizer(os.path.join(model_dir, "step1_chat_tokenizer.model"))
        text_encoder = Step1Model.from_pretrained(model_dir)
        text_encoder.set_train(False)
        text_encoder.to(ms.bfloat16)

        text_encoder = ms.amp.auto_mixed_precision(text_encoder, amp_level="auto", dtype=ms.float16)
        self.text_encoder = text_encoder

    def prompts_to_tokens(self, prompts, max_length=None):
        if type(prompts) is str:
            prompts = [prompts]

        txt_tokens = self.text_tokenizer(
            prompts,
            max_length=max_length or self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        return Tensor(txt_tokens.input_ids), Tensor(txt_tokens.attention_mask)

    # @no_grad
    def construct(self, input_ids, attention_mask=None, with_mask=True):
        # with no_grad(), autocast(dtype=ms.bfloat16):

        y = self.text_encoder(input_ids, attention_mask=attention_mask if with_mask else None)
        y_mask = attention_mask

        return ops.stop_gradient(y.transpose(0, 1)), ops.stop_gradient(y_mask)

    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            p.set_dtype(dtype)
        return self
