from typing import Any, Dict, List, Optional, Tuple, Union

import mindspore as ms
from mindspore import mint, nn, ops

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import FromOriginalModelMixin, PeftAdapterMixin
from ...models.modeling_outputs import Transformer2DModelOutput
from ...models.modeling_utils import ModelMixin
from ...utils import deprecate, logging
from ..attention import Attention
from ..embeddings import TimestepEmbedding, Timesteps
from ..normalization import RMSNorm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class HiDreamImageFeedForwardSwiGLU(nn.Cell):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = mint.nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = mint.nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = mint.nn.Linear(dim, hidden_dim, bias=False)

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        return self.w2(mint.functional.silu(self.w1(x)) * self.w3(x))


class HiDreamImagePooledEmbed(nn.Cell):
    def __init__(self, text_emb_dim, hidden_size):
        super().__init__()
        self.pooled_embedder = TimestepEmbedding(in_channels=text_emb_dim, time_embed_dim=hidden_size)

    def construct(self, pooled_embed: ms.Tensor) -> ms.Tensor:
        return self.pooled_embedder(pooled_embed)


class HiDreamImageTimestepEmbed(nn.Cell):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.time_proj = Timesteps(num_channels=frequency_embedding_size, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=frequency_embedding_size, time_embed_dim=hidden_size)

    def construct(self, timesteps: ms.Tensor, wdtype: Optional[ms.Type] = None):
        t_emb = self.time_proj(timesteps).to(dtype=wdtype)
        t_emb = self.timestep_embedder(t_emb)
        return t_emb


class HiDreamImageOutEmbed(nn.Cell):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = mint.nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = mint.nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.SequentialCell(
            mint.nn.SiLU(), mint.nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def construct(self, hidden_states: ms.Tensor, temb: ms.Tensor) -> ms.Tensor:
        shift, scale = self.adaLN_modulation(temb).chunk(2, dim=1)
        hidden_states = self.norm_final(hidden_states) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        hidden_states = self.linear(hidden_states)
        return hidden_states


class HiDreamImagePatchEmbed(nn.Cell):
    def __init__(
        self,
        patch_size=2,
        in_channels=4,
        out_channels=1024,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.proj = mint.nn.Linear(in_channels * patch_size * patch_size, out_channels, bias=True)

    def construct(self, latent):
        latent = self.proj(latent)
        return latent


def rope(pos: ms.Tensor, dim: int, theta: int) -> ms.Tensor:
    assert dim % 2 == 0, "The dimension must be even."

    # TODO check here, notes: hf use ms.float32 if npu else
    # dtype = ms.float32 if (is_mps or is_npu) else ms.float64
    dtype = ms.float32

    scale = mint.arange(0, dim, 2, dtype=dtype) / dim
    omega = 1.0 / (theta**scale)

    batch_size, seq_length = pos.shape
    out = mint.einsum("...n,d->...nd", pos, omega)
    cos_out = mint.cos(out)
    sin_out = mint.sin(out)

    stacked_out = mint.stack([cos_out, -sin_out, sin_out, cos_out], dim=-1)
    out = stacked_out.view(batch_size, -1, dim // 2, 2, 2)
    return out.float()


class HiDreamImageEmbedND(nn.Cell):
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def construct(self, ids: ms.Tensor) -> ms.Tensor:
        n_axes = ids.shape[-1]
        emb = mint.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(2)


def apply_rope(xq: ms.Tensor, xk: ms.Tensor, freqs_cis: ms.Tensor) -> Tuple[ms.Tensor, ms.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class HiDreamAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        heads: int = 8,
        dim_head: int = 64,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        scale_qk: bool = True,
        eps: float = 1e-5,
        processor=None,
        out_dim: int = None,
        single: bool = False,
    ):
        super(Attention, self).__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.out_dim = out_dim if out_dim is not None else query_dim

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        self.sliceable_head_dim = heads
        self.single = single

        self.to_q = mint.nn.Linear(query_dim, self.inner_dim)
        self.to_k = mint.nn.Linear(self.inner_dim, self.inner_dim)
        self.to_v = mint.nn.Linear(self.inner_dim, self.inner_dim)
        self.to_out = mint.nn.Linear(self.inner_dim, self.out_dim)
        self.q_rms_norm = RMSNorm(self.inner_dim, eps)
        self.k_rms_norm = RMSNorm(self.inner_dim, eps)

        if not single:
            self.to_q_t = mint.nn.Linear(query_dim, self.inner_dim)
            self.to_k_t = mint.nn.Linear(self.inner_dim, self.inner_dim)
            self.to_v_t = mint.nn.Linear(self.inner_dim, self.inner_dim)
            self.to_out_t = mint.nn.Linear(self.inner_dim, self.out_dim)
            self.q_rms_norm_t = RMSNorm(self.inner_dim, eps)
            self.k_rms_norm_t = RMSNorm(self.inner_dim, eps)

        self.set_processor(processor)

    def scaled_dot_product_attention(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ):
        if query.dtype in (ms.float16, ms.bfloat16):
            return self.flash_attention_op(query, key, value, attn_mask, keep_prob=1 - dropout_p, scale=scale)
        else:
            return self.flash_attention_op(
                query.to(ms.float16),
                key.to(ms.float16),
                value.to(ms.float16),
                attn_mask,
                keep_prob=1 - dropout_p,
                scale=scale,
            ).to(query.dtype)

    def flash_attention_op(
        self,
        query: ms.Tensor,
        key: ms.Tensor,
        value: ms.Tensor,
        attn_mask: Optional[ms.Tensor] = None,
        keep_prob: float = 1.0,
        scale: Optional[float] = None,
    ):
        # For most scenarios, qkv has been processed into a BNSD layout before sdp
        input_layout = "BNSD"
        head_num = self.heads

        # In case qkv is 3-dim after `head_to_batch_dim`
        if query.ndim == 3:
            input_layout = "BSH"
            head_num = 1

        # process `attn_mask` as logic is different between PyTorch and Mindspore
        # In MindSpore, False indicates retention and True indicates discard, in PyTorch it is the opposite
        if attn_mask is not None:
            attn_mask = mint.logical_not(attn_mask) if attn_mask.dtype == ms.bool_ else attn_mask.bool()
            attn_mask = mint.broadcast_to(
                attn_mask, (attn_mask.shape[0], attn_mask.shape[1], query.shape[-2], key.shape[-2])
            )[:, :1, :, :]

        return ops.operations.nn_ops.FlashAttentionScore(
            head_num=head_num, keep_prob=keep_prob, scale_value=scale or self.scale, input_layout=input_layout
        )(query, key, value, None, None, None, attn_mask)[3]

    def construct(
        self,
        norm_hidden_states: ms.Tensor,
        hidden_states_masks: ms.Tensor = None,
        norm_encoder_hidden_states: ms.Tensor = None,
        image_rotary_emb: ms.Tensor = None,
    ) -> ms.Tensor:
        return self.processor(
            self,
            hidden_states=norm_hidden_states,
            hidden_states_masks=hidden_states_masks,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )


class HiDreamAttnProcessor:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __call__(
        self,
        attn: HiDreamAttention,
        hidden_states: ms.Tensor,
        hidden_states_masks: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        image_rotary_emb: ms.Tensor = None,
        *args,
        **kwargs,
    ) -> ms.Tensor:
        dtype = hidden_states.dtype
        batch_size = hidden_states.shape[0]

        query_i = attn.q_rms_norm(attn.to_q(hidden_states)).to(dtype=dtype)
        key_i = attn.k_rms_norm(attn.to_k(hidden_states)).to(dtype=dtype)
        value_i = attn.to_v(hidden_states)

        inner_dim = key_i.shape[-1]
        head_dim = inner_dim // attn.heads

        query_i = query_i.view(batch_size, -1, attn.heads, head_dim)
        key_i = key_i.view(batch_size, -1, attn.heads, head_dim)
        value_i = value_i.view(batch_size, -1, attn.heads, head_dim)
        if hidden_states_masks is not None:
            key_i = key_i * hidden_states_masks.view(batch_size, -1, 1, 1)

        if not attn.single:
            query_t = attn.q_rms_norm_t(attn.to_q_t(encoder_hidden_states)).to(dtype=dtype)
            key_t = attn.k_rms_norm_t(attn.to_k_t(encoder_hidden_states)).to(dtype=dtype)
            value_t = attn.to_v_t(encoder_hidden_states)

            query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
            key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
            value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

            num_image_tokens = query_i.shape[1]
            num_text_tokens = query_t.shape[1]
            query = mint.cat([query_i, query_t], dim=1)
            key = mint.cat([key_i, key_t], dim=1)
            value = mint.cat([value_i, value_t], dim=1)
        else:
            query = query_i
            key = key_i
            value = value_i

        if query.shape[-1] == image_rotary_emb.shape[-3] * 2:
            query, key = apply_rope(query, key, image_rotary_emb)

        else:
            query_1, query_2 = query.chunk(2, dim=-1)
            key_1, key_2 = key.chunk(2, dim=-1)
            query_1, key_1 = apply_rope(query_1, key_1, image_rotary_emb)
            query = mint.cat([query_1, query_2], dim=-1)
            key = mint.cat([key_1, key_2], dim=-1)

        hidden_states = attn.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if not attn.single:
            hidden_states_i, hidden_states_t = mint.split(hidden_states, [num_image_tokens, num_text_tokens], dim=1)
            hidden_states_i = attn.to_out(hidden_states_i)
            hidden_states_t = attn.to_out_t(hidden_states_t)
            return hidden_states_i, hidden_states_t
        else:
            hidden_states = attn.to_out(hidden_states)
            return hidden_states


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MoEGate(nn.Cell):
    def __init__(
        self,
        embed_dim,
        num_routed_experts=4,
        num_activated_experts=2,
        aux_loss_alpha=0.01,
        _force_inference_output=False,
    ):
        super().__init__()
        self.top_k = num_activated_experts
        self.n_routed_experts = num_routed_experts

        self.scoring_func = "softmax"
        self.alpha = aux_loss_alpha
        self.seq_aux = False

        # topk selection algorithm
        self.norm_topk_prob = False
        self.gating_dim = embed_dim
        self.weight = ms.Parameter(mint.randn(self.n_routed_experts, self.gating_dim) / embed_dim**0.5)

        self._force_inference_output = _force_inference_output

    def construct(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        # compute gating score
        hidden_states = hidden_states.view(-1, h)
        logits = mint.functional.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(axis=-1)
        else:
            raise NotImplementedError(f"insupportable scoring function for MoE gating: {self.scoring_func}")

        # select top-k experts
        topk_weight, topk_idx = mint.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # norm gate to sum 1
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        # expert-level computation auxiliary loss
        if self.training and self.alpha > 0.0 and not self._force_inference_output:
            scores_for_aux = scores
            aux_topk = self.top_k
            # always compute aux loss based on the naive greedy topk method
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = mint.zeros((bsz, self.n_routed_experts))
                ce.scatter_add_(1, topk_idx_for_aux_loss, mint.ones(bsz, seq_len * aux_topk)).div_(
                    seq_len * aux_topk / self.n_routed_experts
                )
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = mint.functional.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)

                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None
        return topk_idx, topk_weight, aux_loss


# Modified from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
class MOEFeedForwardSwiGLU(nn.Cell):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_routed_experts: int,
        num_activated_experts: int,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.shared_experts = HiDreamImageFeedForwardSwiGLU(dim, hidden_dim // 2)
        self.experts = nn.CellList([HiDreamImageFeedForwardSwiGLU(dim, hidden_dim) for i in range(num_routed_experts)])
        self._force_inference_output = _force_inference_output
        self.gate = MoEGate(
            embed_dim=dim,
            num_routed_experts=num_routed_experts,
            num_activated_experts=num_activated_experts,
            _force_inference_output=_force_inference_output,
        )
        self.num_activated_experts = num_activated_experts

    def construct(self, x):
        wtype = x.dtype
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training and not self._force_inference_output:
            x = x.repeat_interleave(self.num_activated_experts, dim=0)
            y = mint.empty_like(x, dtype=wtype)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(dtype=wtype)
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape).to(dtype=wtype)
            # y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = mint.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().numpy().cumsum(0)
        token_idxs = idxs // self.num_activated_experts
        for i, end_idx in enumerate(tokens_per_expert):
            end_idx = int(end_idx)
            start_idx = 0 if i == 0 else int(tokens_per_expert[i - 1])
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])

            # for fp16 and other dtype
            expert_cache = expert_cache.to(expert_out.dtype)
            # FIXME: mindspore lacks tensor.scatter_reduce_, use an scatter_add_ instead, which is safe here.
            # expert_cache.scatter_reduce_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out, reduce="sum")
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)
        return expert_cache


class TextProjection(nn.Cell):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear = mint.nn.Linear(in_features=in_features, out_features=hidden_size, bias=False)

    def construct(self, caption):
        hidden_states = self.linear(caption)
        return hidden_states


class HiDreamImageSingleTransformerBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.SequentialCell(mint.nn.SiLU(), mint.nn.Linear(dim, 6 * dim, bias=True))

        # 1. Attention
        self.norm1_i = mint.nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor=HiDreamAttnProcessor(),
            single=True,
        )

        # 3. Feed-forward
        self.norm3_i = mint.nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
                _force_inference_output=_force_inference_output,
            )
        else:
            self.ff_i = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)

    def construct(
        self,
        hidden_states: ms.Tensor,
        hidden_states_masks: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        image_rotary_emb: ms.Tensor = None,
    ) -> ms.Tensor:
        wtype = hidden_states.dtype
        shift_msa_i, scale_msa_i, gate_msa_i, shift_mlp_i, scale_mlp_i, gate_mlp_i = self.adaLN_modulation(temb)[
            :, None
        ].chunk(6, dim=-1)

        # 1. MM-Attention
        norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa_i) + shift_msa_i
        attn_output_i = self.attn1(
            norm_hidden_states,
            hidden_states_masks,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = gate_msa_i * attn_output_i + hidden_states

        # 2. Feed-forward
        norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp_i) + shift_mlp_i
        ff_output_i = gate_mlp_i * self.ff_i(norm_hidden_states.to(dtype=wtype))
        hidden_states = ff_output_i + hidden_states
        return hidden_states


class HiDreamImageTransformerBlock(nn.Cell):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        _force_inference_output: bool = False,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.adaLN_modulation = nn.SequentialCell(mint.nn.SiLU(), mint.nn.Linear(dim, 12 * dim, bias=True))

        # 1. Attention
        self.norm1_i = mint.nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.norm1_t = mint.nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.attn1 = HiDreamAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            processor=HiDreamAttnProcessor(),
            single=False,
        )

        # 3. Feed-forward
        self.norm3_i = mint.nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        if num_routed_experts > 0:
            self.ff_i = MOEFeedForwardSwiGLU(
                dim=dim,
                hidden_dim=4 * dim,
                num_routed_experts=num_routed_experts,
                num_activated_experts=num_activated_experts,
                _force_inference_output=_force_inference_output,
            )
        else:
            self.ff_i = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)
        self.norm3_t = mint.nn.LayerNorm(dim, eps=1e-06, elementwise_affine=False)
        self.ff_t = HiDreamImageFeedForwardSwiGLU(dim=dim, hidden_dim=4 * dim)

    def construct(
        self,
        hidden_states: ms.Tensor,
        hidden_states_masks: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        image_rotary_emb: ms.Tensor = None,
    ) -> ms.Tensor:
        wtype = hidden_states.dtype
        (
            shift_msa_i,
            scale_msa_i,
            gate_msa_i,
            shift_mlp_i,
            scale_mlp_i,
            gate_mlp_i,
            shift_msa_t,
            scale_msa_t,
            gate_msa_t,
            shift_mlp_t,
            scale_mlp_t,
            gate_mlp_t,
        ) = self.adaLN_modulation(temb)[:, None].chunk(12, dim=-1)

        # 1. MM-Attention
        norm_hidden_states = self.norm1_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_msa_i) + shift_msa_i
        norm_encoder_hidden_states = self.norm1_t(encoder_hidden_states).to(dtype=wtype)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_msa_t) + shift_msa_t

        attn_output_i, attn_output_t = self.attn1(
            norm_hidden_states,
            hidden_states_masks,
            norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        hidden_states = gate_msa_i * attn_output_i + hidden_states
        encoder_hidden_states = gate_msa_t * attn_output_t + encoder_hidden_states

        # 2. Feed-forward
        norm_hidden_states = self.norm3_i(hidden_states).to(dtype=wtype)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp_i) + shift_mlp_i
        norm_encoder_hidden_states = self.norm3_t(encoder_hidden_states).to(dtype=wtype)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_mlp_t) + shift_mlp_t

        ff_output_i = gate_mlp_i * self.ff_i(norm_hidden_states)
        ff_output_t = gate_mlp_t * self.ff_t(norm_encoder_hidden_states)
        hidden_states = ff_output_i + hidden_states
        encoder_hidden_states = ff_output_t + encoder_hidden_states
        return hidden_states, encoder_hidden_states


class HiDreamBlock(nn.Cell):
    def __init__(self, block: Union[HiDreamImageTransformerBlock, HiDreamImageSingleTransformerBlock]):
        super().__init__()
        self.block = block

    def construct(
        self,
        hidden_states: ms.Tensor,
        hidden_states_masks: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        temb: Optional[ms.Tensor] = None,
        image_rotary_emb: ms.Tensor = None,
    ) -> ms.Tensor:
        return self.block(
            hidden_states=hidden_states,
            hidden_states_masks=hidden_states_masks,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
        )


class HiDreamImageTransformer2DModel(ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin):
    _supports_gradient_checkpointing = True
    _no_split_modules = ["HiDreamImageTransformerBlock", "HiDreamImageSingleTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size: Optional[int] = None,
        in_channels: int = 64,
        out_channels: Optional[int] = None,
        num_layers: int = 16,
        num_single_layers: int = 32,
        attention_head_dim: int = 128,
        num_attention_heads: int = 20,
        caption_channels: List[int] = None,
        text_emb_dim: int = 2048,
        num_routed_experts: int = 4,
        num_activated_experts: int = 2,
        axes_dims_rope: Tuple[int, int] = (32, 32),
        max_resolution: Tuple[int, int] = (128, 128),
        llama_layers: List[int] = None,
        force_inference_output: bool = False,
    ):
        super().__init__()
        self.out_channels = out_channels or in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.t_embedder = HiDreamImageTimestepEmbed(self.inner_dim)
        self.p_embedder = HiDreamImagePooledEmbed(text_emb_dim, self.inner_dim)
        self.x_embedder = HiDreamImagePatchEmbed(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=self.inner_dim,
        )
        self.pe_embedder = HiDreamImageEmbedND(theta=10000, axes_dim=axes_dims_rope)

        self.double_stream_blocks = nn.CellList(
            [
                HiDreamBlock(
                    HiDreamImageTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        num_routed_experts=num_routed_experts,
                        num_activated_experts=num_activated_experts,
                        _force_inference_output=force_inference_output,
                    )
                )
                for _ in range(num_layers)
            ]
        )

        self.single_stream_blocks = nn.CellList(
            [
                HiDreamBlock(
                    HiDreamImageSingleTransformerBlock(
                        dim=self.inner_dim,
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=attention_head_dim,
                        num_routed_experts=num_routed_experts,
                        num_activated_experts=num_activated_experts,
                        _force_inference_output=force_inference_output,
                    )
                )
                for _ in range(num_single_layers)
            ]
        )

        self.final_layer = HiDreamImageOutEmbed(self.inner_dim, patch_size, self.out_channels)

        caption_channels = [caption_channels[1]] * (num_layers + num_single_layers) + [caption_channels[0]]
        caption_projection = []
        for caption_channel in caption_channels:
            caption_projection.append(TextProjection(in_features=caption_channel, hidden_size=self.inner_dim))
        self.caption_projection = nn.CellList(caption_projection)
        self.max_seq = max_resolution[0] * max_resolution[1] // (patch_size * patch_size)

        self.gradient_checkpointing = False

        self.patch_size = self.config.patch_size
        self.force_inference_output = self.config.force_inference_output
        self.llama_layers = self.config.llama_layers

    def unpatchify(self, x: ms.Tensor, img_sizes: List[Tuple[int, int]], is_training: bool) -> List[ms.Tensor]:
        if is_training and not self.force_inference_output:
            B, S, F = x.shape
            C = F // (self.patch_size * self.patch_size)
            x = (
                x.reshape((B, S, self.patch_size, self.patch_size, C))
                .permute(0, 4, 1, 2, 3)
                .reshape((B, C, S, self.patch_size * self.patch_size))
            )
        else:
            x_arr = []
            p1 = self.patch_size
            p2 = self.patch_size
            for i, img_size in enumerate(img_sizes):
                pH, pW = img_size
                t = x[i, : pH * pW].reshape((1, pH, pW, -1))
                F_token = t.shape[-1]
                C = F_token // (p1 * p2)
                t = t.reshape((1, pH, pW, p1, p2, C))
                t = t.permute(0, 5, 1, 3, 2, 4)
                t = t.reshape((1, C, pH * p1, pW * p2))
                x_arr.append(t)
            x = mint.cat(x_arr, dim=0)
        return x

    def patchify(self, hidden_states):
        batch_size, channels, height, width = hidden_states.shape
        patch_size = self.patch_size
        patch_height, patch_width = height // patch_size, width // patch_size
        dtype = hidden_states.dtype

        # create img_sizes
        img_sizes = ms.tensor([patch_height, patch_width], dtype=ms.int64).reshape(-1)
        img_sizes = img_sizes.unsqueeze(0).repeat(batch_size, 1)

        # create hidden_states_masks
        if hidden_states.shape[-2] != hidden_states.shape[-1]:
            hidden_states_masks = mint.zeros((batch_size, self.max_seq), dtype=dtype)
            hidden_states_masks[:, : patch_height * patch_width] = 1.0
        else:
            hidden_states_masks = None

        # create img_ids
        img_ids = mint.zeros((patch_height, patch_width, 3))
        row_indices = mint.arange(patch_height)[:, None]
        col_indices = mint.arange(patch_width)[None, :]
        img_ids[..., 1] = img_ids[..., 1] + row_indices
        img_ids[..., 2] = img_ids[..., 2] + col_indices
        img_ids = img_ids.reshape(patch_height * patch_width, -1)

        if hidden_states.shape[-2] != hidden_states.shape[-1]:
            # Handle non-square latents
            img_ids_pad = mint.zeros((self.max_seq, 3))
            img_ids_pad[: patch_height * patch_width, :] = img_ids
            img_ids = img_ids_pad.unsqueeze(0).repeat(batch_size, 1, 1)
        else:
            img_ids = img_ids.unsqueeze(0).repeat(batch_size, 1, 1)

        # patchify hidden_states
        if hidden_states.shape[-2] != hidden_states.shape[-1]:
            # Handle non-square latents
            out = mint.zeros(
                (batch_size, channels, self.max_seq, patch_size * patch_size),
                dtype=dtype,
            )
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height, patch_size, patch_width, patch_size
            )
            hidden_states = hidden_states.permute(0, 1, 2, 4, 3, 5)
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height * patch_width, patch_size * patch_size
            )
            out[:, :, 0 : patch_height * patch_width] = hidden_states
            hidden_states = out
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch_size, self.max_seq, patch_size * patch_size * channels
            )

        else:
            # Handle square latents
            hidden_states = hidden_states.reshape(
                batch_size, channels, patch_height, patch_size, patch_width, patch_size
            )
            hidden_states = hidden_states.permute(0, 2, 4, 3, 5, 1)
            hidden_states = hidden_states.reshape(
                batch_size, patch_height * patch_width, patch_size * patch_size * channels
            )

        return hidden_states, hidden_states_masks, img_sizes, img_ids

    def construct(
        self,
        hidden_states: ms.Tensor,
        timesteps: ms.Tensor = None,
        encoder_hidden_states_t5: ms.Tensor = None,
        encoder_hidden_states_llama3: ms.Tensor = None,
        pooled_embeds: ms.Tensor = None,
        img_ids: Optional[ms.Tensor] = None,
        img_sizes: Optional[List[Tuple[int, int]]] = None,
        hidden_states_masks: Optional[ms.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
        **kwargs,
    ):
        encoder_hidden_states = kwargs.get("encoder_hidden_states", None)

        if encoder_hidden_states is not None:
            deprecation_message = "The `encoder_hidden_states` argument is deprecated. \
                Please use `encoder_hidden_states_t5` and `encoder_hidden_states_llama3` instead."
            deprecate("encoder_hidden_states", "0.35.0", deprecation_message)
            encoder_hidden_states_t5 = encoder_hidden_states[0]
            encoder_hidden_states_llama3 = encoder_hidden_states[1]

        if img_ids is not None and img_sizes is not None and hidden_states_masks is None:
            deprecation_message = (
                "Passing `img_ids` and `img_sizes` with unpachified `hidden_states` is deprecated and will be ignored."
            )
            deprecate("img_ids", "0.35.0", deprecation_message)

        if hidden_states_masks is not None and (img_ids is None or img_sizes is None):
            raise ValueError("if `hidden_states_masks` is passed, `img_ids` and `img_sizes` must also be passed.")
        elif hidden_states_masks is not None and hidden_states.ndim != 3:
            raise ValueError(
                "if `hidden_states_masks` is passed, `hidden_states` must be a 3D tensors with shape \
                    (batch_size, patch_height * patch_width, patch_size * patch_size * channels)"
            )

        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        # spatial forward
        batch_size = hidden_states.shape[0]
        hidden_states_type = hidden_states.dtype

        # Patchify the input
        if hidden_states_masks is None:
            hidden_states, hidden_states_masks, img_sizes, img_ids = self.patchify(hidden_states)

        # Embed the hidden states
        hidden_states = self.x_embedder(hidden_states)

        # 0. time
        timesteps = self.t_embedder(timesteps, hidden_states_type)
        p_embedder = self.p_embedder(pooled_embeds)
        temb = timesteps + p_embedder

        encoder_hidden_states = [encoder_hidden_states_llama3[k] for k in self.llama_layers]

        if self.caption_projection is not None:
            new_encoder_hidden_states = []
            for i, enc_hidden_state in enumerate(encoder_hidden_states):
                enc_hidden_state = self.caption_projection[i](enc_hidden_state)
                enc_hidden_state = enc_hidden_state.view(batch_size, -1, hidden_states.shape[-1])
                new_encoder_hidden_states.append(enc_hidden_state)
            encoder_hidden_states = new_encoder_hidden_states
            encoder_hidden_states_t5 = self.caption_projection[-1](encoder_hidden_states_t5)
            encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states.append(encoder_hidden_states_t5)

        txt_ids = mint.zeros(
            (
                batch_size,
                encoder_hidden_states[-1].shape[1]
                + encoder_hidden_states[-2].shape[1]
                + encoder_hidden_states[0].shape[1],
                3,
            ),
            dtype=img_ids.dtype,
        )
        ids = mint.cat((img_ids, txt_ids), dim=1)
        image_rotary_emb = self.pe_embedder(ids)

        # 2. Blocks
        block_id = 0
        initial_encoder_hidden_states = mint.cat([encoder_hidden_states[-1], encoder_hidden_states[-2]], dim=1)
        initial_encoder_hidden_states_seq_len = initial_encoder_hidden_states.shape[1]
        for bid, block in enumerate(self.double_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            cur_encoder_hidden_states = mint.cat(
                [initial_encoder_hidden_states, cur_llama31_encoder_hidden_states], dim=1
            )

            hidden_states, initial_encoder_hidden_states = block(
                hidden_states=hidden_states,
                hidden_states_masks=hidden_states_masks,
                encoder_hidden_states=cur_encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            initial_encoder_hidden_states = initial_encoder_hidden_states[:, :initial_encoder_hidden_states_seq_len]
            block_id += 1

        image_tokens_seq_len = hidden_states.shape[1]
        hidden_states = mint.cat([hidden_states, initial_encoder_hidden_states], dim=1)
        hidden_states_seq_len = hidden_states.shape[1]
        if hidden_states_masks is not None:
            encoder_attention_mask_ones = mint.ones(
                (batch_size, initial_encoder_hidden_states.shape[1] + cur_llama31_encoder_hidden_states.shape[1]),
                dtype=hidden_states_masks.dtype,
            )
            hidden_states_masks = mint.cat([hidden_states_masks, encoder_attention_mask_ones], dim=1)

        for bid, block in enumerate(self.single_stream_blocks):
            cur_llama31_encoder_hidden_states = encoder_hidden_states[block_id]
            hidden_states = mint.cat([hidden_states, cur_llama31_encoder_hidden_states], dim=1)

            hidden_states = block(
                hidden_states=hidden_states,
                hidden_states_masks=hidden_states_masks,
                encoder_hidden_states=None,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
            )
            hidden_states = hidden_states[:, :hidden_states_seq_len]
            block_id += 1

        hidden_states = hidden_states[:, :image_tokens_seq_len, ...]
        output = self.final_layer(hidden_states, temb)
        output = self.unpatchify(output, img_sizes, self.training)
        if hidden_states_masks is not None:
            hidden_states_masks = hidden_states_masks[:, :image_tokens_seq_len]

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
