"""
CLIPConfig class, which consists of CLIPTextCfg and CLIPVisionCfg
"""
from typing import Optional, Tuple, Union


class CLIPTextCfg:
    def __init__(
        self,
        context_length: Optional[int] = 77,
        vocab_size: Optional[int] = 49408,
        width: Optional[int] = 512,
        heads: Optional[int] = 8,
        layers: Optional[int] = 12,
        mlp_ratio: Optional[float] = 4.0,
        ls_init_value: Optional[float] = None,
        embed_cls: Optional[bool] = False,
        pad_id: Optional[int] = 0,
        no_causal_mask: Optional[bool] = False,  # disable causal masking
        final_ln_after_pool: Optional[bool] = False,  # apply final LayerNorm after pooling
        pool_type: Optional[str] = "argmax",
        proj_bias: Optional[bool] = False,
        output_tokens: Optional[bool] = False,
        act_kwargs: Optional[dict] = None,
        norm_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.heads = heads
        self.layers = layers
        self.mlp_ratio = mlp_ratio
        self.ls_init_value = ls_init_value
        self.embed_cls = embed_cls
        self.pad_id = pad_id
        self.no_causal_mask = no_causal_mask
        self.final_ln_after_pool = final_ln_after_pool
        self.pool_type = pool_type
        self.proj_bias = proj_bias
        self.output_tokens = output_tokens
        self.act_kwargs = act_kwargs
        self.norm_kwargs = norm_kwargs


class CLIPVisionCfg:
    def __init__(
        self,
        layers: Union[Tuple[int, int, int, int], int] = 12,
        width: Optional[int] = 768,
        head_width: Optional[int] = 64,
        mlp_ratio: Optional[float] = 4.0,
        patch_size: Optional[int] = 16,
        image_size: Union[Tuple[int, int], int] = 224,
        ls_init_value: Optional[float] = None,  # layer scale initial value
        patch_dropout: Optional[float] = 0.0,  # what fraction of patches to dropout during training
        # (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
        attentional_pool: Optional[
            bool
        ] = False,  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
        attn_pooler_queries: Optional[int] = 256,  # n_queries for attentional pooler
        attn_pooler_heads: Optional[int] = 8,  # n heads for attentional_pooling
        no_ln_pre: Optional[bool] = False,  # disable pre transformer LayerNorm
        pos_embed_type: Optional[str] = "learnable",
        final_ln_after_pool: Optional[bool] = False,  # apply final LayerNorm after pooling
        pool_type: Optional[str] = "tok",
        output_tokens: Optional[bool] = False,
        act_kwargs: Optional[dict] = None,
        norm_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        self.layers = layers
        self.width = width
        self.head_width = head_width
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.image_size = image_size
        self.ls_init_value = ls_init_value
        self.patch_dropout = patch_dropout
        self.attentional_pool = attentional_pool
        self.attn_pooler_queries = attn_pooler_queries
        self.attn_pooler_heads = attn_pooler_heads
        self.no_ln_pre = no_ln_pre
        self.pos_embed_type = pos_embed_type
        self.final_ln_after_pool = final_ln_after_pool
        self.pool_type = pool_type
        self.output_tokens = output_tokens
        self.act_kwargs = act_kwargs
        self.norm_kwargs = norm_kwargs
