from dataclasses import dataclass
from typing import Dict, Optional, Union

import mindspore as ms
from mindspore import mint, nn, ops

from ...configuration_utils import ConfigMixin, register_to_config
from ...loaders import PeftAdapterMixin, UNet2DConditionLoadersMixin
from ...utils import BaseOutput
from ..attention import BasicTransformerBlock
from ..attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from ..embeddings import TimestepEmbedding, Timesteps
from ..modeling_utils import ModelMixin
from ..normalization import LayerNorm


@dataclass
class PriorTransformerOutput(BaseOutput):
    """
    The output of [`PriorTransformer`].

    Args:
        predicted_image_embedding (`ms.Tensor` of shape `(batch_size, embedding_dim)`):
            The predicted CLIP image embedding conditioned on the CLIP text embedding input.
    """

    predicted_image_embedding: ms.Tensor


class PriorTransformer(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    """
    A Prior Transformer model.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 32): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_layers (`int`, *optional*, defaults to 20): The number of layers of Transformer blocks to use.
        embedding_dim (`int`, *optional*, defaults to 768): The dimension of the model input `hidden_states`
        num_embeddings (`int`, *optional*, defaults to 77):
            The number of embeddings of the model input `hidden_states`
        additional_embeddings (`int`, *optional*, defaults to 4): The number of additional tokens appended to the
            projected `hidden_states`. The actual length of the used `hidden_states` is `num_embeddings +
            additional_embeddings`.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        time_embed_act_fn (`str`, *optional*, defaults to 'silu'):
            The activation function to use to create timestep embeddings.
        norm_in_type (`str`, *optional*, defaults to None): The normalization layer to apply on hidden states before
            passing to Transformer blocks. Set it to `None` if normalization is not needed.
        embedding_proj_norm_type (`str`, *optional*, defaults to None):
            The normalization layer to apply on the input `proj_embedding`. Set it to `None` if normalization is not
            needed.
        encoder_hid_proj_type (`str`, *optional*, defaults to `linear`):
            The projection layer to apply on the input `encoder_hidden_states`. Set it to `None` if
            `encoder_hidden_states` is `None`.
        added_emb_type (`str`, *optional*, defaults to `prd`): Additional embeddings to condition the model.
            Choose from `prd` or `None`. if choose `prd`, it will prepend a token indicating the (quantized) dot
            product between the text embedding and image embedding as proposed in the unclip paper
            https://arxiv.org/abs/2204.06125 If it is `None`, no additional embeddings will be prepended.
        time_embed_dim (`int, *optional*, defaults to None): The dimension of timestep embeddings.
            If None, will be set to `num_attention_heads * attention_head_dim`
        embedding_proj_dim (`int`, *optional*, default to None):
            The dimension of `proj_embedding`. If None, will be set to `embedding_dim`.
        clip_embed_dim (`int`, *optional*, default to None):
            The dimension of the output. If None, will be set to `embedding_dim`.
    """

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_layers: int = 20,
        embedding_dim: int = 768,
        num_embeddings=77,
        additional_embeddings=4,
        dropout: float = 0.0,
        time_embed_act_fn: str = "silu",
        norm_in_type: Optional[str] = None,  # layer
        embedding_proj_norm_type: Optional[str] = None,  # layer
        encoder_hid_proj_type: Optional[str] = "linear",  # linear
        added_emb_type: Optional[str] = "prd",  # prd
        time_embed_dim: Optional[int] = None,
        embedding_proj_dim: Optional[int] = None,
        clip_embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.additional_embeddings = additional_embeddings

        time_embed_dim = time_embed_dim or inner_dim
        embedding_proj_dim = embedding_proj_dim or embedding_dim
        clip_embed_dim = clip_embed_dim or embedding_dim

        self.time_proj = Timesteps(inner_dim, True, 0)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, out_dim=inner_dim, act_fn=time_embed_act_fn)

        self.proj_in = mint.nn.Linear(embedding_dim, inner_dim)

        if embedding_proj_norm_type is None:
            self.embedding_proj_norm = None
        elif embedding_proj_norm_type == "layer":
            self.embedding_proj_norm = LayerNorm(embedding_proj_dim)
        else:
            raise ValueError(f"unsupported embedding_proj_norm_type: {embedding_proj_norm_type}")

        self.embedding_proj = mint.nn.Linear(embedding_proj_dim, inner_dim)

        if encoder_hid_proj_type is None:
            self.encoder_hidden_states_proj = None
        elif encoder_hid_proj_type == "linear":
            self.encoder_hidden_states_proj = mint.nn.Linear(embedding_dim, inner_dim)
        else:
            raise ValueError(f"unsupported encoder_hid_proj_type: {encoder_hid_proj_type}")

        self.positional_embedding = ms.Parameter(
            mint.zeros((1, num_embeddings + additional_embeddings, inner_dim)), name="positional_embedding"
        )

        if added_emb_type == "prd":
            self.prd_embedding = ms.Parameter(mint.zeros((1, 1, inner_dim)), name="prd_embedding")
        elif added_emb_type is None:
            self.prd_embedding = None
        else:
            raise ValueError(
                f"`added_emb_type`: {added_emb_type} is not supported. Make sure to choose one of `'prd'` or `None`."
            )

        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    activation_fn="gelu",
                    attention_bias=True,
                )
                for d in range(num_layers)
            ]
        )

        if norm_in_type == "layer":
            self.norm_in = LayerNorm(inner_dim)
        elif norm_in_type is None:
            self.norm_in = None
        else:
            raise ValueError(f"Unsupported norm_in_type: {norm_in_type}.")

        self.norm_out = LayerNorm(inner_dim)

        self.proj_to_clip_embeddings = mint.nn.Linear(inner_dim, clip_embed_dim)

        causal_attention_mask = mint.full(
            [num_embeddings + additional_embeddings, num_embeddings + additional_embeddings], -10000.0
        )
        causal_attention_mask = causal_attention_mask.triu(1)
        causal_attention_mask = causal_attention_mask[None, ...]
        self.causal_attention_mask = causal_attention_mask

        self.clip_mean = ms.Parameter(mint.zeros((1, clip_embed_dim)), name="clip_mean")
        self.clip_std = ms.Parameter(mint.zeros((1, clip_embed_dim)), name="clip_std")

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:  # type: ignore
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: nn.Cell, processors: Dict[str, AttentionProcessor]):  # type: ignore
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.name_cells().items():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.name_cells().items():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):  # type: ignore
        r"""
        Sets the attention processor to use to compute attention.
        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.
                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.
        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: nn.Cell, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.name_cells().items():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.name_cells().items():
            fn_recursive_attn_processor(name, module, processor)

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_default_attn_processor
    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        if all(proc.__class__ in CROSS_ATTENTION_PROCESSORS for proc in self.attn_processors.values()):
            processor = AttnProcessor()
        else:
            raise ValueError(
                f"Cannot call `set_default_attn_processor` when attention processors are of type {next(iter(self.attn_processors.values()))}"
            )

        self.set_attn_processor(processor)

    def construct(
        self,
        hidden_states,
        timestep: Union[ms.Tensor, float, int],
        proj_embedding: ms.Tensor,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        return_dict: bool = False,
    ):
        """
        The [`PriorTransformer`] forward method.

        Args:
            hidden_states (`ms.Tensor` of shape `(batch_size, embedding_dim)`):
                The currently predicted image embeddings.
            timestep (`ms.Tensor`):
                Current denoising step.
            proj_embedding (`ms.Tensor` of shape `(batch_size, embedding_dim)`):
                Projected embedding vector the denoising process is conditioned on.
            encoder_hidden_states (`ms.Tensor` of shape `(batch_size, num_embeddings, embedding_dim)`):
                Hidden states of the text embeddings the denoising process is conditioned on.
            attention_mask (`ms.Tensor` of shape `(batch_size, num_embeddings)`):
                Text mask for the text embeddings.
            return_dict (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~models.transformers.prior_transformer.PriorTransformerOutput`] instead of
                a plain tuple.

        Returns:
            [`~models.transformers.prior_transformer.PriorTransformerOutput`] or `tuple`:
                If return_dict is True, a [`~models.transformers.prior_transformer.PriorTransformerOutput`] is
                returned, otherwise a tuple is returned where the first element is the sample tensor.
        """
        batch_size = hidden_states.shape[0]

        timesteps = timestep
        # todo: unavailable mint interface
        if not ops.is_tensor(timesteps):
            timesteps = ms.Tensor([timesteps], dtype=ms.int64)
        # todo: unavailable mint interface
        elif ops.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None]

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * mint.ones(batch_size, dtype=timesteps.dtype)

        timesteps_projected = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might be fp16, so we need to cast here.
        timesteps_projected = timesteps_projected.to(dtype=self.dtype)
        time_embeddings = self.time_embedding(timesteps_projected)

        if self.embedding_proj_norm is not None:
            proj_embedding = self.embedding_proj_norm(proj_embedding)

        proj_embeddings = self.embedding_proj(proj_embedding)
        if self.encoder_hidden_states_proj is not None and encoder_hidden_states is not None:
            encoder_hidden_states = self.encoder_hidden_states_proj(encoder_hidden_states)
        elif self.encoder_hidden_states_proj is not None and encoder_hidden_states is None:
            raise ValueError("`encoder_hidden_states_proj` requires `encoder_hidden_states` to be set")

        hidden_states = self.proj_in(hidden_states)

        positional_embeddings = self.positional_embedding.to(hidden_states.dtype)

        additional_embeds = []
        additional_embeddings_len = 0

        if encoder_hidden_states is not None:
            additional_embeds.append(encoder_hidden_states)
            additional_embeddings_len += encoder_hidden_states.shape[1]

        if len(proj_embeddings.shape) == 2:
            proj_embeddings = proj_embeddings[:, None, :]

        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states[:, None, :]

        additional_embeds = additional_embeds + [
            proj_embeddings,
            time_embeddings[:, None, :],
            hidden_states,
        ]

        if self.prd_embedding is not None:
            prd_embedding = self.prd_embedding.to(hidden_states.dtype).broadcast_to((batch_size, -1, -1))
            additional_embeds.append(prd_embedding)

        hidden_states = mint.cat(
            additional_embeds,
            dim=1,
        )

        # Allow positional_embedding to not include the `addtional_embeddings` and instead pad it with zeros for these additional tokens
        additional_embeddings_len = additional_embeddings_len + proj_embeddings.shape[1] + 1
        if positional_embeddings.shape[1] < hidden_states.shape[1]:
            positional_embeddings = mint.nn.functional.pad(
                positional_embeddings,
                (
                    0,
                    0,
                    additional_embeddings_len,
                    self.prd_embedding.shape[1] if self.prd_embedding is not None else 0,
                ),
                value=0.0,
            )

        hidden_states = hidden_states + positional_embeddings

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = mint.nn.functional.pad(attention_mask, (0, self.additional_embeddings), value=0.0)
            attention_mask = (attention_mask[:, None, :] + self.causal_attention_mask).to(hidden_states.dtype)
            attention_mask = attention_mask.repeat_interleave(self.config["num_attention_heads"], dim=0)

        if self.norm_in is not None:
            hidden_states = self.norm_in(hidden_states)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        hidden_states = self.norm_out(hidden_states)

        if self.prd_embedding is not None:
            hidden_states = hidden_states[:, -1]
        else:
            hidden_states = hidden_states[:, additional_embeddings_len:]

        predicted_image_embedding = self.proj_to_clip_embeddings(hidden_states)

        if not return_dict:
            return (predicted_image_embedding,)

        return PriorTransformerOutput(predicted_image_embedding=predicted_image_embedding)

    def post_process_latents(self, prior_latents):
        prior_latents = (prior_latents * self.clip_std) + self.clip_mean
        return prior_latents
