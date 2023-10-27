# reference to https://github.com/huggingface/transformers


from typing import Optional, Tuple

import numpy as np
from gm.modules.transformers.activations import ACT2FN
from transformers.models.clip.configuration_clip import CLIPConfig, CLIPTextConfig

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

MIN_VALUE = -1e5
MAX_VALUE = 1e5


class CLIPTextEmbeddings(nn.Cell):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        embed_dim = config.hidden_size

        self.token_embedding = nn.Embedding(config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_ids = Parameter(
            Tensor(np.arange(config.max_position_embeddings).reshape((1, -1)), ms.int32), requires_grad=False
        )

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class CLIPAttention(nn.Cell):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Dense(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Dense(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).swapaxes(1, 2)

    def construct(
        self,
        hidden_states: Tensor,
        causal_attention_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.shape

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.shape[1]
        attn_weights = ops.BatchMatMul()(query_states, key_states.swapaxes(1, 2))

        if attn_weights.shape != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.shape != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.shape}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = ops.softmax(attn_weights, axis=-1)

        attn_probs = ops.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = ops.BatchMatMul()(attn_probs, value_states)

        if attn_output.shape != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.swapaxes(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, None


class CLIPMLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Dense(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Dense(config.intermediate_size, config.hidden_size)

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CLIPEncoderLayer(nn.Cell):
    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm([self.embed_dim], epsilon=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm([self.embed_dim], epsilon=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: Tensor,
        causal_attention_mask: Tensor,
    ) -> Tuple[Tensor]:
        """
        Args:
            hidden_states (`Tensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            causal_attention_mask=causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs


class CLIPEncoder(nn.Cell):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.config = config
        self.output_hidden_states = self.config.output_hidden_states
        self.layers = nn.CellList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        inputs_embeds,
        causal_attention_mask: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            causal_attention_mask (`Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
        """
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )

        encoder_states = ()
        all_attentions = ()

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            layer_outputs = encoder_layer(
                hidden_states,
                causal_attention_mask,
            )

            hidden_states = layer_outputs[0]

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        # rename
        last_hidden_state = hidden_states
        hidden_states = encoder_states
        attentions = all_attentions

        return last_hidden_state, hidden_states, attentions


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape, dtype, past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = ops.full((tgt_len, tgt_len), MIN_VALUE, dtype=ms.float32)
    mask_cond = ops.arange(mask.shape[-1])
    mask = ops.masked_fill(mask, mask_cond < (mask_cond + 1).view(mask.shape[-1], 1), 0)
    mask = mask.astype(dtype)

    if past_key_values_length > 0:
        mask = ops.concat((ops.zeros((tgt_len, past_key_values_length), dtype=dtype), mask), axis=-1)
    return mask[None, None, :, :].broadcast_to((bsz, 1, tgt_len, tgt_len + past_key_values_length))


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask, dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].broadcast_to((bsz, 1, tgt_len, src_len)).astype(dtype)

    inverted_mask = 1.0 - expanded_mask

    return ops.masked_fill(inverted_mask, inverted_mask.astype(ms.bool_), MIN_VALUE)


class CLIPTextTransformer(nn.Cell):
    def __init__(self, config: CLIPTextConfig):
        super().__init__()
        self.config = config
        self.output_hidden_states = self.config.output_hidden_states
        embed_dim = config.hidden_size
        self.embeddings = CLIPTextEmbeddings(config)
        self.encoder = CLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm([embed_dim], epsilon=config.layer_norm_eps)

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.shape
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _make_causal_mask(input_shape, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            causal_attention_mask=causal_attention_mask,
            output_hidden_states=output_hidden_states,
        )  # return (last_hidden_state, hidden_states, attentions)

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            ops.arange(last_hidden_state.shape[0]),
            input_ids.argmax(axis=-1),
        ]

        return (last_hidden_state, pooled_output) + encoder_outputs[1:]  # last, pool, hid, att


class CLIPTextModel(nn.Cell):
    config_class = CLIPTextConfig

    def __init__(self, config_path, weight=None):
        config = self.from_pretrained(config_path)  # CLIPTextConfig
        super().__init__(config)
        self.text_model = CLIPTextTransformer(config)
        self.load_checkpoint(weight)

    def get_input_embeddings(self) -> nn.Cell:
        return self.text_model.embeddings.token_embedding

    def set_input_embeddings(self, value):
        self.text_model.embeddings.token_embedding = value

    def load_checkpoint(self, checkpoint_path=None):
        if checkpoint_path is not None:
            if checkpoint_path.endswith(".ckpt"):
                param_dict = ms.load_checkpoint(checkpoint_path)
                ms.load_param_into_net(self, param_dict)
                print(f'Pretrain model load from "{checkpoint_path}" success.')
            else:
                raise ValueError(f"checkpoint path expect '*.ckpt', but got '{checkpoint_path}'.")

    def construct(
        self,
        input_ids: Optional[Tensor] = None,
        output_hidden_states: Optional[bool] = None,
    ):
        r"""
        Returns:

        Examples:

        ```python
        >>> from transformers import CLIPTokenizer

        >>> model = CLIPTextModel(config_path='openai/clip-vit-large-patch14', weight=None)  #.from_pretrained("openai/clip-vit-base-patch32")
        >>> tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True)
        >>> input_ids, output_hidden_states = Tensor(inputs["input_ids"]), True

        >>> outputs = model(input_ids=input_ids, output_hidden_states=output_hidden_states)
        >>> last_hidden_state = outputs[0]
        >>> pooled_output = outputs[1]  # pooled (EOS token) states
        >>> print(f"Input input_ids.shape: {input_ids.shape}")
        >>> print(f"Outputs last_hidden_state shape: {last_hidden_state.shape}")
        >>> print(f"Outputs pooled_output shape: {pooled_output.shape}")
        ```
        """

        return self.text_model(
            input_ids=input_ids,
            output_hidden_states=output_hidden_states,
        )

    def from_pretrained(self, config_path):
        config_path = config_path
        config, _ = CLIPTextConfig.from_pretrained(
            config_path,
            cache_dir=None,
            return_unused_kwargs=True,
            force_download=False,
            resume_download=False,
            proxies=None,
            local_files_only=False,
            token=None,
            revision="main",
            subfolder="",
            _from_auto=False,
            _from_pipeline=None,
        )
        return config
