""" MindSpore Dino ViT model."""


import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

from transformers import ViTConfig

import mindspore as ms
import mindspore.nn as nn
from mindspore import Parameter, Tensor, mint, ops
from mindspore.common.initializer import TruncatedNormal, initializer
from mindspore.nn import LayerNorm

from mindone.transformers.activations import ACT2FN
from mindone.transformers.mindspore_utils import find_pruneable_heads_and_indices, prune_linear_layer
from mindone.transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from mindone.transformers.modeling_utils import MSPreTrainedModel as PreTrainedModel


class ViTEmbeddings(nn.Cell):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = Parameter(mint.normal(size=(1, 1, config.hidden_size)))
        self.mask_token = Parameter(mint.zeros((1, 1, config.hidden_size))) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = Parameter(mint.normal(size=(1, num_patches + 1, config.hidden_size)))
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
        self.config = config

    def interpolate_pos_encoding(self, embeddings: Tensor, height: int, width: int) -> Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher
        resolution images.

        Source:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        dim = embeddings.shape[-1]
        h0 = height // self.config.patch_size
        w0 = width // self.config.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        h0, w0 = h0 + 0.1, w0 + 0.1
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # FIXME this ops huge difference
        patch_pos_embed = ops.interpolate(
            patch_pos_embed,
            scale_factor=(h0 / math.sqrt(num_positions), w0 / math.sqrt(num_positions)),
            recompute_scale_factor=True,
            mode="bicubic",
            align_corners=False,
        )

        assert int(h0) == patch_pos_embed.shape[-2] and int(w0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return mint.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def construct(
        self,
        pixel_values: Tensor,
        bool_masked_pos: Optional[Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand((batch_size, seq_length, -1))
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).to(mask_tokens.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.to(embeddings.dtype).broadcast_to((batch_size, -1, -1))
        embeddings = mint.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ViTPatchEmbeddings(nn.Cell):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size, has_bias=True)

    def construct(self, pixel_values: Tensor, interpolate_pos_encoding: bool = False) -> Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(start_dim=2).swapaxes(1, 2)
        return embeddings


class ViTSelfAttention(nn.Cell):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size, } is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Dense(config.hidden_size, self.all_head_size, bias_init=config.qkv_bias)
        self.key = nn.Dense(config.hidden_size, self.all_head_size, bias_init=config.qkv_bias)
        self.value = nn.Dense(config.hidden_size, self.all_head_size, bias_init=config.qkv_bias)
        self.dropout = nn.Dropout(p=config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: Tensor) -> Tensor:
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self, hidden_states, head_mask: Optional[Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = mint.matmul(query_layer, key_layer.swapaxes(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = mint.nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = mint.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class ViTSelfOutput(nn.Cell):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Cell):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def construct(
        self,
        hidden_states: Tensor,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ViTIntermediate(nn.Cell):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Cell):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Dense(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)  # broadcasting


class ViTLayer(nn.Cell):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        # self.chunk_size_feed_construct = config.chunk_size_feed_construct
        self.seq_len_dim = 1
        self.attention = ViTAttention(config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        self.layernorm_after = LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)

        self.adaLN_modulation = nn.SequentialCell(
            nn.SiLU(), nn.Dense(config.hidden_size, 4 * config.hidden_size, bias_init=True)
        )
        # nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        # nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        self.adaLN_modulation[-1].weight = mint.zeros_like(self.adaLN_modulation[-1].weight)
        self.adaLN_modulation[-1].bias = mint.zeros_like(self.adaLN_modulation[-1].bias)

    def construct(
        self,
        hidden_states: Tensor,
        adaln_input: Tensor = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor]]:
        shift_msa, scale_msa, shift_mlp, scale_mlp = self.adaLN_modulation(adaln_input).chunk(4, axis=1)

        self_attention_outputs = self.attention(
            modulate(
                self.layernorm_before(hidden_states), shift_msa, scale_msa
            ),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = modulate(self.layernorm_after(hidden_states), shift_mlp, scale_mlp)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTEncoder(nn.Cell):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.CellList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: Tensor,
        adaln_input: Tensor = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        # return_dict: bool = True,  # not returning dict, as graph mode does not support the dict wrapper which is a non-cell class
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(hidden_states, adaln_input, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # logger.info('in vit encoder, retruning tuple')
        return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)


class ViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViTEmbeddings", "ViTLayer"]

    def _init_weights(self, module: Union[nn.Dense, nn.Conv2d, LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Dense, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.set_data(
                initializer(
                    TruncatedNormal(sigma=self.config.initializer_range, mean=0.0), module.weight.shape, ms.float32
                )
            )
            # module.weight.data = nn.init.trunc_normal_(  # torch references
            #     module.weight.data.to(ms.float32), mean=0.0, std=self.config.initializer_range
            # ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.set_data(initializer("Zero", module.bias.shape, ms.float32))
                # module.bias.data.zero_()
        elif isinstance(module, LayerNorm):
            module.beta.set_data(initializer("Zero", module.beta.shape, ms.float32))
            module.gamma.set_data(initializer("One", module.gamma.shape, ms.float32))
        elif isinstance(module, ViTEmbeddings):
            # module.position_embeddings.data = nn.init.trunc_normal_(
            #     module.position_embeddings.data.to(ms.float32),
            #     mean=0.0,
            #     std=self.config.initializer_range,
            # ).to(module.position_embeddings.dtype)
            module.position_embeddings.set_data(
                initializer(
                    TruncatedNormal(sigma=self.config.initializer_range, mean=0.0),
                    module.position_embeddings.shape,
                    ms.float32,
                )
            )

            # module.cls_token.data = nn.init.trunc_normal_(
            #     module.cls_token.data.to(ms.float32),
            #     mean=0.0,
            #     std=self.config.initializer_range,
            # ).to(module.cls_token.dtype)
            module.cls_token.set_data(
                initializer(
                    TruncatedNormal(sigma=self.config.initializer_range, mean=0.0), module.cls_token.shape, ms.float32
                )
            )


class ViTPooler(nn.Cell):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Dense(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def construct(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config
        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm([config.hidden_size], epsilon=config.layer_norm_eps)
        # self.layernorm = mo_LayerNorm([config.hidden_size], eps=config.layer_norm_eps, dtype=ms.float32)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def construct(
        self,
        pixel_values: Optional[Tensor] = None,
        adaln_input: Optional[Tensor] = None,
        bool_masked_pos: Optional[Tensor] = None,
        head_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        # return_dict: Optional[bool] = False,  # not returning dict, as graph mode does not support the dict wrapper which is a non-cell class
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`ms.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            adaln_input=adaln_input,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        # logger.info(f'seq output shape {sequence_output.shape}, dtype {sequence_output.dtype}')
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # print(f'in vit model, retruning tuple. and pooled output is {pooled_output}')
        head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
        return head_outputs + encoder_outputs[1:]
