import collections.abc
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import mindspore as ms
from mindspore import mint, nn

from transformers.models.ijepa.configuration_ijepa import IJepaConfig

from ...activations import ACT2FN
from ...mindspore_adapter import scaled_dot_product_attention
from ...mindspore_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import MSPreTrainedModel
from ...utils import (  # add_code_sample_docstrings,; add_start_docstrings,; add_start_docstrings_to_model_forward,
    logging,
    mindspore_int,
)

logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "facebook/ijepa_vith14_1k"

# General docstring
_CONFIG_FOR_DOC = "IJepaConfig"


class IJepaPatchEmbeddings(nn.Cell):
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

        self.projection = mint.nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def construct(self, pixel_values: ms.Tensor, interpolate_pos_encoding: bool = False) -> ms.Tensor:
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
        embeddings = mint.transpose(self.projection(pixel_values).flatten(2), 1, 2)
        return embeddings


class IJepaEmbeddings(nn.Cell):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: IJepaConfig, use_mask_token: bool = False) -> None:
        super().__init__()
        self.mask_token = ms.Parameter(mint.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = IJepaPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = ms.Parameter(mint.randn(1, num_patches, config.hidden_size))
        self.dropout = mint.nn.Dropout(p=config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: ms.Tensor, height: int, width: int) -> ms.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1]
        num_positions = self.position_embeddings.shape[1]

        if num_patches == num_positions and height == width:
            return self.position_embeddings

        patch_pos_embed = self.position_embeddings

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = mindspore_int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = mint.nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return patch_pos_embed

    def construct(
        self,
        pixel_values: ms.Tensor,
        bool_masked_pos: Optional[ms.Tensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> ms.Tensor:
        batch_size, _, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).to(mask_tokens.dtype)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)
        return embeddings


class IJepaPreTrainedModel(MSPreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = IJepaConfig
    base_model_prefix = "ijepa"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["IJepaEmbeddings", "IJepaLayer"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True


def eager_attention_construct(
    module: nn.Cell,
    query: ms.Tensor,
    key: ms.Tensor,
    value: ms.Tensor,
    attention_mask: Optional[ms.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    # Take the dot product between "query" and "key" to get the raw attention scores.
    attn_weights = mint.matmul(query, mint.transpose(key, -1, -2)) * scaling

    # Normalize the attention scores to probabilities.
    attn_weights = mint.nn.functional.softmax(attn_weights, dim=-1, dtype=ms.float32).to(query.dtype)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attn_weights = mint.nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    # Mask heads if we want to
    if attention_mask is not None:
        attn_weights = attn_weights * attention_mask

    attn_output = mint.matmul(attn_weights, value)
    attn_output = attn_output.swapaxes(1, 2).contiguous()

    return attn_output, attn_weights


ALL_ATTENTION_FUNCTIONS = {
    "eager": eager_attention_construct,
    "sdpa": scaled_dot_product_attention,
}


class IJepaSelfAttention(nn.Cell):
    def __init__(self, config: IJepaConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.config = config
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.dropout_prob = config.attention_probs_dropout_prob
        self.scaling = self.attention_head_size**-0.5
        self.is_causal = False

        self.query = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = mint.nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

    def transpose_for_scores(self, x: ms.Tensor) -> ms.Tensor:
        new_x_shape = x.shape[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def construct(
        self, hidden_states, head_mask: Optional[ms.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[ms.Tensor, ms.Tensor], Tuple[ms.Tensor]]:
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        attention_interface: Callable = eager_attention_construct
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and output_attentions:
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        context_layer, attention_probs = attention_interface(
            self,
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            is_causal=self.is_causal,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.dropout_prob,
        )

        new_context_layer_shape = context_layer.shape[:-2] + (self.all_head_size,)
        context_layer = context_layer.reshape(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class IJepaSelfOutput(nn.Cell):
    """
    The residual connection is defined in IJepaLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: IJepaConfig) -> None:
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = mint.nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class IJepaAttention(nn.Cell):
    def __init__(self, config: IJepaConfig) -> None:
        super().__init__()
        self.attention = IJepaSelfAttention(config)
        self.output = IJepaSelfOutput(config)
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
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[ms.Tensor, ms.Tensor], Tuple[ms.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class IJepaIntermediate(nn.Cell):
    def __init__(self, config: IJepaConfig) -> None:
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class IJepaOutput(nn.Cell):
    def __init__(self, config: IJepaConfig) -> None:
        super().__init__()
        self.dense = mint.nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = mint.nn.Dropout(p=config.hidden_dropout_prob)

    def construct(self, hidden_states: ms.Tensor, input_tensor: ms.Tensor) -> ms.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class IJepaLayer(nn.Cell):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: IJepaConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = IJepaAttention(config)
        self.intermediate = IJepaIntermediate(config)
        self.output = IJepaOutput(config)
        self.layernorm_before = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[ms.Tensor, ms.Tensor], Tuple[ms.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in IJepa, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in IJepa, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class IJepaEncoder(nn.Cell):
    def __init__(self, config: IJepaConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.CellList([IJepaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def construct(
        self,
        hidden_states: ms.Tensor,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class IJepaPooler(nn.Cell):
    def __init__(self, config: IJepaConfig):
        super().__init__()
        self.dense = mint.nn.Linear(config.hidden_size, config.pooler_output_size)
        self.activation = ACT2FN[config.pooler_act]

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


IJEPA_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`IJepaImageProcessor.__call__`]
            for details.

        head_mask (`ms.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


_EXPECTED_OUTPUT_SHAPE = [1, 256, 1280]

IJEPA_START_DOCSTRING = r"""
    This model is a MindSpore [mindspore.nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) subclass. Use it
    as a regular MindSpore Cell and refer to the MindSpore documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`IJepaConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~MSPreTrainedModel.from_pretrained`] method to load the model weights.
"""


# @add_start_docstrings(
#     "The bare IJepa Model transformer outputting raw hidden-states without any specific head on top.",
#     IJEPA_START_DOCSTRING,
# )
class IJepaModel(IJepaPreTrainedModel):
    def __init__(self, config: IJepaConfig, add_pooling_layer: bool = False, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config
        self.embeddings = IJepaEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = IJepaEncoder(config)

        self.layernorm = mint.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = IJepaPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> IJepaPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class MSPreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # @add_start_docstrings_to_model_forward(IJEPA_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     checkpoint=_CHECKPOINT_FOR_DOC,
    #     output_type=BaseModelOutputWithPooling,
    #     config_class=_CONFIG_FOR_DOC,
    #     modality="vision",
    #     expected_output=_EXPECTED_OUTPUT_SHAPE,
    # )
    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        bool_masked_pos: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`ms.Tensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


_IMAGE_CLASS_CHECKPOINT = "facebook/ijepa_vith14_1k"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


# @add_start_docstrings(
#     """
#     IJepa Model transformer with an image classification head on top (a linear layer on top of the final hidden states)
#     e.g. for ImageNet.

#     <Tip>

#         Note that it's possible to fine-tune IJepa on higher resolution images than the ones it has been trained on, by
#         setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
#         position embeddings to the higher resolution.


#     </Tip>
#     """,
#     IJEPA_START_DOCSTRING,
# )
class IJepaForImageClassification(IJepaPreTrainedModel):
    def __init__(self, config: IJepaConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.ijepa = IJepaModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = (
            mint.nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else mint.nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    # @add_start_docstrings_to_model_forward(IJEPA_INPUTS_DOCSTRING)
    # @add_code_sample_docstrings(
    #     checkpoint=_IMAGE_CLASS_CHECKPOINT,
    #     output_type=ImageClassifierOutput,
    #     config_class=_CONFIG_FOR_DOC,
    #     expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    # )
    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        head_mask: Optional[ms.Tensor] = None,
        labels: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`ms.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.ijepa(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output.mean(axis=1))

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == ms.int32 or labels.dtype == ms.int64):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = mint.nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = mint.nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = mint.nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["IJepaPreTrainedModel", "IJepaModel", "IJepaForImageClassification"]
