# coding=utf-8
# Copyright 2023 HUST-VL and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mindspore ViTMatte model."""

from dataclasses import dataclass
from typing import Optional, Tuple

from transformers import VitMatteConfig
from transformers.utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)

import mindspore as ms
from mindspore import mint, nn

from mindone.models.utils import normal_, zeros_

from ...modeling_utils import PreTrainedModel
from ...utils.backbone_utils import load_backbone

# General docstring
_CONFIG_FOR_DOC = "VitMatteConfig"


@dataclass
class ImageMattingOutput(ModelOutput):
    """
    Class for outputs of image matting models.

    Args:
        loss (`ms.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Loss.
        alphas (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
           Estimated alpha values.
        hidden_states (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `ms.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(ms.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `ms.Tensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[ms.Tensor] = None
    alphas: ms.Tensor = None
    hidden_states: Optional[Tuple[ms.Tensor]] = None
    attentions: Optional[Tuple[ms.Tensor]] = None


class VitMattePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VitMatteConfig
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = []

    def _init_weights(self, module):
        if isinstance(module, mint.nn.Conv2d):
            normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                zeros_(module.bias)


class VitMatteBasicConv3x3(nn.Cell):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """

    def __init__(self, config, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        self.conv = mint.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.batch_norm = mint.nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)
        self.relu = mint.nn.ReLU()

    def construct(self, hidden_state):
        hidden_state = self.conv(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = self.relu(hidden_state)

        return hidden_state


class VitMatteConvStream(nn.Cell):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """

    def __init__(self, config):
        super().__init__()

        # We use a default in-case there isn't a backbone config set. This is for backwards compatibility and
        # to enable loading HF backbone models.
        in_channels = 4
        if config.backbone_config is not None:
            in_channels = config.backbone_config.num_channels

        out_channels = config.convstream_hidden_sizes

        convs = []
        self.conv_chans = [in_channels] + out_channels

        for i in range(len(self.conv_chans) - 1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i + 1]
            convs.append(VitMatteBasicConv3x3(config, in_chan_, out_chan_))

        self.convs = nn.CellList(convs)

    def construct(self, pixel_values):
        out_dict = {"detailed_feature_map_0": pixel_values}
        embeddings = pixel_values
        for i in range(len(self.convs)):
            embeddings = self.convs[i](embeddings)
            name_ = "detailed_feature_map_" + str(i + 1)
            out_dict[name_] = embeddings

        return out_dict


class VitMatteFusionBlock(nn.Cell):
    """
    Simple fusion block to fuse features from ConvStream and Plain Vision Transformer.
    """

    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.conv = VitMatteBasicConv3x3(config, in_channels, out_channels, stride=1, padding=1)

    def construct(self, features, detailed_feature_map):
        # FIXME mint.nn.functional.interpolate does not support `scale_factor` when `mode="bilinear"`, so we use `size` instead.
        # upscaled_features = mint.nn.functional.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        _, _, h, w = features.shape
        upscaled_features = mint.nn.functional.interpolate(
            features, size=(h * 2, w * 2), mode="bilinear", align_corners=False
        )

        out = mint.cat([detailed_feature_map, upscaled_features], dim=1)
        out = self.conv(out)

        return out


class VitMatteHead(nn.Cell):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """

    def __init__(self, config):
        super().__init__()

        in_channels = config.fusion_hidden_sizes[-1]
        mid_channels = 16

        self.matting_convs = nn.SequentialCell(
            mint.nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            mint.nn.BatchNorm2d(mid_channels),
            mint.nn.ReLU(),
            mint.nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0),
        )

    def construct(self, hidden_state):
        hidden_state = self.matting_convs(hidden_state)

        return hidden_state


class VitMatteDetailCaptureModule(nn.Cell):
    """
    Simple and lightweight Detail Capture Module for ViT Matting.
    """

    def __init__(self, config):
        super().__init__()
        if len(config.fusion_hidden_sizes) != len(config.convstream_hidden_sizes) + 1:
            raise ValueError(
                "The length of fusion_hidden_sizes should be equal to the length of convstream_hidden_sizes + 1."
            )

        self.config = config
        self.convstream = VitMatteConvStream(config)
        self.conv_chans = self.convstream.conv_chans

        fusion_blocks = []
        self.fusion_channels = [config.hidden_size] + config.fusion_hidden_sizes

        for i in range(len(self.fusion_channels) - 1):
            fusion_blocks.append(
                VitMatteFusionBlock(
                    config=config,
                    in_channels=self.fusion_channels[i] + self.conv_chans[-(i + 1)],
                    out_channels=self.fusion_channels[i + 1],
                )
            )
        self.fusion_blocks = nn.CellList(fusion_blocks)
        self.matting_head = VitMatteHead(config)

    def construct(self, features, pixel_values):
        detail_features = self.convstream(pixel_values)
        for i in range(len(self.fusion_blocks)):
            detailed_feature_map_name = "detailed_feature_map_" + str(len(self.fusion_blocks) - i - 1)
            features = self.fusion_blocks[i](features, detail_features[detailed_feature_map_name])

        alphas = mint.sigmoid(self.matting_head(features))

        return alphas


VITMATTE_START_DOCSTRING = r"""
    Parameters:
    This model is a MindSpore [mindspore.nn.Cell](https://www.mindspore.cn/docs/en/master/api_python/nn/mindspore.nn.Cell.html) subclass.
    Use it as a regular MindSpore Module and refer to the MindSpore documentation for all matter related to general usage and behavior.

        config ([`UperNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VITMATTE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`ms.Tensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`VitMatteImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers in case the backbone has them. See
            `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers of the backbone. See `hidden_states` under
            returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """ViTMatte framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    VITMATTE_START_DOCSTRING,
)
class VitMatteForImageMatting(VitMattePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.backbone = load_backbone(config)
        self.decoder = VitMatteDetailCaptureModule(config)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VITMATTE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=ImageMattingOutput, config_class=_CONFIG_FOR_DOC)
    def construct(
        self,
        pixel_values: Optional[ms.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[ms.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        labels (`ms.Tensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth image matting for computing the loss.

        Returns:

        Examples:

        ```python
        >>> from transformers import VitMatteImageProcessor
        >>> from mindone.transformers import VitMatteForImageMatting
        >>> import mindspore as ms
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
        >>> model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")
        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
        ... )
        >>> trimap = Image.open(filepath).convert("L")

        >>> # prepare image + trimap for the model
        >>> inputs = processor(images=image, trimaps=trimap, return_tensors="np")
        >>> inputs = {k: ms.tensor(v) for k, v in inputs.items()}

        >>> alphas = model(**inputs).alphas
        >>> print(alphas.shape)
        [1, 1, 640, 960]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        loss = None
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )

        features = outputs.feature_maps[-1]
        alphas = self.decoder(features, pixel_values)

        if not return_dict:
            output = (alphas,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageMattingOutput(
            loss=loss,
            alphas=alphas,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["VitMattePreTrainedModel", "VitMatteForImageMatting"]
