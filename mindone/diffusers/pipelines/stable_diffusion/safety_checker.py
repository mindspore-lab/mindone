# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from transformers import CLIPConfig

from mindspore import Parameter, Tensor, nn, ops

from mindone.transformers import CLIPVisionModel, MSPreTrainedModel

from ...utils import logging

logger = logging.get_logger(__name__)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = ops.L2Normalize(axis=1, epsilon=1e-12)(image_embeds)
    normalized_text_embeds = ops.L2Normalize(axis=1, epsilon=1e-12)(text_embeds)
    return ops.mm(normalized_image_embeds, normalized_text_embeds.t())


class StableDiffusionSafetyChecker(MSPreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Dense(config.vision_config.hidden_size, config.projection_dim, has_bias=False)

        self.concept_embeds = Parameter(ops.ones((17, config.projection_dim)), requires_grad=False)
        self.special_care_embeds = Parameter(ops.ones((3, config.projection_dim)), requires_grad=False)

        self.concept_embeds_weights = Parameter(ops.ones(17), requires_grad=False)
        self.special_care_embeds_weights = Parameter(ops.ones(3), requires_grad=False)

    # TODO: this is the onnx version of pytorch implementation, which works well in the graph.
    def construct(self, clip_input: Tensor, images: Tensor):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds)
        cos_dist = cosine_distance(image_embeds, self.concept_embeds)

        # increase this value to create a stronger `nsfw` filter
        # at the cost of increasing the possibility of filtering benign images
        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        # special_scores = special_scores.round(decimals=3)
        special_care = ops.any(special_scores > 0, axis=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).tile((1, cos_dist.shape[1]))

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # concept_scores = concept_scores.round(decimals=3)
        has_nsfw_concepts = ops.any(concept_scores > 0, axis=1)

        images[has_nsfw_concepts] = 0.0  # black image

        return images, has_nsfw_concepts
