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

from mindspore import Parameter, Tensor, mint, ops

from mindone.transformers import CLIPVisionModel, MSPreTrainedModel

from ...utils import logging

logger = logging.get_logger(__name__)


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = mint.nn.functional.normalize(image_embeds)
    normalized_text_embeds = mint.nn.functional.normalize(text_embeds)
    return mint.mm(normalized_image_embeds, normalized_text_embeds.t())


class StableDiffusionSafetyChecker(MSPreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "clip_input"

    _keys_to_ignore_on_load_unexpected = ["vision_model.vision_model.embeddings.position_ids"]
    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = mint.nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = Parameter(mint.ones((17, config.projection_dim)), requires_grad=False)
        self.special_care_embeds = Parameter(mint.ones((3, config.projection_dim)), requires_grad=False)

        self.concept_embeds_weights = Parameter(mint.ones(17), requires_grad=False)
        self.special_care_embeds_weights = Parameter(mint.ones(3), requires_grad=False)

    # TODO: this is the onnx version of pytorch implementation, which works well in the graph.
    def construct(self, clip_input: Tensor, images: Tensor):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        special_cos_dist = cosine_distance(image_embeds, self.special_care_embeds).float()
        cos_dist = cosine_distance(image_embeds, self.concept_embeds).float()

        # increase this value to create a stronger `nsfw` filter
        # at the cost of increasing the possibility of filtering benign images
        adjustment = 0.0

        special_scores = special_cos_dist - self.special_care_embeds_weights + adjustment
        # special_scores = special_scores.round(decimals=3)
        special_scores = mint.round(special_scores, decimals=3)
        special_care = mint.any(special_scores > 0, dim=1)
        special_adjustment = special_care * 0.01
        special_adjustment = special_adjustment.unsqueeze(1).tile((1, cos_dist.shape[1]))

        concept_scores = (cos_dist - self.concept_embeds_weights) + special_adjustment
        # concept_scores = concept_scores.round(decimals=3)
        concept_scores = mint.round(concept_scores, decimals=3)
        has_nsfw_concepts = mint.any(concept_scores > 0, dim=1)

        if ops.is_tensor(images):
            images[has_nsfw_concepts] = 0.0  # black image
        else:
            # TODO: if has_nsfw_concepts is tensor and images is array, the images will be wrong.
            images[has_nsfw_concepts.numpy()] = 0.0  # black image

        return images, has_nsfw_concepts
