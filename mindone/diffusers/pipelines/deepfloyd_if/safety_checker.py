from transformers import CLIPConfig

from mindspore import mint

from ....transformers import CLIPVisionModelWithProjection, MSPreTrainedModel
from ...utils import logging

logger = logging.get_logger(__name__)


class IFSafetyChecker(MSPreTrainedModel):
    config_class = CLIPConfig

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModelWithProjection(config.vision_config)

        self.p_head = mint.nn.Linear(config.vision_config.projection_dim, 1)
        self.w_head = mint.nn.Linear(config.vision_config.projection_dim, 1)

    # Refer to onnx version in pipelines.stable_diffusion.safe_checker.StableDiffusionSafetyChecker
    def construct(self, clip_input, images, p_threshold=0.5, w_threshold=0.5):
        image_embeds = self.vision_model(clip_input)[0]

        nsfw_detected = self.p_head(image_embeds)
        nsfw_detected = mint.any(nsfw_detected > p_threshold, dim=1)
        images[nsfw_detected] = 0.0

        watermark_detected = self.w_head(image_embeds)
        watermark_detected = mint.any(watermark_detected > w_threshold, dim=1)
        images[watermark_detected] = 0.0

        return images, nsfw_detected, watermark_detected
