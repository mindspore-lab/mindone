"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/controlnet/multicontrolnet.py."""

from ...models.controlnets.multicontrolnet import MultiControlNetModel
from ...utils import deprecate, logging

logger = logging.get_logger(__name__)


class MultiControlNetModel(MultiControlNetModel):
    def __init__(self, *args, **kwargs):
        deprecation_message = "Importing `MultiControlNetModel` from `diffusers.pipelines.controlnet.multicontrolnet` is deprecated and this will be removed in a future version. Please use `from diffusers.models.controlnets.multicontrolnet import MultiControlNetModel`, instead."  # noqa: E501
        deprecate("diffusers.pipelines.controlnet.multicontrolnet.MultiControlNetModel", "0.34", deprecation_message)
        super().__init__(*args, **kwargs)
