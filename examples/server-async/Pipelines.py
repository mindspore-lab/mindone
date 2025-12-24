import logging
import os
from dataclasses import dataclass, field
from typing import List, Optional

from pydantic import BaseModel

import mindspore as ms

from mindone.diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import StableDiffusion3Pipeline

logger = logging.getLogger(__name__)


class TextToImageInput(BaseModel):
    model: str
    prompt: str
    size: Optional[str] = None
    n: Optional[int] = None


@dataclass
class PresetModels:
    SD3: List[str] = field(default_factory=lambda: ["stabilityai/stable-diffusion-3-medium"])
    SD3_5: List[str] = field(
        default_factory=lambda: [
            "stabilityai/stable-diffusion-3.5-large",
            "stabilityai/stable-diffusion-3.5-large-turbo",
            "stabilityai/stable-diffusion-3.5-medium",
        ]
    )


class TextToImagePipelineSD3:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.getenv("MODEL_PATH")
        self.pipeline: Optional[StableDiffusion3Pipeline] = None

    def start(self):
        if ms.device_context.ascend.is_available():
            model_path = self.model_path or "stabilityai/stable-diffusion-3.5-large"
            logger.info("Loading Ascend")
            self.pipeline = StableDiffusion3Pipeline.from_pretrained(
                model_path,
                mindspore_dtype=ms.float16,
            )
        else:
            raise Exception("No Ascend device available")


class ModelPipelineInitializer:
    def __init__(self, model: str = "", type_models: str = "t2im"):
        self.model = model
        self.type_models = type_models
        self.pipeline = None
        self.model_type = None

    def initialize_pipeline(self):
        if not self.model:
            raise ValueError("Model name not provided")

        # Check if model exists in PresetModels
        preset_models = PresetModels()

        # Determine which model type we're dealing with
        if self.model in preset_models.SD3:
            self.model_type = "SD3"
        elif self.model in preset_models.SD3_5:
            self.model_type = "SD3_5"

        # Create appropriate pipeline based on model type and type_models
        if self.type_models == "t2im":
            if self.model_type in ["SD3", "SD3_5"]:
                self.pipeline = TextToImagePipelineSD3(self.model)
            else:
                raise ValueError(f"Model type {self.model_type} not supported for text-to-image")
        elif self.type_models == "t2v":
            raise ValueError(f"Unsupported type_models: {self.type_models}")

        return self.pipeline
