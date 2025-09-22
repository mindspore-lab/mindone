# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
from typing import Any, Dict, List, Union

import numpy as np
from transformers.utils import add_end_docstrings

import mindspore as ms

from ..utils import is_mindspore_available, is_vision_available, logging, requires_backends
from .base import Pipeline, build_pipeline_init_args

if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_mindspore_available():
    from ..models.auto.modeling_auto import (
        MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
        MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES,
    )


logger = logging.get_logger(__name__)


Prediction = Dict[str, Any]
Predictions = List[Prediction]


@add_end_docstrings(build_pipeline_init_args(has_image_processor=True))
class ImageSegmentationPipeline(Pipeline):
    """
    Image segmentation pipeline using any `AutoModelForXXXSegmentation`. This pipeline predicts masks of objects and
    their classes.

    Example:

    ```python
    >>> from mindone.transformers import pipeline

    >>> segmenter = pipeline("image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")
    >>> segments = segmenter("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/segmentation_input.jpg")
    >>> len(segments)
    8

    >>> segments[0]["label"]
    'building'

    >>> segments[1]["label"]
    'sky'

    >>> type(segments[0]["mask"])  # This is a black and white mask showing where is the road on the original image.
    <class 'PIL.Image.Image'>

    >>> segments[0]["mask"].size
    (612, 415)
    ```


    This image segmentation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"image-segmentation"`.

    See the list of available models on
    [huggingface.co/models](https://huggingface.co/models?filter=image-segmentation).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.framework != "ms":
            raise ValueError(f"The {self.__class__} is only available in MindSpore.")

        requires_backends(self, "vision")
        mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES.copy()
        mapping.update(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES)
        mapping.update(MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES)
        mapping.update(MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES)
        self.check_model_type(mapping)

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        postprocess_kwargs = {}
        if "subtask" in kwargs:
            postprocess_kwargs["subtask"] = kwargs["subtask"]
            preprocess_kwargs["subtask"] = kwargs["subtask"]
        if "threshold" in kwargs:
            postprocess_kwargs["threshold"] = kwargs["threshold"]
        if "mask_threshold" in kwargs:
            postprocess_kwargs["mask_threshold"] = kwargs["mask_threshold"]
        if "overlap_mask_area_threshold" in kwargs:
            postprocess_kwargs["overlap_mask_area_threshold"] = kwargs["overlap_mask_area_threshold"]
        if "timeout" in kwargs:
            preprocess_kwargs["timeout"] = kwargs["timeout"]

        return preprocess_kwargs, {}, postprocess_kwargs

    def __call__(self, inputs=None, **kwargs) -> Union[Predictions, List[Prediction]]:
        """
        Perform segmentation (detect masks & classes) in the image(s) passed as inputs.

        Args:
            inputs (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing an HTTP(S) link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images. Images in a batch must all be in the
                same format: all as HTTP(S) links, all as local paths, or all as PIL images.
            subtask (`str`, *optional*):
                Segmentation task to be performed, choose [`semantic`, `instance` and `panoptic`] depending on model
                capabilities. If not set, the pipeline will attempt tp resolve in the following order:
                  `panoptic`, `instance`, `semantic`.
            threshold (`float`, *optional*, defaults to 0.9):
                Probability threshold to filter out predicted masks.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold to use when turning the predicted masks into binary values.
            overlap_mask_area_threshold (`float`, *optional*, defaults to 0.5):
                Mask overlap threshold to eliminate small, disconnected segments.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A dictionary or a list of dictionaries containing the result. If the input is a single image, will return a
            list of dictionaries, if the input is a list of several images, will return a list of list of dictionaries
            corresponding to each image.

            The dictionaries contain the mask, label and score (where applicable) of each detected object and contains
            the following keys:

            - **label** (`str`) -- The class label identified by the model.
            - **mask** (`PIL.Image`) -- A binary mask of the detected object as a Pil Image of shape (width, height) of
              the original image. Returns a mask filled with zeros if no object is found.
            - **score** (*optional* `float`) -- Optionally, when the model is capable of estimating a confidence of the
              "object" described by the label and the mask.
        """
        # After deprecation of this is completed, remove the default `None` value for `images`
        if "images" in kwargs:
            inputs = kwargs.pop("images")
        if inputs is None:
            raise ValueError("Cannot call the image-segmentation pipeline without an inputs argument!")
        return super().__call__(inputs, **kwargs)

    def preprocess(self, image, subtask=None, timeout=None):
        image = load_image(image, timeout=timeout)
        target_size = [(image.height, image.width)]
        if self.model.config.__class__.__name__ == "OneFormerConfig":
            if subtask is None:
                kwargs = {}
            else:
                kwargs = {"task_inputs": [subtask]}
            inputs = self.image_processor(images=[image], return_tensors="np", **kwargs)
            if self.framework == "ms":
                for k, v in inputs.items():
                    inputs[k] = ms.Tensor(v, dtype=self.mindspore_dtype)
            inputs["task_inputs"] = self.tokenizer(
                inputs["task_inputs"],
                padding="max_length",
                max_length=self.model.config.task_seq_len,
                return_tensors=self.framework,
            )["input_ids"]
        else:
            inputs = self.image_processor(images=[image], return_tensors="np")
            if self.framework == "ms":
                for k, v in inputs.items():
                    inputs[k] = ms.Tensor(v, dtype=self.mindspore_dtype)
        inputs["target_size"] = target_size
        return inputs

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        model_outputs = self.model(**model_inputs)
        model_outputs["target_size"] = target_size
        return model_outputs

    def postprocess(
        self, model_outputs, subtask=None, threshold=0.9, mask_threshold=0.5, overlap_mask_area_threshold=0.5
    ):
        fn = None
        if subtask in {"panoptic", None} and hasattr(self.image_processor, "post_process_panoptic_segmentation"):
            fn = self.image_processor.post_process_panoptic_segmentation
        elif subtask in {"instance", None} and hasattr(self.image_processor, "post_process_instance_segmentation"):
            fn = self.image_processor.post_process_instance_segmentation

        if fn is not None:
            outputs = fn(
                model_outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                overlap_mask_area_threshold=overlap_mask_area_threshold,
                target_sizes=model_outputs["target_size"],
            )[0]

            annotation = []
            segmentation = outputs["segmentation"]

            for segment in outputs["segments_info"]:
                mask = (segmentation == segment["id"]) * 255
                mask = Image.fromarray(mask.numpy().astype(np.uint8), mode="L")
                label = self.model.config.id2label[segment["label_id"]]
                score = segment["score"]
                annotation.append({"score": score, "label": label, "mask": mask})

        elif subtask in {"semantic", None} and hasattr(self.image_processor, "post_process_semantic_segmentation"):
            outputs = self.image_processor.post_process_semantic_segmentation(
                model_outputs, target_sizes=model_outputs["target_size"]
            )[0]

            annotation = []
            segmentation = outputs.numpy()
            labels = np.unique(segmentation)

            for label in labels:
                mask = (segmentation == label) * 255
                mask = Image.fromarray(mask.astype(np.uint8), mode="L")
                label = self.model.config.id2label[label]
                annotation.append({"score": None, "label": label, "mask": mask})
        else:
            raise ValueError(f"Subtask {subtask} is not supported for model {type(self.model)}")
        return annotation
