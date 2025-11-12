# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
from typing import Any, Union

from ..utils import is_vision_available, requires_backends
from .base import GenericTensor, Pipeline

if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image


class ImageFeatureExtractionPipeline(Pipeline):
    """
    Image feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```python
    >>> from mindone.transformers import pipeline

    >>> extractor = pipeline(model="google/vit-base-patch16-224", task="image-feature-extraction")
    >>> result = extractor("https://huggingface.co/datasets/Narsil/image_dummy/raw/main/parrots.png", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, sequence_lenth, hidden_dimension] representing the input image.
    [1, 197, 768]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This image feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"image-feature-extraction"`.

    All vision models may be used for this pipeline. See a list of all models, including community-contributed models on
    [huggingface.co/models](https://huggingface.co/models).
    """

    _load_processor = False
    _load_image_processor = True
    _load_feature_extractor = False
    _load_tokenizer = False

    def _sanitize_parameters(self, image_processor_kwargs=None, return_tensors=None, pool=None, **kwargs):
        preprocess_params = {} if image_processor_kwargs is None else image_processor_kwargs

        postprocess_params = {}
        if pool is not None:
            postprocess_params["pool"] = pool
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        if "timeout" in kwargs:
            preprocess_params["timeout"] = kwargs["timeout"]

        return preprocess_params, {}, postprocess_params

    def preprocess(self, image, timeout=None, **image_processor_kwargs) -> dict[str, GenericTensor]:
        image = load_image(image, timeout=timeout)
        try:
            model_inputs = self.image_processor(image, return_tensors=self.framework, **image_processor_kwargs)
            if self.framework == "ms":
                model_inputs = model_inputs.to(self.dtype)
        except ValueError:
            # for transformer image processor compatibility,
            # FIXME: consider to drop this branch if all processors are migrated to mindone.transformers in future
            requires_backends(self, ["mindspore"])
            import mindspore as ms  # noqa

            model_inputs = self.image_processor(image, return_tensors="np", **image_processor_kwargs)
            if self.framework == "ms":
                for k, v in model_inputs.items():
                    model_inputs[k] = ms.tensor(v, dtype=self.dtype)

        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, pool=None, return_tensors=False):
        pool = pool if pool is not None else False

        if pool:
            if "pooler_output" not in model_outputs:
                raise ValueError(
                    "No pooled output was returned. Make sure the model has a `pooler` layer when using the `pool` option."
                )
            outputs = model_outputs["pooler_output"]
        else:
            # [0] is the first available tensor, logits or last_hidden_state.
            outputs = model_outputs[0]

        if return_tensors:
            return outputs
        if self.framework == "ms":
            return outputs.float().tolist()

    def __call__(self, *args: Union[str, "Image.Image", list["Image.Image"], list[str]], **kwargs: Any) -> list[Any]:
        """
        Extract the features of the input(s).

        Args:
            images (`str`, `list[str]`, `PIL.Image` or `list[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is used and
                the call may block forever.
        Return:
            A nested list of `float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs)
