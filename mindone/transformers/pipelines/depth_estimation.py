# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
from typing import List, Union

from ..utils import is_mindspore_available, is_vision_available, logging, requires_backends
from .base import Pipeline

if is_vision_available():
    from PIL import Image

    from ..image_utils import load_image

if is_mindspore_available():
    from ..models.auto.modeling_auto import MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES

logger = logging.get_logger(__name__)


class DepthEstimationPipeline(Pipeline):
    """
    Depth estimation pipeline using any `AutoModelForDepthEstimation`. This pipeline predicts the depth of an image.

    Example:

    ```python
    >>> from mindone.transformers import pipeline

    >>> depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf")
    >>> output = depth_estimator("http://images.cocodataset.org/val2017/000000039769.jpg")
    >>> # This is a tensor with the values being the depth expressed in meters for each pixel
    >>> output["predicted_depth"].shape
    [480, 640]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)


    This depth estimation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"depth-estimation"`.

    See the list of available models on [huggingface.co/models](https://huggingface.co/models?filter=depth-estimation).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        requires_backends(self, "vision")
        self.check_model_type(MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES)

    def __call__(self, inputs: Union[str, List[str], "Image.Image", List["Image.Image"]] = None, **kwargs):
        """
        Predict the depth(s) of the image(s) passed as inputs.

        Args:
            inputs (`str`, `List[str]`, `PIL.Image` or `List[PIL.Image]`):
                The pipeline handles three types of images:

                - A string containing a http link pointing to an image
                - A string containing a local path to an image
                - An image loaded in PIL directly

                The pipeline accepts either a single image or a batch of images, which must then be passed as a string.
                Images in a batch must all be in the same format: all as http links, all as local paths, or all as PIL
                images.
            parameters (`Dict`, *optional*):
                A dictionary of argument names to parameter values, to control pipeline behaviour.
                The only parameter available right now is `timeout`, which is the length of time, in seconds,
                that the pipeline should wait before giving up on trying to download an image.
            timeout (`float`, *optional*, defaults to None):
                The maximum time in seconds to wait for fetching images from the web. If None, no timeout is set and
                the call may block forever.

        Return:
            A dictionary or a list of dictionaries containing result. If the input is a single image, will return a
            dictionary, if the input is a list of several images, will return a list of dictionaries corresponding to
            the images.

            The dictionaries contain the following keys:

            - **predicted_depth** (`mindspore.Tensor`) -- The predicted depth by the model as a `mindspore.Tensor`.
            - **depth** (`PIL.Image`) -- The predicted depth by the model as a `PIL.Image`.
        """
        # After deprecation of this is completed, remove the default `None` value for `images`
        if "images" in kwargs:
            inputs = kwargs.pop("images")
        if inputs is None:
            raise ValueError("Cannot call the depth-estimation pipeline without an inputs argument!")
        return super().__call__(inputs, **kwargs)

    def _sanitize_parameters(self, timeout=None, parameters=None, **kwargs):
        preprocess_params = {}
        if timeout is not None:
            preprocess_params["timeout"] = timeout
        if isinstance(parameters, dict) and "timeout" in parameters:
            preprocess_params["timeout"] = parameters["timeout"]
        return preprocess_params, {}, {}

    def preprocess(self, image, timeout=None):
        image = load_image(image, timeout)
        try:
            model_inputs = self.image_processor(images=image, return_tensors=self.framework)
            if self.framework == "ms":
                model_inputs = model_inputs.to(self.mindspore_dtype)
        except ValueError:
            # for transformer image processor compatibility
            # FIXME: consider to drop this branch if all processors are migrated to mindone.transformers in future
            requires_backends(self, ["mindspore"])
            import mindspore as ms  # noqa

            model_inputs = self.image_processor(image, return_tensors="np")
            if self.framework == "ms":
                for k, v in model_inputs.items():
                    model_inputs[k] = ms.tensor(v, dtype=self.mindspore_dtype)

        model_inputs["target_size"] = image.size[::-1]
        return model_inputs

    def _forward(self, model_inputs):
        target_size = model_inputs.pop("target_size")
        model_outputs = self.model(**model_inputs)
        model_outputs["target_size"] = target_size
        return model_outputs

    def postprocess(self, model_outputs):
        outputs = self.image_processor.post_process_depth_estimation(
            model_outputs,
            # this acts as `source_sizes` for ZoeDepth and as `target_sizes` for the rest of the models so do *not*
            # replace with `target_sizes = [model_outputs["target_size"]]`
            [model_outputs["target_size"]],
        )

        formatted_outputs = []
        for output in outputs:
            depth = output["predicted_depth"].float().numpy()
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = Image.fromarray((depth * 255).astype("uint8"))

            formatted_outputs.append({"predicted_depth": output["predicted_depth"], "depth": depth})

        return formatted_outputs[0] if len(outputs) == 1 else formatted_outputs
