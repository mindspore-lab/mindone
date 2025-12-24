# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
from typing import Any, Union

from ..utils import requires_backends
from .base import GenericTensor, Pipeline


class FeatureExtractionPipeline(Pipeline):
    """
    Feature extraction pipeline uses no model head. This pipeline extracts the hidden states from the base
    transformer, which can be used as features in downstream tasks.

    Example:

    ```python
    >>> from mindone.transformers import pipeline

    >>> extractor = pipeline(model="google-bert/bert-base-uncased", task="feature-extraction")
    >>> result = extractor("This is a simple test.", return_tensors=True)
    >>> result.shape  # This is a tensor of shape [1, sequence_length, hidden_dimension] representing the input string.
    [1, 8, 768]
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This feature extraction pipeline can currently be loaded from [`pipeline`] using the task identifier:
    `"feature-extraction"`.

    All models may be used for this pipeline. See a list of all models, including community-contributed models on
    [huggingface.co/models](https://huggingface.co/models).
    """

    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_tokenizer = True

    def _sanitize_parameters(self, truncation=None, tokenize_kwargs=None, return_tensors=None, **kwargs):
        if tokenize_kwargs is None:
            tokenize_kwargs = {}

        if truncation is not None:
            if "truncation" in tokenize_kwargs:
                raise ValueError(
                    "truncation parameter defined twice (given as keyword argument as well as in tokenize_kwargs)"
                )
            tokenize_kwargs["truncation"] = truncation

        preprocess_params = tokenize_kwargs

        postprocess_params = {}
        if return_tensors is not None:
            postprocess_params["return_tensors"] = return_tensors

        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, **tokenize_kwargs) -> dict[str, GenericTensor]:
        model_inputs = self.tokenizer(inputs, return_tensors="np", **tokenize_kwargs)
        if self.framework == "ms":
            requires_backends(self, ["mindspore"])
            import mindspore as ms  # noqa

            for k, v in model_inputs.items():
                model_inputs[k] = ms.tensor(v)
        return model_inputs

    def _forward(self, model_inputs):
        model_outputs = self.model(**model_inputs)
        return model_outputs

    def postprocess(self, model_outputs, return_tensors=False):
        # [0] is the first available tensor, logits or last_hidden_state.
        if return_tensors:
            return model_outputs[0]
        if self.framework == "ms":
            return model_outputs[0].float().tolist()

    def __call__(self, *args: Union[str, list[str]], **kwargs: Any) -> Union[Any, list[Any]]:
        """
        Extract the features of the input(s).

        Args:
            args (`str` or `list[str]`): One or several texts (or one list of texts) to get the features of.

        Return:
            A nested list of `float`: The features computed by the model.
        """
        return super().__call__(*args, **kwargs)
