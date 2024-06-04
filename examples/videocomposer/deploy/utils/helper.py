import logging
import os
from typing import Union

import mindspore_lite as mslite
import numpy as np
from mindspore_lite import Model, Tensor

__all__ = ["MSLiteModelBuilder", "lite_predict"]

_logger = logging.getLogger(__name__)


class MSLiteModelBuilder:
    def __init__(
        self, device_target: str = "ascend", device_id: int = 0, lite_model_root: str = "./models/lite"
    ) -> None:
        self.lite_model_root = lite_model_root
        self.context = mslite.Context()
        self.context.target = [device_target]
        if device_target.lower() == "ascend":
            self.context.ascend.device_id = device_id
            self.context.ascend.precision_mode = "preferred_fp32"
        elif device_target.lower() == "gpu":
            self.context.gpu.device_id = device_id

    def __call__(self, name: str) -> Model:
        model = mslite.Model()

        lite_model_path = os.path.join(self.lite_model_root, f"{name}_lite.mindir")
        if not os.path.isfile(lite_model_path):
            lite_model_path = os.path.join(self.lite_model_root, f"{name}_lite_graph.mindir")

        if not os.path.isfile(lite_model_path):
            raise ValueError(f"Cannot find model at `{lite_model_path}`")

        model.build_from_file(lite_model_path, mslite.ModelType.MINDIR, self.context)
        _logger.info(f"Network built from Mindspore Lite model `{name}` with input size `{len(model.get_inputs())}`")
        return model


def lite_predict(model: Model, *data_inputs: np.ndarray, return_tensor: bool = False) -> Union[np.ndarray, Tensor]:
    """A prediction wrapper performing Mindspore Lite Predition"""
    inputs = model.get_inputs()

    if len(inputs) != len(data_inputs):
        raise ValueError(
            f"Number of inputs is not consistent between model `{len(inputs)}` and data `{len(data_inputs)}`."
        )

    for i in range(len(inputs)):
        inputs[i].set_data_from_numpy(data_inputs[i])

    # assume single output
    output = model.predict(inputs)[0]

    if not return_tensor:
        output = output.get_data_to_numpy()
    return output
