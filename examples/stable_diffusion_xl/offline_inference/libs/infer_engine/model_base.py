"""Inference Model Base"""

from abc import ABC

import mindspore_lite as mslite


class ModelBase(ABC):
    """
    base class for model load and infer
    """

    def __init__(self, device_target="ascend", device_id=0):
        super().__init__()
        self.model = None
        context = mslite.Context()
        context.target = [device_target.lower()]
        if device_target == "Ascend":
            context.ascend.device_id = device_id
            context.ascend.precision_mode = "preferred_fp32"
        elif device_target == "GPU":
            context.gpu.device_id = device_id
        self.context = context

    def _init_model(self, model_path):
        model = mslite.Model()
        model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)
        return model
