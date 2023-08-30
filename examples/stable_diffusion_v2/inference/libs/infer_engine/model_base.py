"""Inference Model Base"""

from abc import ABCMeta

import mindspore_lite as mslite


class ModelBase(metaclass=ABCMeta):
    """
    base class for model load and infer
    """

    def __init__(self, device_target="ascend", device_id=0):
        super().__init__()
        self.model = None
        context = mslite.Context()
        context.target = [device_target.lower()]
        context.ascend.device_id = device_id
        self.context = context

    def _init_model(self, model_path):
        model = mslite.Model()
        model.build_from_file(model_path, mslite.ModelType.MINDIR, self.context)
        return model
