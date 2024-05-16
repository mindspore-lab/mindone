__version__ = "0.27.1"

from typing import TYPE_CHECKING

from .utils import _LazyModule

# Lazy Import based on
# https://github.com/huggingface/transformers/blob/main/src/transformers/__init__.py

# When adding a new object to this init, please add it to `_import_structure`. The `_import_structure` is a dictionary submodule to list of object names,
# and is used to defer the actual importing for when the objects are requested.
# This way `import diffusers` provides the names in the namespace without actually importing anything (and especially none of the backends).

_import_structure = {
    "configuration_utils": ["ConfigMixin"],
    "models": [
        "AutoencoderKL",
        "ModelMixin",
        "UNet2DConditionModel",
        "UNet2DModel",
    ],
    "optimization": [
        "get_constant_schedule",
        "get_constant_schedule_with_warmup",
        "get_cosine_schedule_with_warmup",
        "get_cosine_with_hard_restarts_schedule_with_warmup",
        "get_linear_schedule_with_warmup",
        "get_polynomial_decay_schedule_with_warmup",
        "get_scheduler",
    ],
    "pipelines": [
        "DDIMPipeline",
        "DDPMPipeline",
        "DiffusionPipeline",
        "StableDiffusionPipeline",
        "StableDiffusionXLPipeline",
    ],
    "schedulers": [
        "ConsistencyDecoderScheduler",
        "CMStochasticIterativeScheduler",
        "DDIMScheduler",
        "DDIMInverseScheduler",
        "DDIMParallelScheduler",
        "DDPMScheduler",
        "DDPMParallelScheduler",
        "DDPMWuerstchenScheduler",
        "DEISMultistepScheduler",
        "DPMSolverMultistepScheduler",
        "DPMSolverMultistepInverseScheduler",
        "DPMSolverSinglestepScheduler",
        "EDMEulerScheduler",
        "EulerAncestralDiscreteScheduler",
        "EulerDiscreteScheduler",
        "HeunDiscreteScheduler",
        "KDPM2AncestralDiscreteScheduler",
        "KDPM2DiscreteScheduler",
        "LCMScheduler",
        "LMSDiscreteScheduler",
        "PNDMScheduler",
        "RePaintScheduler",
        "SASolverScheduler",
        "UnCLIPScheduler",
        "UniPCMultistepScheduler",
        "VQDiffusionScheduler",
        "SchedulerMixin",
    ],
    "utils": [
        "logging",
    ],
}

if TYPE_CHECKING:
    from .configuration_utils import ConfigMixin
    from .models import AutoencoderKL, ModelMixin, UNet2DConditionModel, UNet2DModel
    from .optimization import (
        get_constant_schedule,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_cosine_with_hard_restarts_schedule_with_warmup,
        get_linear_schedule_with_warmup,
        get_polynomial_decay_schedule_with_warmup,
        get_scheduler,
    )
    from .pipelines import (
        DDIMPipeline,
        DDPMPipeline,
        DiffusionPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )
    from .schedulers import (
        CMStochasticIterativeScheduler,
        ConsistencyDecoderScheduler,
        DDIMInverseScheduler,
        DDIMParallelScheduler,
        DDIMScheduler,
        DDPMParallelScheduler,
        DDPMScheduler,
        DDPMWuerstchenScheduler,
        DEISMultistepScheduler,
        DPMSolverMultistepInverseScheduler,
        DPMSolverMultistepScheduler,
        DPMSolverSinglestepScheduler,
        EDMDPMSolverMultistepScheduler,
        EDMEulerScheduler,
        EulerAncestralDiscreteScheduler,
        EulerDiscreteScheduler,
        HeunDiscreteScheduler,
        IPNDMScheduler,
        KDPM2AncestralDiscreteScheduler,
        KDPM2DiscreteScheduler,
        LCMScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        RePaintScheduler,
        SASolverScheduler,
        SchedulerMixin,
        ScoreSdeVeScheduler,
        TCDScheduler,
        UnCLIPScheduler,
        UniPCMultistepScheduler,
        VQDiffusionScheduler,
    )
    from .utils import logging

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
