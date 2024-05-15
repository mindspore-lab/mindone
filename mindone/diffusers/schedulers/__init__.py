# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

from ..utils import _LazyModule

_import_structure = {
    "scheduling_ddim": ["DDIMScheduler"],
    "scheduling_ddpm": ["DDPMScheduler"],
    "scheduling_deis_multistep": ["DEISMultistepScheduler"],
    "scheduling_dpmsolver_multistep": ["DPMSolverMultistepScheduler"],
    "scheduling_dpmsolver_singlestep": ["DPMSolverSinglestepScheduler"],
    "scheduling_euler_ancestral_discrete": ["EulerAncestralDiscreteScheduler"],
    "scheduling_euler_discrete": ["EulerDiscreteScheduler"],
    "scheduling_heun_discrete": ["HeunDiscreteScheduler"],
    "scheduling_k_dpm_2_ancestral_discrete": ["KDPM2AncestralDiscreteScheduler"],
    "scheduling_k_dpm_2_discrete": ["KDPM2DiscreteScheduler"],
    "scheduling_lms_discrete": ["LMSDiscreteScheduler"],
    "scheduling_pndm": ["PNDMScheduler"],
    "scheduling_unipc_multistep": ["UniPCMultistepScheduler"],
    "scheduling_utils": ["KarrasDiffusionSchedulers", "SchedulerMixin"],
}

if TYPE_CHECKING:
    from .scheduling_ddim import DDIMScheduler
    from .scheduling_ddpm import DDPMScheduler
    from .scheduling_deis_multistep import DEISMultistepScheduler
    from .scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    from .scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
    from .scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
    from .scheduling_euler_discrete import EulerDiscreteScheduler
    from .scheduling_heun_discrete import HeunDiscreteScheduler
    from .scheduling_k_dpm_2_ancestral_discrete import KDPM2AncestralDiscreteScheduler
    from .scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
    from .scheduling_lms_discrete import LMSDiscreteScheduler
    from .scheduling_pndm import PNDMScheduler
    from .scheduling_unipc_multistep import UniPCMultistepScheduler
    from .scheduling_utils import KarrasDiffusionSchedulers, SchedulerMixin

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
