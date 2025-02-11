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
    "scheduling_consistency_models": ["CMStochasticIterativeScheduler"],
    "scheduling_consistency_decoder": ["ConsistencyDecoderScheduler"],
    "scheduling_ddim_inverse": ["DDIMInverseScheduler"],
    "scheduling_ddim_parallel": ["DDIMParallelScheduler"],
    "scheduling_ddim": ["DDIMScheduler"],
    "scheduling_ddim_cogvideox": ["CogVideoXDDIMScheduler"],
    "scheduling_ddpm_parallel": ["DDPMParallelScheduler"],
    "scheduling_ddpm": ["DDPMScheduler"],
    "scheduling_ddpm_wuerstchen": ["DDPMWuerstchenScheduler"],
    "scheduling_deis_multistep": ["DEISMultistepScheduler"],
    "scheduling_dpm_cogvideox": ["CogVideoXDPMScheduler"],
    "scheduling_dpmsolver_multistep": ["DPMSolverMultistepScheduler"],
    "scheduling_dpmsolver_multistep_inverse": ["DPMSolverMultistepInverseScheduler"],
    "scheduling_dpmsolver_singlestep": ["DPMSolverSinglestepScheduler"],
    "scheduling_edm_dpmsolver_multistep": ["EDMDPMSolverMultistepScheduler"],
    "scheduling_edm_euler": ["EDMEulerScheduler"],
    "scheduling_euler_ancestral_discrete": ["EulerAncestralDiscreteScheduler"],
    "scheduling_euler_discrete": ["EulerDiscreteScheduler"],
    "scheduling_flow_match_euler_discrete": ["FlowMatchEulerDiscreteScheduler"],
    "scheduling_flow_match_heun_discrete": ["FlowMatchHeunDiscreteScheduler"],
    "scheduling_heun_discrete": ["HeunDiscreteScheduler"],
    "scheduling_ipndm": ["IPNDMScheduler"],
    "scheduling_k_dpm_2_ancestral_discrete": ["KDPM2AncestralDiscreteScheduler"],
    "scheduling_k_dpm_2_discrete": ["KDPM2DiscreteScheduler"],
    "scheduling_lcm": ["LCMScheduler"],
    "scheduling_lms_discrete": ["LMSDiscreteScheduler"],
    "scheduling_pndm": ["PNDMScheduler"],
    "scheduling_repaint": ["RePaintScheduler"],
    "scheduling_sasolver": ["SASolverScheduler"],
    "scheduling_sde_ve": ["ScoreSdeVeScheduler"],
    "scheduling_tcd": ["TCDScheduler"],
    "scheduling_unclip": ["UnCLIPScheduler"],
    "scheduling_unipc_multistep": ["UniPCMultistepScheduler"],
    "scheduling_vq_diffusion": ["VQDiffusionScheduler"],
    "scheduling_utils": ["AysSchedules", "KarrasDiffusionSchedulers", "SchedulerMixin"],
}

if TYPE_CHECKING:
    from .scheduling_consistency_decoder import ConsistencyDecoderScheduler
    from .scheduling_consistency_models import CMStochasticIterativeScheduler
    from .scheduling_ddim import DDIMScheduler
    from .scheduling_ddim_cogvideox import CogVideoXDDIMScheduler
    from .scheduling_ddim_inverse import DDIMInverseScheduler
    from .scheduling_ddim_parallel import DDIMParallelScheduler
    from .scheduling_ddpm import DDPMScheduler
    from .scheduling_ddpm_parallel import DDPMParallelScheduler
    from .scheduling_ddpm_wuerstchen import DDPMWuerstchenScheduler
    from .scheduling_deis_multistep import DEISMultistepScheduler
    from .scheduling_dpm_cogvideox import CogVideoXDPMScheduler
    from .scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
    from .scheduling_dpmsolver_multistep_inverse import DPMSolverMultistepInverseScheduler
    from .scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
    from .scheduling_edm_dpmsolver_multistep import EDMDPMSolverMultistepScheduler
    from .scheduling_edm_euler import EDMEulerScheduler
    from .scheduling_euler_ancestral_discrete import EulerAncestralDiscreteScheduler
    from .scheduling_euler_discrete import EulerDiscreteScheduler
    from .scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    from .scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
    from .scheduling_heun_discrete import HeunDiscreteScheduler
    from .scheduling_ipndm import IPNDMScheduler
    from .scheduling_k_dpm_2_ancestral_discrete import KDPM2AncestralDiscreteScheduler
    from .scheduling_k_dpm_2_discrete import KDPM2DiscreteScheduler
    from .scheduling_lcm import LCMScheduler
    from .scheduling_lms_discrete import LMSDiscreteScheduler
    from .scheduling_pndm import PNDMScheduler
    from .scheduling_repaint import RePaintScheduler
    from .scheduling_sasolver import SASolverScheduler
    from .scheduling_sde_ve import ScoreSdeVeScheduler
    from .scheduling_tcd import TCDScheduler
    from .scheduling_unclip import UnCLIPScheduler
    from .scheduling_unipc_multistep import UniPCMultistepScheduler
    from .scheduling_utils import AysSchedules, KarrasDiffusionSchedulers, SchedulerMixin
    from .scheduling_vq_diffusion import VQDiffusionScheduler

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
