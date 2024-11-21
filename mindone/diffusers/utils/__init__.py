# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from .constants import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    DIFFUSERS_DYNAMIC_MODULE_NAME,
    FLAX_WEIGHTS_NAME,
    HF_MODULES_CACHE,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFETENSORS_FILE_EXTENSION,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)
from .deprecation_utils import deprecate
from .dynamic_modules_utils import get_class_from_dynamic_module
from .export_utils import export_to_gif, export_to_obj, export_to_ply, export_to_video
from .hub_utils import (
    PushToHubMixin,
    _add_variant,
    _get_checkpoint_shard_files,
    _get_model_file,
    extract_commit_hash,
    http_user_agent,
)
from .import_utils import (
    BACKENDS_MAPPING,
    _LazyModule,
    is_bs4_available,
    is_ftfy_available,
    is_invisible_watermark_available,
    is_matplotlib_available,
    is_mindspore_version,
    is_opencv_available,
    is_peft_version,
    is_scipy_available,
    is_sentencepiece_available,
    is_transformers_available,
    maybe_import_module_in_mindone,
)
from .loading_utils import load_image, load_video
from .logging import get_logger
from .outputs import BaseOutput
from .peft_utils import (
    delete_adapter_layers,
    get_adapter_name,
    get_peft_kwargs,
    recurse_remove_peft_layers,
    scale_lora_layers,
    set_adapter_layers,
    set_weights_and_activate_adapters,
    unscale_lora_layers,
)
from .pil_utils import PIL_INTERPOLATION, make_image_grid, ms_to_pil, numpy_to_pil
from .state_dict_utils import (
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_peft,
    convert_unet_state_dict_to_peft,
)
