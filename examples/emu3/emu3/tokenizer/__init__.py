# Copyright 2024 BAAI and the HuggingFace Inc. team. All rights reserved.
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

from transformers.utils import _LazyModule

_import_structure = {"configuration_emu3visionvq": ["Emu3VisionVQConfig"]}
_import_structure["modeling_emu3visionvq"] = [
    "Emu3VisionVQModel",
    "Emu3VisionVQPretrainedModel",
]
_import_structure["image_processing_emu3visionvq"] = ["Emu3VisionVQImageProcessor"]

if TYPE_CHECKING:
    from .configuration_emu3visionvq import Emu3VisionVQConfig
    from .image_processing_emu3visionvq import Emu3VisionVQImageProcessor
    from .modeling_emu3visionvq import Emu3VisionVQModel, Emu3VisionVQPretrainedModel
else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
