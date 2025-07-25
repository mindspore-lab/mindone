# Copyright 2024 BAAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/baaivision/Emu3 to work with MindSpore.
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

_import_structure = {
    "configuration_emu3": ["Emu3Config"],
    "tokenization_emu3": ["Emu3Tokenizer"],
    "processing_emu3": ["Emu3Processor"],
}

_import_structure["modeling_emu3"] = [
    "Emu3Model",
    "Emu3PretrainedModel",
    "Emu3ForCausalLM",
]

if TYPE_CHECKING:
    from .configuration_emu3 import Emu3Config
    from .modeling_emu3 import Emu3ForCausalLM, Emu3Model, Emu3PretrainedModel
    from .processing_emu3 import Emu3Processor
    from .tokenization_emu3 import Emu3Tokenizer

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
