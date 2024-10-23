# Copyright 2020 The HuggingFace Team. All rights reserved.
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

from ...utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_sentencepiece_available,
    is_tokenizers_available,
)


_import_structure = {"configuration_t5": ["T5_PRETRAINED_CONFIG_ARCHIVE_MAP", "T5Config", "T5OnnxConfig"]}

try:
    if not is_sentencepiece_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_t5"] = ["T5Tokenizer"]

try:
    if not is_tokenizers_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tokenization_t5_fast"] = ["T5TokenizerFast"]

_import_structure["modeling_t5"] = [
    "T5_PRETRAINED_MODEL_ARCHIVE_LIST",
    "T5EncoderModel",
    "T5ForConditionalGeneration",
    "T5Model",
    "T5PreTrainedModel",
    "load_tf_weights_in_t5",
]

if TYPE_CHECKING:
    from .configuration_t5 import T5_PRETRAINED_CONFIG_ARCHIVE_MAP, T5Config, T5OnnxConfig

    try:
        if not is_sentencepiece_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_t5 import T5Tokenizer

    try:
        if not is_tokenizers_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tokenization_t5_fast import T5TokenizerFast

    from .modeling_t5 import (
        T5_PRETRAINED_MODEL_ARCHIVE_LIST,
        T5EncoderModel,
        T5ForConditionalGeneration,
        T5Model,
        T5PreTrainedModel,
        load_tf_weights_in_t5,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
