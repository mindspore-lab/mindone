# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
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
import transformers
from packaging import version

from . import (
    albert,
    aria,
    auto,
    bart,
    bert,
    big_bird,
    bigbird_pegasus,
    bit,
    blip,
    blip_2,
    clap,
    clip,
    dpt,
    fuyu,
    gemma,
    gemma2,
    gemma3,
    glm,
    glpn,
    gpt2,
    granite,
    granitemoe,
    granitemoeshared,
    hiera,
    idefics3,
    ijepa,
    imagegpt,
    levit,
    llama,
    llava,
    megatron_bert,
    minicpm4,
    mistral,
    mixtral,
    mobilebert,
    mpt,
    paligemma,
    persimmon,
    phi,
    phi3,
    qwen2,
    qwen2_5_omni,
    qwen2_5_vl,
    qwen2_audio,
    qwen2_vl,
    roberta,
    siglip,
    speecht5,
    switch_transformers,
    t5,
    umt5,
    vits,
    wav2vec2,
    xlm_roberta,
)

if version.parse(transformers.__version__) >= version.parse("4.51.0"):
    from . import qwen3

if version.parse(transformers.__version__) >= version.parse("4.51.3"):
    from . import glm4

if version.parse(transformers.__version__) >= version.parse("4.53.0"):
    from . import glm4v, vjepa2
