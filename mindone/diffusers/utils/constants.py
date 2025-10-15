# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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
import os

from huggingface_hub.constants import HF_HOME

from .import_utils import ENV_VARS_TRUE_VALUES

CKPT_FILE_EXTENSION = "ckpt"
CONFIG_NAME = "config.json"
WEIGHTS_NAME = "diffusion_pytorch_model.bin"
WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.bin.index.json"
FLAX_WEIGHTS_NAME = "diffusion_flax_model.msgpack"
ONNX_WEIGHTS_NAME = "model.onnx"
SAFETENSORS_WEIGHTS_NAME = "diffusion_pytorch_model.safetensors"
SAFE_WEIGHTS_INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"
SAFETENSORS_FILE_EXTENSION = "safetensors"
GGUF_FILE_EXTENSION = "gguf"
ONNX_EXTERNAL_WEIGHTS_NAME = "weights.pb"
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
DIFFUSERS_DYNAMIC_MODULE_NAME = "diffusers_modules"
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(HF_HOME, "modules"))
DEPRECATED_REVISION_ARGS = ["fp16", "non-ema"]
DIFFUSERS_REQUEST_TIMEOUT = 60
DIFFUSERS_ATTN_BACKEND = os.getenv("DIFFUSERS_ATTN_BACKEND", "native")
DIFFUSERS_ATTN_CHECKS = os.getenv("DIFFUSERS_ATTN_CHECKS", "0") in ENV_VARS_TRUE_VALUES
DEFAULT_HF_PARALLEL_LOADING_WORKERS = 8
HF_ENABLE_PARALLEL_LOADING = os.environ.get("HF_ENABLE_PARALLEL_LOADING", "").upper() in ENV_VARS_TRUE_VALUES


DECODE_ENDPOINT_SD_V1 = "https://q1bj3bpq6kzilnsu.us-east-1.aws.endpoints.huggingface.cloud/"
DECODE_ENDPOINT_SD_XL = "https://x2dmsqunjd6k9prw.us-east-1.aws.endpoints.huggingface.cloud/"
DECODE_ENDPOINT_FLUX = "https://whhx50ex1aryqvw6.us-east-1.aws.endpoints.huggingface.cloud/"
DECODE_ENDPOINT_HUNYUAN_VIDEO = "https://o7ywnmrahorts457.us-east-1.aws.endpoints.huggingface.cloud/"


ENCODE_ENDPOINT_SD_V1 = "https://qc6479g0aac6qwy9.us-east-1.aws.endpoints.huggingface.cloud/"
ENCODE_ENDPOINT_SD_XL = "https://xjqqhmyn62rog84g.us-east-1.aws.endpoints.huggingface.cloud/"
ENCODE_ENDPOINT_FLUX = "https://ptccx55jz97f9zgo.us-east-1.aws.endpoints.huggingface.cloud/"
