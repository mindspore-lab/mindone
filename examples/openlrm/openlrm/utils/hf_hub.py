# Copyright (c) 2023-2024, Zexin He
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from typing import Dict, Optional, Union

from huggingface_hub import ModelHubMixin, constants, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError

import mindspore as ms
from mindspore import nn

from mindone.safetensors.mindspore import load_file


def wrap_model_hub(model_cls: nn.Cell):
    """
    1. load config.json to LRMModel
    2. load model weights to LRMModel
    """

    class HfModel(model_cls, ModelHubMixin):
        def __init__(self, config: dict):
            super().__init__(**config)
            self.config = config

        @classmethod
        def _from_pretrained(
            cls,
            *,
            model_id: str,
            revision: Optional[str],
            cache_dir: Optional[Union[str, Path]],
            force_download: bool,
            proxies: Optional[Dict],
            resume_download: Optional[bool],
            local_files_only: bool,
            token: Union[str, bool, None],
            strict: bool = True,
            **model_kwargs,
        ):
            use_safetensors = model_kwargs.pop("use_safetensors", None)
            mindspore_dtype = model_kwargs.pop("mindspore_dtype", None)
            ckpt_name = model_kwargs.pop("ckpt_name", None)  # higher priority

            """Load pretrained weights and return the loaded model."""
            model = cls(**model_kwargs)
            if os.path.isdir(model_id) and (
                os.path.isfile(os.path.join(model_id, "ckpt", "train_resume.ckpt"))
                or (ckpt_name is not None and (os.path.isfile(os.path.join(model_id, "ckpt", ckpt_name))))
            ):  # if trained checkpoint, and saved as a local ms.ckpt
                if ckpt_name is not None:
                    model_file = os.path.join(model_id, "ckpt", ckpt_name)
                else:
                    model_file = os.path.join(model_id, "ckpt", "train_resume.ckpt")
                print(f"Loading weights from local pretrained directory: {model_file}")
                state_dict = ms.load_checkpoint(model_file)
            elif os.path.isdir(model_id) and use_safetensors:  # if single safetensors
                model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE)  # "model.safetensors"
                print(f"Loading weights from local directory: {model_file}")
                state_dict = load_file(model_file)

            else:
                try:
                    model_file = hf_hub_download(  # download safetensors
                        repo_id=model_id,
                        filename=constants.SAFETENSORS_SINGLE_FILE,
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                    state_dict = load_file(model_file)
                except EntryNotFoundError:
                    model_file = hf_hub_download(
                        repo_id=model_id,
                        filename=constants.PYTORCH_WEIGHTS_NAME,  # "pytorch_model.bin"
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )

                    if model_file.endswith(".bin"):
                        raise ValueError(f"Fail to load {model_file}, convert it to model.safetensors first. Stopped.")

            # Check loading keys:
            model_state_dict = {k: v for k, v in model.parameters_and_names()}
            state_dict_tmp = {}
            for k, v in state_dict.items():
                if ("norm" in k) and ("mlp" not in k):  # for LayerNorm but not ModLN's mlp
                    k = k.replace(".weight", ".gamma").replace(".bias", ".beta")
                if "lrm_generator." in k:  # training model name
                    k = k.replace("lrm_generator.", "")
                if "adam_" not in k:  # not to load optimizer
                    state_dict_tmp[k] = v
            state_dict = state_dict_tmp
            loaded_keys = list(state_dict.keys())
            expexted_keys = list(model_state_dict.keys())
            original_loaded_keys = loaded_keys
            missing_keys = list(set(expexted_keys) - set(loaded_keys))
            unexpected_keys = list(set(loaded_keys) - set(expexted_keys))
            mismatched_keys = []
            for checkpoint_key in original_loaded_keys:
                if (
                    checkpoint_key in model_state_dict
                    and checkpoint_key in state_dict
                    and state_dict[checkpoint_key].shape != model_state_dict[checkpoint_key].shape
                ):
                    mismatched_keys.append(
                        (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[checkpoint_key].shape)
                    )

            print(
                f"Loading LRMModel...\nmissing_keys: {missing_keys}, \nunexpected_keys: {unexpected_keys}, \nmismatched_keys: {mismatched_keys}"
            )

            print(f"state_dict.dtype {state_dict[loaded_keys[0]].dtype}")  # float32

            # Instantiate the model
            param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict, strict_load=strict)
            print(f"Loaded checkpoint: param_not_load {param_not_load}, ckpt_not_load {ckpt_not_load}")

            if mindspore_dtype is not None:
                model.to(dtype=mindspore_dtype)
                print(f"Use {mindspore_dtype} for LRMModel.")

            return model

    return HfModel
