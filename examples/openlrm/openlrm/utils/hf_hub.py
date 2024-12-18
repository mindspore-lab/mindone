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
import mindspore as ms
from mindspore import nn
from huggingface_hub import ModelHubMixin, constants, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import validate_hf_hub_args
import inspect
from typing import Dict, Optional, Union
from pathlib import Path
from mindone.safetensors.mindspore import load_file


def wrap_model_hub(model_cls: nn.Cell):

    '''
        1. load config.json to LRMModel
        2. load model weights to LRMModel
    '''
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
            use_safetensors = kwargs.pop("use_safetensors", None)
            mindspore_dtype = kwargs.pop("mindspore_dtype", None)

            """Load pretrained weights and return the loaded model."""
            model = cls(**model_kwargs)
            if model_id.endswith(".ckpt"):  # if convereted to a local ms.ckpt
                state_dict = ms.load_checkpoint(model_id)
            elif os.path.isdir(model_id) and use_safetensors: # if single safetensors
                print("Loading weights from local directory")
                model_file = os.path.join(model_id, constants.SAFETENSORS_SINGLE_FILE) # "model.safetensors"
                state_dict = load_file(ckpt_path)
                
            else:
                try:
                    model_file = hf_hub_download( # download safetensors
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
                except EntryNotFoundError:
                    model_file = hf_hub_download(
                        repo_id=model_id,
                        filename=constants.PYTORCH_WEIGHTS_NAME, #"pytorch_model.bin"
                        revision=revision,
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        resume_download=resume_download,
                        token=token,
                        local_files_only=local_files_only,
                    )
                        
                    if model_file.endswith(".bin"):
                        new_model_file = model_file
                        print("Fail to load {model_file}, convert it to model.safetensors first. Stopped.")
                        exit()

            
            # Check loading keys:
            model_state_dict = {k: v for k, v in model.parameters_and_names()}
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

            print(f"state_dict.dtype {state_dict[loaded_keys[0]].dtype}")  # float16
            print(f"model.dtype {model.dtype}")
            if state_dict[loaded_keys[0]].dtype != model.dtype:
                model = model.to(state_dict[loaded_keys[0]].dtype)
            print(f"Use {model.dtype} for LRMModel.")

            # Instantiate the model
            param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict, strict_load=strict)
            print(f"Loaded checkpoint: param_not_load {param_not_load}, ckpt_not_load {ckpt_not_load}")

            
            # Save where the model was instantiated from
            # model.register_to_config(_name_or_path=pretrained_model_name_or_path)

            if mindspore_dtype is not None:
                model.to(dtype=mindspore_dtype)


        @classmethod
        def _get_signature_keys(cls, obj):
            parameters = inspect.signature(obj.__init__).parameters
            required_parameters = {k: v for k, v in parameters.items() if v.default == inspect._empty}
            optional_parameters = set({k for k, v in parameters.items() if v.default != inspect._empty})
            expected_modules = set(required_parameters.keys()) - {"self"}

            optional_names = list(optional_parameters)
            for name in optional_names:
                if name in cls._optional_components:
                    expected_modules.add(name)
                    optional_parameters.remove(name)

            return expected_modules, optional_parameters

        def to(self, dtype):
            module_names, _ = self._get_signature_keys(self)
            modules = [getattr(self, n, None) for n in module_names]
            modules = [m for m in modules if isinstance(m, nn.Cell)]
            for module in modules:
                module.to(dtype)
            return self

    return HfModel
