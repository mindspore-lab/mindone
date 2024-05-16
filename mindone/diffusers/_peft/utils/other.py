# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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
import copy
from typing import List, Optional

import mindspore as ms
from mindspore import nn


class ModulesToSaveWrapper(nn.Cell):
    def __init__(self, module_to_save, adapter_name):
        super().__init__()
        self.original_module = module_to_save
        self.modules_to_save = nn.CellDict({})
        self._active_adapter = adapter_name
        self._disable_adapters = False
        self.update(adapter_name)

    @property
    def disable_adapters(self) -> bool:
        # use a property to ensure that disable_adapters is not set directly, instead use the enable_adapters method
        return self._disable_adapters

    @property
    def active_adapter(self) -> str:
        # use a property to ensure that active_adapter is not set directly, instead use the set_adapter method
        return self._active_adapter

    @property
    def weight(self):
        if self.active_adapter not in self.modules_to_save:
            return self.original_module.weight
        return self.modules_to_save[self.active_adapter].weight

    def update(self, adapter_name):
        # TODO: deepcopy might not work on nn.Cell
        self.modules_to_save.update(nn.CellDict({adapter_name: copy.deepcopy(self.original_module)}))
        # self.original_module.requires_grad_(False)
        for p in self.original_module.get_parameters():
            p.requires_grad = False
        if adapter_name == self.active_adapter:
            # self.modules_to_save[adapter_name].requires_grad_(True)
            for p in self.modules_to_save[adapter_name].get_parameters():
                p.requires_grad = True

    def construct(self, *args, **kwargs):
        if self.disable_adapters or (self.active_adapter not in self.modules_to_save):
            return self.original_module(*args, **kwargs)
        return self.modules_to_save[self.active_adapter](*args, **kwargs)

    def enable_adapters(self, enabled: bool):
        """Toggle the enabling and disabling of adapters

        Takes care of setting the requires_grad flag for the adapter weights.

        Args:
            enabled (bool): True to enable adapters, False to disable adapters
        """
        if self._disable_adapters is not enabled:
            # already in the desired state, do nothing
            return

        if enabled:
            # self.original_module.requires_grad_(False)
            for p in self.original_module.get_parameters():
                p.requires_grad = False
            # self.modules_to_save[self.active_adapter].requires_grad_(True)
            for p in self.modules_to_save[self.active_adapter].get_parameters():
                p.requires_grad = True
            self._disable_adapters = False
        else:
            # self.original_module.requires_grad_(True)
            for p in self.original_module.get_parameters():
                p.requires_grad = True
            # self.modules_to_save.requires_grad_(False)
            for p in self.modules_to_save.get_parameters():
                p.requires_grad = False
            self._disable_adapters = True

    def set_adapter(self, adapter_name: str):
        """Set the active adapter

        Args:
            adapter_name (str): The name of the adapter to set as active
        """
        if adapter_name not in self.modules_to_save:
            raise ValueError(f"Adapter {adapter_name} not found in {self.modules_to_save.keys()}")

        # self.modules_to_save[self.active_adapter].requires_grad_(False)
        for p in self.modules_to_save[self.active_adapter].get_parameters():
            p.requires_grad = False
        # self.modules_to_save[adapter_name].requires_grad_(True)
        for p in self.modules_to_save[adapter_name].get_parameters():
            p.requires_grad = True
        self._active_adapter = adapter_name


def _get_subcell(mod: nn.Cell, target: str) -> "nn.Cell":
    """See https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.get_submodule"""
    if target == "":
        return mod

    atoms: List[str] = target.split(".")
    for item in atoms:
        if not hasattr(mod, item):
            raise AttributeError(f"{mod.__class__.__name__} has no attribute `{item}`")
        mod = getattr(mod, item)
        if not isinstance(mod, nn.Cell):
            raise AttributeError(f"`{item}` is not an nn.Cell")
    return mod


def _get_submodules(model, key):
    parent = _get_subcell(model, ".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = _get_subcell(model, key)
    return parent, target, target_name


def _freeze_adapter(model, adapter_name):
    for n, p in model.parameters_and_names():
        if adapter_name in n:
            p.requires_grad = False


def _set_trainable(model, adapter_name):
    key_list = [key for key, _ in model.cells_and_names()]
    for key in key_list:
        target_module_found = any(key.endswith(target_key) for target_key in model.modules_to_save)
        if target_module_found:
            parent, target, target_name = _get_submodules(model, key)
            if isinstance(target, ModulesToSaveWrapper):
                target.update(adapter_name)
                target.set_adapter(target.active_adapter)
            else:
                new_module = ModulesToSaveWrapper(target, adapter_name)
                new_module.set_adapter(adapter_name)
                setattr(parent, target_name, new_module)


def _set_adapter(model, adapter_name):
    def check_adapter_name(adapter_name):
        if isinstance(adapter_name, str):
            return adapter_name

        # adapter_name is a list of str
        if len(adapter_name) > 1:
            raise ValueError("Only one adapter can be set at a time for modules_to_save")
        elif len(adapter_name) == 0:
            raise ValueError("Please specify at least one adapter to set")
        adapter_name = adapter_name[0]
        return adapter_name

    for _, module in model.cells_and_names():
        if isinstance(module, ModulesToSaveWrapper):
            # only check the adapter_name if we actually encounter a ModulesToSaveWrapper, otherwise we don't care
            adapter_name = check_adapter_name(adapter_name)
            module.set_adapter(adapter_name)


def _prepare_prompt_learning_config(peft_config, model_config):
    if peft_config.num_layers is None:
        if "num_hidden_layers" in model_config:
            num_layers = model_config["num_hidden_layers"]
        elif "num_layers" in model_config:
            num_layers = model_config["num_layers"]
        elif "n_layer" in model_config:
            num_layers = model_config["n_layer"]
        else:
            raise ValueError("Please specify `num_layers` in `peft_config`")
        peft_config.num_layers = num_layers

    if peft_config.token_dim is None:
        if "hidden_size" in model_config:
            token_dim = model_config["hidden_size"]
        elif "n_embd" in model_config:
            token_dim = model_config["n_embd"]
        elif "d_model" in model_config:
            token_dim = model_config["d_model"]
        else:
            raise ValueError("Please specify `token_dim` in `peft_config`")
        peft_config.token_dim = token_dim

    if peft_config.num_attention_heads is None:
        if "num_attention_heads" in model_config:
            num_attention_heads = model_config["num_attention_heads"]
        elif "n_head" in model_config:
            num_attention_heads = model_config["n_head"]
        elif "num_heads" in model_config:
            num_attention_heads = model_config["num_heads"]
        elif "encoder_attention_heads" in model_config:
            num_attention_heads = model_config["encoder_attention_heads"]
        else:
            raise ValueError("Please specify `num_attention_heads` in `peft_config`")
        peft_config.num_attention_heads = num_attention_heads

    if getattr(peft_config, "encoder_hidden_size", None) is None:
        setattr(peft_config, "encoder_hidden_size", peft_config.token_dim)

    return peft_config


def transpose(weight, fan_in_fan_out):
    if not fan_in_fan_out:
        return weight

    if isinstance(weight, ms.Parameter):
        return ms.Parameter(weight.T)
    return weight.T


def _is_valid_match(key: str, target_key: str):
    """
    Helper function to match module names target_key and key. Makes sure that either the key is exactly the target_key
    or the target_key is a submodule of key
    """
    if key.endswith(target_key):
        if len(key) > len(target_key):
            return key.endswith("." + target_key)  # must be a sub module
        return True
    return False


def _get_batch_size(input_ids: Optional[ms.Tensor], inputs_embeds: Optional[ms.Tensor]) -> int:
    """Get the batch size based on either input_ids or input_embeds

    Raises an ValueError if both are None.

    """
    if (input_ids is None) and (inputs_embeds is None):
        raise ValueError("You have to provide either input_ids or inputs_embeds")

    if input_ids is not None:
        batch_size = input_ids.shape[0]
    else:
        batch_size = inputs_embeds.shape[0]
    return batch_size
