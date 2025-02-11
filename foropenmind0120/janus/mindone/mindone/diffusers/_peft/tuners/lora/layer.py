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
from __future__ import annotations

import math
import warnings
from typing import Any, List, Optional, Union

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import HeUniform, Normal, Zero, initializer

from ...utils.other import transpose
from ..tuners_utils import BaseTunerLayer, check_adapters_to_merge
from .config import LoraConfig


def normal_(tensor: Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    tensor.set_data(initializer(Normal(std, mean), tensor.shape, tensor.dtype))


def zeros_(tensor: Tensor) -> None:
    tensor.set_data(initializer(Zero(), tensor.shape, tensor.dtype))


def kaiming_uniform_(tensor: Tensor, a=0.0, mode="fan_in", nonlinearity="leaky_relu") -> None:
    tensor.set_data(initializer(HeUniform(a, mode, nonlinearity), tensor.shape, tensor.dtype))


class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Cell, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.CellDict({})
        self.lora_A = nn.CellDict({})
        self.lora_B = nn.CellDict({})
        # TODO: LoRA Embedding should be a ParameterDict. Add it here if we support Embedding.
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Dense):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.vocab_size, base_layer.embedding_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.CellDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = nn.Dense(self.in_features, r, has_bias=False)
        self.lora_B[adapter_name] = nn.Dense(r, self.out_features, has_bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.is_floating_point() or weight.is_complex():
                    for p in self.get_parameters():
                        p.set_dtype(weight.dtype)
                else:
                    pass
                break
        self.set_adapter(self.active_adapters)

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return

        if adapter_name in self.lora_A.keys():
            if init_lora_weights is True:
                # initialize A the same way as the default for nn.Linear and B to zero
                # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
                kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            elif init_lora_weights.lower() == "gaussian":
                normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
            else:
                raise ValueError(f"Unknown initialization {init_lora_weights=}")
            zeros_(self.lora_B[adapter_name].weight)

    def loftq_init(self, adapter_name):
        raise NotImplementedError("loftq_init is not yet supported.")

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class Linear(nn.Cell, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not ops.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    ops.assign(base_layer.weight, orig_weights)
                else:
                    ops.assign_add(base_layer.weight, self.get_delta_weight(active_adapter))
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                ops.assign_sub(self.get_base_layer().weight, self.get_delta_weight(active_adapter))

    def get_delta_weight(self, adapter) -> ms.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]
        return output_tensor

    def construct(self, x: ms.Tensor, *args: Any, **kwargs: Any) -> ms.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                raise RuntimeError(
                    f"{self} has disabled adapters, but the adapters are already merged. "
                    f"Please manually call `unmerge()` before invoke forwarding."
                )
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(nn.Cell, LoraLayer):
    # Lora implemented in a conv2d layer
    def __init__(
        self,
        base_layer: nn.Cell,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        pad_mode = base_layer.pad_mode
        padding = base_layer.padding
        self.lora_A[adapter_name] = nn.Conv2d(
            self.in_features, r, kernel_size, stride, pad_mode, padding, has_bias=False
        )
        self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), has_bias=False)
        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        weight = getattr(base_layer, "weight", None)
        if weight is not None:
            # the layer is already completely initialized, this is an update
            for p in self.get_parameters():
                p.set_dtype(weight.dtype)
        self.set_adapter(self.active_adapters)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        adapter_names = check_adapters_to_merge(self, adapter_names)
        if not adapter_names:
            # no adapter to merge
            return

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not ops.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    ops.assign(base_layer.weight, orig_weights)
                else:
                    ops.assign_add(base_layer.weight, self.get_delta_weight(active_adapter))
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                ops.assign_sub(self.get_base_layer().weight, self.get_delta_weight(active_adapter))

    def get_delta_weight(self, adapter) -> ms.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        # https://github.com/bmaltais/kohya_ss/blob/feb6728762a8f463d15ba936d189d4c3abfaa1ab/networks/lora.py#L117
        if self.get_base_layer().weight.shape[2:4] == (1, 1):
            # conv2d 1x1
            output_tensor = (weight_B.squeeze(3).squeeze(2) @ weight_A.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(
                3
            ) * self.scaling[adapter]
        else:
            # conv2d 3x3
            output_tensor = (
                ops.conv2d(
                    weight_A.permute(1, 0, 2, 3),
                    weight_B,
                ).permute(1, 0, 2, 3)
                * self.scaling[adapter]
            )
        return output_tensor

    def construct(self, x: ms.Tensor, *args, **kwargs) -> ms.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                raise RuntimeError(
                    f"{self} has disabled adapters, but the adapters are already merged. "
                    f"Please manually call `unmerge()` before invoke forwarding."
                )
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)
                result += lora_B(lora_A(dropout(x))) * scaling

        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


def dispatch_default(
    target: nn.Cell,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[nn.Cell]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, nn.Embedding):
        raise NotImplementedError("Embedding layer is not yet supported.")
    elif isinstance(target_base_layer, nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Dense):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `mindspore.nn.Dense`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)

    return new_module
