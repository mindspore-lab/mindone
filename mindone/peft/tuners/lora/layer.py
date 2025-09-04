# Copyright 2023-present the HuggingFace Inc. team.
#
# This code is adapted from https://github.com/huggingface/peft
# with modifications to run peft on mindspore.
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
from typing import Any, Iterable, Optional, Tuple, Union

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint, nn, ops
from mindspore.common.initializer import HeUniform, Normal, Zero, initializer

from mindone.peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from mindone.peft.utils.other import transpose
from mindone.transformers.mindspore_utils import Conv1D

from .config import LoraConfig


def normal_(tensor: ms.Tensor, mean: float = 0.0, std: float = 1.0) -> None:
    tensor.set_data(initializer(Normal(std, mean), tensor.shape, tensor.dtype))


def zeros_(tensor: ms.Tensor) -> None:
    tensor.set_data(initializer(Zero(), tensor.shape, tensor.dtype))


def kaiming_uniform_(tensor: ms.Tensor, a=0.0, mode="fan_in", nonlinearity="leaky_relu") -> None:
    tensor.set_data(initializer(HeUniform(a, mode, nonlinearity), tensor.shape, tensor.dtype))


class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Cell, ephemeral_gpu_offload: bool = False, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.CellDict({})
        self.lora_A = nn.CellDict({})
        self.lora_B = nn.CellDict({})
        # For Embedding layer
        # TODO: LoRA Embedding should be a ParameterDict. Add it here if we support Embedding.
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_bias: dict[str, bool] = {}
        self.lora_magnitude_vector = nn.CellDict()  # for DoRA
        self._caches: dict[str, Any] = {}
        self.ephemeral_gpu_offload: bool = ephemeral_gpu_offload
        # flag to enable/disable casting of input to weight dtype during forward call
        self.cast_input_dtype_enabled: bool = True
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        # mint nn layers
        if isinstance(base_layer, mint.nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, mint.nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, mint.nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        # classic nn layers
        elif isinstance(base_layer, nn.Dense):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv1d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Conv3d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif isinstance(base_layer, nn.MultiheadAttention):
            if not base_layer._qkv_same_embed_dim:
                raise ValueError(f"Only same dim for query/key/value is supported as of now for {self.__class__}.")
            in_features, out_features = base_layer.embed_dim, 3 * base_layer.embed_dim
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        elif hasattr(base_layer, "codebooks") and base_layer.__class__.__name__ == "QuantizedLinear":
            # AQLM QuantLinear
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "w_bit") and base_layer.__class__.__name__ == "WQLinear_GEMM":
            # Awq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif base_layer.__class__.__name__ == "EetqLinear":
            # Eetq layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif hasattr(base_layer, "W_q") and base_layer.__class__.__name__ == "HQQLinear":
            # HQQ layers
            in_features, out_features = base_layer.in_features, base_layer.out_features
        else:
            # possibly support user provided custom layer types using dynamic dispatch
            if hasattr(base_layer, "in_features") and hasattr(base_layer, "out_features"):
                in_features, out_features = base_layer.in_features, base_layer.out_features
            else:
                in_features, out_features = None, None
            warnings.warn(
                f"Unsupported layer type '{type(base_layer)}' encountered, proceed at your own risk.", UserWarning
            )

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(
        self,
        adapter_name,
        r,
        lora_alpha,
        lora_dropout,
        init_lora_weights,
        use_rslora,
        use_dora: bool = False,
        lora_bias: bool = False,
    ):
        # This code works for linear layers, override for other layer types
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = mint.nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.CellDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        self.lora_A[adapter_name] = mint.nn.Linear(self.in_features, r, bias=False)
        self.lora_B[adapter_name] = mint.nn.Linear(r, self.out_features, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # for inits that require access to the base weight, use gather_param_ctx so that the weight is gathered when using DeepSpeed
        if isinstance(init_lora_weights, str) and init_lora_weights.startswith("pissa"):
            self.pissa_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.startswith("corda"):
            self.corda_init(adapter_name, init_lora_weights)
        elif isinstance(init_lora_weights, str) and init_lora_weights.lower() == "olora":
            self.olora_init(adapter_name)
        elif init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights == "eva":
            zeros_(self.lora_B[adapter_name].weight)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

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
            if self.lora_bias[adapter_name]:
                zeros_(self.lora_B[adapter_name].bias)

    def olora_init(self, adapter_name):
        base_layer = self.get_base_layer()
        orig_weight = base_layer.weight
        dtype = orig_weight.dtype

        if dtype in [ms.float32, ms.float16, ms.bfloat16]:
            weight_tensor = orig_weight
        else:
            raise TypeError(f"Unsupported data type for the base layer. Got {dtype}.")

        scale_factor = self.scaling[adapter_name]
        r = self.r[adapter_name]
        weight_tensor = weight_tensor.to(ms.float32)
        Q, R = mint.linalg.qr(weight_tensor)

        Qr, Rr = Q[:, :r], R[:r]

        self.lora_A[adapter_name].weight = Rr.contiguous()
        self.lora_B[adapter_name].weight = Qr.contiguous()

        weight_tensor -= scale_factor * self.lora_B[adapter_name].weight @ self.lora_A[adapter_name].weight
        weight_tensor = weight_tensor.to(dtype)
        base_layer.weight = weight_tensor

    def pissa_init(self, adapter_name, init_lora_weights):
        weight = self.get_base_layer().weight
        dtype = weight.dtype
        if dtype not in [ms.float32, ms.float16, ms.bfloat16]:
            raise TypeError(
                "Please initialize PiSSA under float32, float16, or bfloat16. "
                "Subsequently, re-quantize the residual model to help minimize quantization errors."
            )
        weight = transpose(weight.to(ms.float32), self.fan_in_fan_out)
        if init_lora_weights == "pissa":
            # USV^T = W <-> VSU^T = W^T, where W^T = weight.data in R^{out_channel, in_channel},
            S, Uh, V = ops.svd(weight, full_matrices=False)
            V = V.T
            Vr = V[:, : self.r[adapter_name]]
            Sr = S[: self.r[adapter_name]]
            Sr /= self.scaling[adapter_name]
            Uhr = Uh[: self.r[adapter_name]]
        elif len(init_lora_weights.split("_niter_")) == 2:
            raise NotImplementedError("`svd_lowrank` is not implemented for MindSpore yet.")
        else:
            raise ValueError(
                f"init_lora_weights should be 'pissa' or 'pissa_niter_[number of iters]', got {init_lora_weights} instead."
            )

        lora_A = mint.diag(mint.sqrt(Sr)) @ Uhr
        lora_B = Vr @ mint.diag(mint.sqrt(Sr))
        self.lora_A[adapter_name].weight = lora_A
        self.lora_B[adapter_name].weight = lora_B
        weight = weight - self.scaling[adapter_name] * lora_B @ lora_A
        weight = transpose(weight.to(dtype), self.fan_in_fan_out)
        self.get_base_layer().weight = weight

    def corda_init(self, adapter_name, init_lora_weights):
        raise NotImplementedError

    def loftq_init(self, adapter_name):
        raise NotImplementedError

    def dora_init(self, adapter_name: str) -> None:
        raise NotImplementedError

    def _cache_store(self, key: str, value: Any) -> None:
        self._caches[key] = value

    def _cache_pop(self, key: str) -> Any:
        value = self._caches.pop(key)
        return value

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

    def peft_parameters_and_names(self, name_prefix: str = "") -> Iterable[Tuple[str, ms.Parameter]]:
        assert isinstance(self, nn.Cell), f"{self.__class__.__name__} is not a nn.Cell"

        for par_name, par in self.parameters_and_names(name_prefix=name_prefix):
            if any(name in par_name for name in self.adapter_layer_names):
                yield par_name, par

    def _check_forward_args(self, x, *args, **kwargs):
        """Check if the arguments are compatible with the configs and state of the model"""
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is None:
            return

        if len(x) != len(adapter_names):
            msg = (
                "Length of `adapter_names` should be the same as the number of inputs, but got "
                f"{len(adapter_names)} and {len(x)} respectively."
            )
            raise ValueError(msg)

        if self.merged:
            # It is unclear what would be the right thing to do if users pass adapter_names and there are merged
            # adapters. Therefore, it is better to raise an error in this case.
            msg = "Cannot pass `adapter_names` when there are merged adapters, please call `unmerge_adapter` first."
            raise ValueError(msg)

        # DoRA is not supported (yet), check that it's not being used. Don't check "__base__", as this is the
        # placeholder for the base model.
        unique_adapters = {name for name in adapter_names if name != "__base__"}
        for adapter_name in unique_adapters:
            if self.use_dora.get(adapter_name, False):
                msg = "Cannot pass `adapter_names` when DoRA is enabled."
                raise ValueError(msg)

    def _mixed_batch_forward(self, x: ms.Tensor, *args: Any, adapter_names: list[str], **kwargs: Any) -> ms.Tensor:
        # This is a special method that handles the case when users pass the argument `adapter_names`. This is an
        # extra argument that allows mixing different adapters in the same batch at inference time.
        result = self.base_layer(x, *args, **kwargs)
        mindspore_result_dtype = result.dtype

        unique_adapters = set(adapter_names)
        sub_batch_indices_list = []
        for adapter in unique_adapters:
            sub_batch_indices_list.append([index for index, item in enumerate(adapter_names) if item == adapter])

        for i, active_adapter in enumerate(unique_adapters):
            if active_adapter == "__base__":
                continue
            if active_adapter not in self.lora_A.keys():
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]

            # getting the sub-batch, passing it to LoRA layers and updating the corresponding indices of the linear
            # layer output
            sub_batch = x[sub_batch_indices_list[i]].to(lora_A.weight.dtype)
            lora_output = lora_B(lora_A(dropout(sub_batch))) * scaling
            result[sub_batch_indices_list[i]] += lora_output.to(mindspore_result_dtype)

        return result

    def _cast_input_dtype(self, x: ms.Tensor, dtype: ms.Type) -> ms.Tensor:
        """
        Whether to cast the dtype of the input to the forward method.

        Usually, we want to enable this to align the input dtype with the dtype of the weight, but by setting
        layer.cast_input_dtype=False, this can be disabled if necessary.

        Enabling or disabling can be managed via the peft.helpers.disable_lora_input_dtype_casting context manager.
        """
        if (not self.cast_input_dtype_enabled) or (x.dtype == dtype):
            return x
        return x.to(dtype=dtype)


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
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
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
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self.lora_magnitude_vector[active_adapter].get_weight_norm(
                            orig_weights, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        orig_weights = dora_factor * (orig_weights + delta_weight)

                    if not mint.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not mint.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self.lora_magnitude_vector[active_adapter].get_weight_norm(
                            base_layer.weight, transpose(delta_weight, self.fan_in_fan_out), scaling=1
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        dora_factor = transpose(dora_factor.view(-1, 1), self.fan_in_fan_out)
                        new_weight = dora_factor * (base_layer.weight + delta_weight)
                        base_layer.weight = new_weight

                    if self.lora_bias[active_adapter]:
                        base_layer.bias += self.lora_B[active_adapter].bias

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
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight / dora_factor.view(-1, 1) - delta_weight
                    weight = weight_orig

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias -= self.lora_B[active_adapter].bias

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
        # Equivalent to: adapter_names = kwargs.pop("adapter_names", None)
        # We do so since kwargs.pop() is not supported in MindSpore GRAPH MODE.
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is not None:
            copied_kwargs = {}
            for k, v in kwargs.items():
                if k == "adapter_names":
                    continue
                copied_kwargs[k] = v
            kwargs = copied_kwargs

        if self.disable_adapters:
            if self.merged:
                raise RuntimeError(
                    f"{self} has disabled adapters, but the adapters are already merged. "
                    f"Please manually call `unmerge()` before invoke forwarding."
                )
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            mindspore_result_dtype = result.dtype

            lora_A_keys = self.lora_A.keys()
            for active_adapter in self.active_adapters:
                if active_adapter not in lora_A_keys:
                    continue

                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.to(mindspore_result_dtype)

        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class _ConvNd(nn.Cell, LoraLayer):
    # Lora implemented in a conv(2,3)d layer
    def __init__(
        self,
        base_layer: nn.Cell,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        lora_bias: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        LoraLayer.__init__(self, base_layer)

        self._active_adapter = adapter_name
        self._kernel_dim = base_layer.weight.dim()

        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
            lora_bias=lora_bias,
        )

    def update_layer(
        self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora, lora_bias
    ):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = mint.nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer
        # Actual trainable parameters
        base_layer = self.get_base_layer()
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        conv_layer = type(base_layer)
        out_kernel = out_stride = (1,) * (self._kernel_dim - 2)
        self.lora_A[adapter_name] = conv_layer(self.in_features, r, kernel_size, stride, padding, bias=False)
        self.lora_B[adapter_name] = conv_layer(r, self.out_features, out_kernel, out_stride, bias=lora_bias)
        self.lora_bias[adapter_name] = lora_bias

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def _get_dora_factor_view(self):
        return (-1,) + (1,) * (self._kernel_dim - 1)

    def dora_init(self, adapter_name: str) -> None:
        raise NotImplementedError

    def _get_dora_layer_class(self):
        # Subclasses should override this method to return the appropriate DoraLayer class
        raise NotImplementedError

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
        """
        Merge the active adapter weights inside the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`list[str]`, *optional*):
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
                    delta_weight = self.get_delta_weight(active_adapter)

                    if not self.use_dora[active_adapter]:
                        orig_weights += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self.lora_magnitude_vector[active_adapter].get_weight_norm(
                            orig_weights, delta_weight, scaling=1
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        orig_weights = dora_factor.view(*self._get_dora_factor_view()) * (orig_weights + delta_weight)

                    if not mint.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )
                    base_layer.weight = orig_weights

                    if self.lora_bias[active_adapter]:
                        new_bias = base_layer.bias + self.lora_B[active_adapter].bias
                        if not mint.isfinite(new_bias).all():
                            raise ValueError(
                                f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                            )
                        base_layer.bias = new_bias

                else:
                    delta_weight = self.get_delta_weight(active_adapter)
                    if not self.use_dora[active_adapter]:
                        base_layer.weight += delta_weight
                    else:
                        # handle dora
                        # since delta_weight already includes scaling, set it to 1 here
                        weight_norm = self.lora_magnitude_vector[active_adapter].get_weight_norm(
                            base_layer.weight, delta_weight, scaling=1
                        )
                        # We need to cache weight_norm because it has to be based on the original weights. We
                        # cannot calculate it on the fly based on the merged weights when unmerging because its a
                        # different value
                        self._cache_store(f"{active_adapter}-weight_norm", weight_norm)
                        dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                        new_weight = dora_factor.view(*self._get_dora_factor_view()) * (
                            base_layer.weight + delta_weight
                        )
                        base_layer.weight = new_weight

                    if self.lora_bias[active_adapter]:
                        base_layer.bias += self.lora_B[active_adapter].bias

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
                weight = self.get_base_layer().weight
                delta_weight = self.get_delta_weight(active_adapter)
                if not self.use_dora[active_adapter]:
                    weight -= delta_weight
                else:
                    weight_norm = self._cache_pop(f"{active_adapter}-weight_norm")
                    dora_factor = self.lora_magnitude_vector[active_adapter].weight / weight_norm
                    weight_orig = weight / dora_factor.view(*self._get_dora_factor_view()) - delta_weight
                    weight = weight_orig

                if self.lora_bias[active_adapter]:
                    self.get_base_layer().bias -= self.lora_B[active_adapter].bias

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
            output_tensor = (
                self.conv_fn(
                    weight_A.swapaxes(0, 1),
                    weight_B,
                ).swapaxes(0, 1)
                * self.scaling[adapter]
            )
        return output_tensor

    def construct(self, x: ms.Tensor, *args, **kwargs) -> ms.Tensor:
        # Equivalent to: adapter_names = kwargs.pop("adapter_names", None)
        # We do so since kwargs.pop() is not supported in MindSpore GRAPH MODE.
        adapter_names = kwargs.get("adapter_names", None)
        if adapter_names is not None:
            copied_kwargs = {}
            for k, v in kwargs.items():
                if k == "adapter_names":
                    continue
                copied_kwargs[k] = v
            kwargs = copied_kwargs

        if self.disable_adapters:
            if self.merged:
                raise RuntimeError(
                    f"{self} has disabled adapters, but the adapters are already merged. "
                    f"Please manually call `unmerge()` before invoke forwarding."
                )
            result = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)

        else:
            result = self.base_layer(x, *args, **kwargs)
            mindspore_result_dtype = result.dtype

            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                lora_B = self.lora_B[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = self._cast_input_dtype(x, lora_A.weight.dtype)

                if not self.use_dora[active_adapter]:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
                else:
                    if isinstance(dropout, nn.Identity) or not self.training:
                        base_result = result
                    else:
                        x = dropout(x)
                        base_result = None

                    result = result + self.lora_magnitude_vector[active_adapter](
                        x,
                        lora_A=lora_A,
                        lora_B=lora_B,
                        scaling=scaling,
                        base_layer=self.get_base_layer(),
                        base_result=base_result,
                    )

            result = result.to(mindspore_result_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep


class Conv2d(_ConvNd):
    # Lora implemented in a conv2d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 4:
            raise ValueError(f"Conv2d layer kernel must have 4 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv2d

    def _get_dora_layer_class(self):
        raise NotImplementedError


class Conv1d(_ConvNd):
    # Lora implemented in a conv1d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 3:
            raise ValueError(f"Conv1d layer kernel must have 3 dimensions, not {self._kernel_dim}")
        self.conv_fn = ops.conv1d  # TODO: replace by mint-api once supported

    def _get_dora_layer_class(self):
        raise NotImplementedError


class Conv3d(_ConvNd):
    # Lora implemented in a conv3d layer
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._kernel_dim == 5:
            raise ValueError(f"Conv3d layer kernel must have 5 dimensions, not {self._kernel_dim}")
        self.conv_fn = F.conv3d

    def _get_dora_layer_class(self):
        raise NotImplementedError


class MultiheadAttention(nn.Cell, LoraLayer):
    """LoRA implemented in a multihead attention layer

    This is currently only implemented for the case of `_qkv_same_embed_dim = True`, i.e. query, key, and value having
    the same dimension.

    Note: LoRA is applied to both the in_proj (query/key/value) and out_proj. There is currently no way to specify only
    one of them. Don't try to apply LoRA to the out_proj of MultiheadAttention by targeting that layer specifically,
    since the forward method of that layer is not being used, hence the LoRA adapter would be ignored.

    This is a little bit hacky because of the way that MultiheadAttention is implemented in PyTorch: There are no
    `nn.Linear` layers which we can hook onto or, in case of output projection, `.forward` is not used. This
    implementation works around these problems by merging the weights before the forward call and unmerging them after
    the forward call.
    """

    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ) -> None:
        # TODO work with separate weights
        if not getattr(base_layer, "_qkv_same_embed_dim", True):
            # default for this value appears to be True:
            # https://github.com/pytorch/pytorch/blob/701ba5203fe68d55d655bd4d6c008be94cf34ea5/torch/nn/modules/activation.py#L1128-L1130
            raise ValueError(
                f"Only same embed for query/key/value is supported as of now for {self.__class__.__name__}."
            )
        if use_dora:
            # TODO: probably not so hard to implement
            raise ValueError(f"{self.__class__.__name__} does not support DoRA (yet), please set use_dora to False")

        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)

        # Note: LoRA is applied to both in_proj and out_proj. There is currently no way to only specify one of them.
        if isinstance(base_layer.out_proj, (mint.nn.Linear, nn.Dense)):
            self.base_layer.out_proj = Linear(
                base_layer.out_proj,
                adapter_name,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                init_lora_weights=init_lora_weights,
                use_rslora=use_rslora,
                use_dora=use_dora,
                **kwargs,
            )
        else:
            raise ValueError(f"out_proj must be an instance of nn.Linear for {self.__class__.__name__}.")

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)

    @property
    def embed_dim(self) -> int:
        return self.get_base_layer().embed_dim

    @property
    def kdim(self) -> Optional[int]:
        return self.get_base_layer().kdim

    @property
    def vdim(self) -> Optional[int]:
        return self.get_base_layer().vdim

    @property
    def _qkv_same_embed_dim(self) -> bool:
        return self.get_base_layer()._qkv_same_embed_dim

    @property
    def num_heads(self) -> int:
        return self.get_base_layer().num_heads

    @property
    def dropout(self) -> float:
        return self.get_base_layer().dropout

    @property
    def batch_first(self) -> bool:
        return self.get_base_layer().batch_first

    @property
    def head_dim(self) -> int:
        return self.get_base_layer().head_dim

    @property
    def in_proj_weight(self) -> ms.Parameter:
        return self.get_base_layer().in_proj_weight

    @property
    def in_proj_bias(self) -> ms.Parameter:
        return self.get_base_layer().in_proj_bias

    @property
    def out_proj(self) -> nn.Cell:
        return self.get_base_layer().out_proj.get_base_layer()

    @property
    def bias_k(self) -> Optional[ms.Parameter]:
        return self.get_base_layer().bias_k

    @property
    def bias_v(self) -> Optional[ms.Parameter]:
        return self.get_base_layer().bias_v

    def merge_masks(self, *args, **kwargs) -> tuple[Optional[ms.Tensor], Optional[int]]:
        return self.get_base_layer().merge_masks(*args, **kwargs)

    @property
    def add_zero_attn(self) -> bool:
        return self.get_base_layer().add_zero_attn

    def update_layer(self, *args, **kwargs) -> None:
        super().update_layer(*args, **kwargs)
        # Note: LoRA is applied to both in_proj and out_proj. There is currently no way to only specify one of them.
        self.base_layer.out_proj.update_layer(*args, **kwargs)

    def merge(self, safe_merge: bool = False, adapter_names: Optional[list[str]] = None) -> None:
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

        # Implementation follows this:
        # https://github.com/Baijiong-Lin/LoRA-Torch/blob/4bfed6820b64fcf47064c30f30606a190a4f0d2e/loratorch/layers.py#L73-L79
        # Notably, instead of mutating the weight, we delete the original weight and replace it by the merged weight
        # TODO: work with separate weights
        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # TODO: work with separate weights
                    # merging in_proj (ms.Parameter)
                    orig_weights_in = base_layer.in_proj_weight.clone()
                    orig_weights_in += self.get_delta_weight(active_adapter)
                    if not mint.isfinite(orig_weights_in).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    # merging out_proj (subclass of nn.Linear)
                    orig_weights_out = base_layer.out_proj.weight.clone()
                    orig_weights_out += base_layer.out_proj.get_delta_weight(active_adapter)
                    if not mint.isfinite(orig_weights_out).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    # unregister parameter implicitly and overwrite using merged weights; gradients are computed after
                    # forward and, thus, after unmerging (see forward()), therefore this is safe to do.
                    del base_layer.in_proj_weight
                    base_layer.in_proj_weight = orig_weights_in

                    del base_layer.out_proj.get_base_layer().weight
                    base_layer.out_proj.get_base_layer().weight = orig_weights_out
                    base_layer.out_proj.merge(adapter_names=[active_adapter])
                else:
                    # merging in_proj (ms.Parameter)
                    # TODO: work with separate weights
                    weight_merged = base_layer.in_proj_weight + self.get_delta_weight(active_adapter)

                    # unregister parameter implicitly and overwrite using merged weights; gradients are computed after
                    # forward and, thus, after unmerging (see forward()), therefore this is safe to do.
                    del base_layer.in_proj_weight
                    base_layer.in_proj_weight = weight_merged

                    # merging out_proj (subclass of nn.Linear)
                    weight_merged = base_layer.out_proj.weight + base_layer.out_proj.get_delta_weight(active_adapter)
                    del base_layer.out_proj.get_base_layer().weight
                    base_layer.out_proj.get_base_layer().weight = weight_merged
                    base_layer.out_proj.merge(adapter_names=[active_adapter])
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        # TODO work with separate weights
        base_layer = self.get_base_layer()
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                # Ensure that requires_grad=False for the base weights after unmerging. This may not matter since
                # requires_grad was False when the optimizer was initialized, but still let's try to be correct here.

                # in_proj
                old_weight = base_layer.in_proj_weight - self.get_delta_weight(active_adapter)
                del base_layer.in_proj_weight
                # base_layer.register_parameter("in_proj_weight", nn.Parameter(old_weight, requires_grad=False))
                base_layer.in_proj_weight = ms.Parameter(old_weight, requires_grad=False)

                # out_proj
                old_weight = base_layer.out_proj.base_layer.weight - base_layer.out_proj.get_delta_weight(
                    active_adapter
                )
                del base_layer.out_proj.base_layer.weight
                # base_layer.out_proj.base_layer.register_parameter(
                #     "weight", nn.Parameter(old_weight, requires_grad=False)
                # )
                base_layer.out_proj.base_layer.weight = ms.Parameter(old_weight, requires_grad=False)

        self.get_base_layer().out_proj.unmerge()

    def unload_and_optionally_merge_module(
        self, merge: bool, safe_merge: bool, adapter_names: Optional[list[str]]
    ) -> nn.MultiheadAttention:
        """
        Merging and unloading of the MultiheadAttention module

        This requires an extra step for MultiheadAttention, which is why there is this special method instead of
        relying on the normal merge_and_unload code path.
        """
        if merge:
            self.merge(safe_merge=safe_merge, adapter_names=adapter_names)
        base_layer = self.get_base_layer()

        # extra steps: re-register weights, take care of out_proj layer
        # in_proj
        weight = base_layer.in_proj_weight
        del base_layer.in_proj_weight
        # base_layer.register_parameter("in_proj_weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))
        base_layer.in_proj_weight = ms.Parameter(weight, requires_grad=weight.requires_grad)

        # out_proj
        out_proj_layer = base_layer.out_proj.get_base_layer()
        weight = out_proj_layer.weight
        del out_proj_layer.weight
        # out_proj_layer.register_parameter("weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))
        out_proj_layer.weight = ms.Parameter(weight, requires_grad=weight.requires_grad)

        base_layer.out_proj = out_proj_layer
        return base_layer

    def get_delta_weight(self, adapter) -> ms.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight
        output_tensor = (weight_B @ weight_A) * self.scaling[adapter]
        return output_tensor

    def _check_forward_args(self, x, *args, **kwargs):
        if "adapter_names" in kwargs:
            raise TypeError(f"lora.{self.__class__.__name__} does not support mixed adapter batches.")
        super()._check_forward_args(x, *args, **kwargs)

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
            raise RuntimeError(
                f"{self} is called neither merged nor with disabled adapters. "
                "Please manually call `merge()` or `enable_adapters(enabled=False)` "
                "before invoke forwarding."
            )

        result = (result[0].to(previous_dtype), result[1].to(previous_dtype) if result[1] is not None else result[1])
        return result

    def _restore_weights(self):
        # Restore the weights as registered parameters on the base layer.
        # This is necessary because the way that weights are merged/unmerged (which is necessary for forward to work
        # correctly), the Module "forgets" these attributes. Therefore, we need to call register_parameter explicitly.
        # We cannot call register_parameter for merging/unmerging because that cuts them off from the autograd graph.
        # Note that this is hacky, since we need to ensure that _restore_weights is called by each method that needs it.

        # in_proj
        # TODO work with separate weights
        base_layer = self.get_base_layer()
        weight = base_layer.in_proj_weight
        del base_layer.in_proj_weight
        # base_layer.register_parameter("in_proj_weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))
        base_layer.in_proj_weight = ms.Parameter(weight, requires_grad=weight.requires_grad)

        # out_proj
        base_layer = base_layer.out_proj.get_base_layer()
        weight = base_layer.weight
        del base_layer.weight
        # base_layer.register_parameter("weight", nn.Parameter(weight.data, requires_grad=weight.requires_grad))
        base_layer.weight = ms.Parameter(weight, requires_grad=weight.requires_grad)

    def state_dict(self, *args, **kwargs):
        self._restore_weights()
        return super().state_dict(*args, **kwargs)

    def named_modules(self, *args, **kwargs):
        # Note: no need to also implement modules(), as modules() calls named_modules() under the hood
        self._restore_weights()
        return super().named_modules(*args, **kwargs)

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

    if isinstance(target_base_layer, (nn.Embedding, mint.nn.Embedding)):
        raise NotImplementedError(
            "LoRA is not implemented for `nn.Embedding` layers yet, since it requires `nn.ParameterDict` "
            "which is not supported in MindSpore. Remove embedding layers in `LoraConfig.target_layers` "
            "to use LoRA with MindSpore."
        )
    elif isinstance(target_base_layer, (nn.Conv2d, mint.nn.Conv2d)):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, (nn.Conv3d, mint.nn.Conv3d)):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv3d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.Conv1d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv1d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, nn.MultiheadAttention):
        kwargs.update(lora_config.loftq_config)
        new_module = MultiheadAttention(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, (nn.Dense, mint.nn.Linear)):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = Linear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
