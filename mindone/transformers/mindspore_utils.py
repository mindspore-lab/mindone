# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import inspect
from typing import Callable, List, Optional, Set, Tuple, Union

from transformers.utils import logging

import mindspore as ms
from mindspore import mint, nn, ops
from mindspore.common.initializer import Normal, Zero, initializer

ALL_LAYERNORM_LAYERS = [nn.LayerNorm]

logger = logging.get_logger(__name__)


def prune_linear_layer(layer: nn.Dense, index: ms.Tensor, dim: int = 0) -> nn.Dense:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`mindspore.nn.Dense`): The layer to prune.
        index (`mindspore.Tensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `mindspore.nn.Dense`: The pruned layer as a new layer with `requires_grad=True`.
    """
    w = layer.weight.index_select(dim, index).clone()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone()
        else:
            b = layer.bias[index].clone()
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = nn.Dense(new_size[1], new_size[0], has_bias=layer.bias is not None)
    new_layer.weight.requires_grad = False
    ops.assign(new_layer.weight, w)
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        ops.assign(new_layer.bias, b)
        new_layer.bias.requires_grad = True
    return new_layer


class Conv1D(nn.Cell):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = ms.Parameter(initializer(Normal(0.02), [nx, nf]), name="weight")
        self.bias = ms.Parameter(initializer(Zero(), [nf]), name="bias")

    def construct(self, x):
        size_out = x.shape[:-1] + (self.nf,)
        x = mint.addmm(self.bias, x.view(-1, x.shape[-1]), self.weight)
        x = x.view(size_out)
        return x


def prune_conv1d_layer(layer: Conv1D, index: ms.Tensor, dim: int = 1) -> Conv1D:
    """
    Prune a Conv1D layer to keep only entries in index. A Conv1D work as a Linear layer (see e.g. BERT) but the weights
    are transposed.

    Used to remove heads.

    Args:
        layer ([`~mindspore_utils.Conv1D`]): The layer to prune.
        index (`ms.Tensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 1): The dimension on which to keep the indices.

    Returns:
        [`~mindspore_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    w = layer.weight.index_select(dim, index).clone()
    if dim == 0:
        b = layer.bias.clone()
    else:
        b = layer.bias[index].clone()
    new_size = list(layer.weight.shape)
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0])
    new_layer.weight.requires_grad = False
    ops.assign(new_layer.weight, w)
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    ops.assign(new_layer.bias, b)
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(layer: Union[nn.Dense, Conv1D], index: ms.Tensor, dim: Optional[int] = None) -> Union[nn.Dense, Conv1D]:
    """
    Prune a Conv1D or Dense layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`Union[mindspore.nn.Dense, Conv1D]`): The layer to prune.
        index (`mindspore.Tensor`): The indices to keep in the layer.
        dim (`int`, *optional*): The dimension on which to keep the indices.

    Returns:
        `mindspore.nn.Dense` or [`~mindspore_utils.Conv1D`]: The pruned layer as a new layer with `requires_grad=True`.
    """
    if isinstance(layer, nn.Dense):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError(f"Can't prune layer of class {layer.__class__}")


def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], ms.Tensor]:
    """
    Finds the heads and their indices taking `already_pruned_heads` into account.

    Args:
        heads (`List[int]`): List of the indices of heads to prune.
        n_heads (`int`): The number of heads in the model.
        head_size (`int`): The size of each head.
        already_pruned_heads (`Set[int]`): A set of already pruned heads.

    Returns:
        `Tuple[Set[int], ms.Tensor]`: A tuple with the indices of heads to prune taking `already_pruned_heads`
        into account and the indices of rows/columns to keep in the layer weight.
    """
    mask = ops.ones((n_heads, head_size))
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).eq(1)
    index = ops.arange(len(mask))[mask].long()
    return heads, index


def apply_chunking_to_forward(
    forward_fn: Callable[..., ms.Tensor], chunk_size: int, chunk_dim: int, *input_tensors
) -> ms.Tensor:
    """
    This function chunks the `input_tensors` into smaller input tensor parts of size `chunk_size` over the dimension
    `chunk_dim`. It then applies a layer `forward_fn` to each chunk independently to save memory.

    If the `forward_fn` is independent across the `chunk_dim` this function will yield the same result as directly
    applying `forward_fn` to `input_tensors`.

    Args:
        forward_fn (`Callable[..., ms.Tensor]`):
            The forward function of the model.
        chunk_size (`int`):
            The chunk size of a chunked tensor: `num_chunks = len(input_tensors[0]) / chunk_size`.
        chunk_dim (`int`):
            The dimension over which the `input_tensors` should be chunked.
        input_tensors (`Tuple[ms.Tensor]`):
            The input tensors of `forward_fn` which will be chunked

    Returns:
        `ms.Tensor`: A tensor with the same shape as the `forward_fn` would have given if applied`.


    Examples:

    ```python
    # rename the usual forward() fn to forward_chunk()
    def forward_chunk(self, hidden_states):
        hidden_states = self.decoder(hidden_states)
        return hidden_states


    # implement a chunked forward function
    def forward(self, hidden_states):
        return apply_chunking_to_forward(self.forward_chunk, self.chunk_size_lm_head, self.seq_len_dim, hidden_states)
    ```"""

    assert len(input_tensors) > 0, f"{input_tensors} has to be a tuple/list of tensors"

    # inspect.signature exist since python 3.5 and is a python method -> no problem with backward compatibility
    num_args_in_forward_chunk_fn = len(inspect.signature(forward_fn).parameters)
    if num_args_in_forward_chunk_fn != len(input_tensors):
        raise ValueError(
            f"forward_chunk_fn expects {num_args_in_forward_chunk_fn} arguments, but only {len(input_tensors)} input "
            "tensors are given"
        )

    if chunk_size > 0:
        tensor_shape = input_tensors[0].shape[chunk_dim]
        for input_tensor in input_tensors:
            if input_tensor.shape[chunk_dim] != tensor_shape:
                raise ValueError(
                    f"All input tenors have to be of the same shape: {tensor_shape}, "
                    f"found shape {input_tensor.shape[chunk_dim]}"
                )

        if input_tensors[0].shape[chunk_dim] % chunk_size != 0:
            raise ValueError(
                f"The dimension to be chunked {input_tensors[0].shape[chunk_dim]} has to be a multiple of the chunk "
                f"size {chunk_size}"
            )

        num_chunks = input_tensors[0].shape[chunk_dim] // chunk_size

        # chunk input tensor into tuples
        input_tensors_chunks = tuple(input_tensor.chunk(num_chunks, dim=chunk_dim) for input_tensor in input_tensors)
        # apply forward fn to every tuple
        output_chunks = tuple(forward_fn(*input_tensors_chunk) for input_tensors_chunk in zip(*input_tensors_chunks))
        # concatenate output at same dimension
        return ops.cat(output_chunks, dim=chunk_dim)

    return forward_fn(*input_tensors)
