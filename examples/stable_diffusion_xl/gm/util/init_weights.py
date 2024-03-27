# Copyright © 2023 Huawei Technologies Co, Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import math

import mindspore.nn as nn
from mindspore.common.initializer import HeNormal, HeUniform, Normal, One, Uniform, XavierUniform, Zero, initializer


def _calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """

    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def init_weights(net, init_type="normal", init_gain=0.02, ignore=None):
    """
    Initialize network weights.

    :param net: network to be initialized
    :type net: nn.Module
    :param init_type: the name of an initialization method: normal | xavier | kaiming | orthogonal
    :type init_type: str
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :type init_gain: float
    """

    if ignore is not None:
        ignore = tuple(ignore)

    for name, cell in net.cells_and_names():
        if ignore is not None and name.startswith(ignore):
            continue

        classname = cell.__class__.__name__

        if hasattr(cell, "in_proj_layer"):
            cell.in_proj_layer = initializer(
                HeUniform(math.sqrt(5.0)), cell.in_proj_layer.shape, cell.in_proj_layer.dtype
            )

        if hasattr(cell, "weight"):
            if init_type == "normal":
                cell.weight = initializer(Normal(init_gain), cell.weight.shape, cell.weight.dtype)
            elif init_type == "xavier":
                cell.weight = initializer(XavierUniform(init_gain), cell.weight.shape, cell.weight.dtype)
            elif init_type == "he":
                cell.weight = initializer(HeUniform(math.sqrt(5.0)), cell.weight.shape, cell.weight.dtype)
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)

            if hasattr(cell, "bias") and cell.bias is not None:
                fan_in, _ = _calculate_fan_in_and_fan_out(cell.weight.shape)
                bound = 1.0 / math.sqrt(fan_in)
                cell.bias = initializer(Uniform(bound), cell.bias.shape, cell.bias.dtype)
        elif classname.find("BatchNorm2d") != -1:
            cell.gamma = initializer(Normal(1.0), cell.gamma.default_input.shape())
            cell.beta = initializer(Zero(), cell.beta.default_input.shape())


def default_init_weights(net, scale=1.0):
    """Initialize network weights.

    Args:
        net (nn.Cell): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual blocks.
    """

    for _, cell in net.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Dense)):
            if hasattr(cell, "weight") and cell.weight is not None:
                kaiming_init = HeNormal(negative_slope=0, mode="fan_in", nonlinearity="relu")
                cell.weight = initializer(kaiming_init, cell.weight.shape, cell.weight.dtype)
                cell.weight *= scale
            if hasattr(cell, "bias") and cell.bias is not None:
                cell.bias = initializer(Zero(), cell.bias.shape, cell.bias.dtype)
        elif isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            cell.gamma = initializer(One(), cell.gamma.shape, cell.gamma.dtype)
            cell.beta = initializer(Zero(), cell.beta.shape, cell.beta.dtype)


def make_layer(block, num_blocks, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        block (nn.Cell): nn.Cell class for basic block.
        num_blocks (int): number of blocks.

    Returns:
        nn.SequentialCell: Stacked blocks in nn.SequentialCell.
    """

    layers = [block(**kwarg) for _ in range(num_blocks)]
    module = nn.SequentialCell(*layers)

    return module
