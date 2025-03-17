from mindspore import mint, nn, ops
from mindspore.communication import get_group_size, get_rank
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from .param_wrapper import ZeroParamWrapper


class _Conv(nn.Cell):
    def __init__(
        self, net, zero_stage: int = 0, optimizer_parallel_group: str = GlobalComm.WORLD_COMM_GROUP, cell_type=None
    ):
        super(_Conv, self).__init__(auto_prefix=False)
        self.net = net
        self.set_param_wrapper(zero_stage, optimizer_parallel_group, cell_type)

    @property
    def weight(self):
        return self.net.weight

    @property
    def bias(self):
        return self.net.bias

    def set_param_wrapper(self, zero_stage, optimizer_parallel_group, cell_type=None):
        self.param_wrapper_w = nn.Identity()
        self.param_wrapper_b = nn.Identity()
        if zero_stage == 3:
            # Init parallel settings
            is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
            op_group_size = get_group_size(optimizer_parallel_group) if is_parallel else 1
            op_rank_id = get_rank(optimizer_parallel_group) if is_parallel else 0
            self.param_wrapper_w = ZeroParamWrapper(self.net.weight, zero_stage, optimizer_parallel_group, cell_type)
            split_op = ops.Split(0, op_group_size)
            if self.param_wrapper_w.need_rewrite:
                self.net.weight.assign_value(split_op(self.net.weight)[op_rank_id])
            if self.net.bias is not None:
                self.param_wrapper_b = ZeroParamWrapper(self.net.bias, zero_stage, optimizer_parallel_group, cell_type)
                if self.param_wrapper_b.need_rewrite:
                    self.net.bias.assign_value(split_op(self.net.bias)[op_rank_id])


class Conv1d(_Conv):
    def construct(self, x):
        x = self.net.expand_dims(x, 2)
        output = self.net.conv2d(x, self.param_wrapper_w(self.net.weight))
        if self.net.has_bias:
            output = self.net.bias_add(output, self.param_wrapper_b(self.net.bias))

        output = self.net.squeeze(output)
        return output


class Conv2d(_Conv):
    def construct(self, x):
        output = self.net.conv2d(x, self.param_wrapper_w(self.net.weight))
        if self.net.has_bias:
            output = self.net.bias_add(output, self.param_wrapper_b(self.net.bias))
        return output


class Conv3d(_Conv):
    def construct(self, x):
        weight = self.param_wrapper_w(self.net.weight)
        bias = self.param_wrapper_b(self.net.bias)
        if self.net.group == 1:
            out = self.net.conv3d(x, weight)
            if self.net.has_bias:
                out = self.net.bias_add(out, bias)
        else:
            features = self.net.split_1(x)
            weights = self.net.split_0(weight)
            outputs = ()
            for i in range(self.net.group):
                output = self.net.conv3d(features[i], weights[i])
                outputs = outputs + (output,)
            out = self.net.concat(outputs)
            if self.net.bias is not None:
                new_shape = [1 for _ in range(out.ndim)]
                new_shape[1] = self.net.out_channels
                out = out + bias.reshape(new_shape)
        return out


class Mint_Conv2d(_Conv):
    def construct(self, x):
        weight = self.param_wrapper_w(self.net.weight)
        bias = self.param_wrapper_b(self.net.bias)
        if self.net.padding_mode != "zeros":
            output = self.net.conv2d(
                mint.pad(x, self.net._reversed_padding, mode=self.net.padding_mode),
                weight,
                bias,
                self.net.stride,
                (0, 0),
                self.net.dilation,
                self.net.groups,
            )
        else:
            output = self.net.conv2d(
                x, weight, bias, self.net.stride, self.net.padding, self.net.dilation, self.net.groups
            )
        return output


class Mint_Conv3d(_Conv):
    def construct(self, x):
        weight = self.param_wrapper_w(self.net.weight)
        bias = self.param_wrapper_b(self.net.bias)
        if self.net.padding_mode != "zeros":
            output = self.net.conv3d(
                mint.pad(x, self.net._reversed_padding, mode=self.net.padding_mode),
                weight,
                bias,
                self.net.stride,
                (0, 0, 0),
                self.net.dilation,
                self.net.groups,
            )
        else:
            output = self.net.conv3d(
                x, weight, bias, self.net.stride, self.net.padding, self.net.dilation, self.net.groups
            )
        return output
