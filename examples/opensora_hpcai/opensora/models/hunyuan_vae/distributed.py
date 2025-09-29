from typing import Literal, Optional, Union

from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import mint, ops
from mindspore.communication import get_group_size, get_rank

from ...acceleration import SplitForwardGatherBackward, get_sequence_parallel_group

__all__ = ["initialize_parallel_group", "GroupNormTP", "Conv3dTPCol", "Conv3dTPRow"]


def initialize_parallel_group() -> tuple[Optional[str], int, int]:
    tp_group, tp_size, rank = get_sequence_parallel_group(), 1, 0
    if tp_group is not None:
        tp_size = get_group_size(tp_group)
        rank = get_rank(tp_group)
    return tp_group, tp_size, rank


class GroupNormTP(mint.nn.GroupNorm):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
        dtype: Optional[mstype.Type] = None,
        enable_tp: bool = False,
    ):
        self._enable_tp = enable_tp
        self._tp_group, self._tp_size, self._rank = initialize_parallel_group()
        if enable_tp and self._tp_group is not None:
            if num_groups % self._tp_size != 0:
                raise ValueError(
                    f"GroupNormTP: num_groups ({num_groups}) must be divisible by tp_size ({self._tp_size})."
                )
            num_groups = num_groups // self._tp_size
        super().__init__(num_groups, num_channels, eps, affine, dtype)

    def split_weights(self) -> "GroupNormTP":
        """
        Split cell weights across tensor parallel groups for distributed execution.
        """
        if self._enable_tp and self._tp_group is not None and self.affine:
            size = self.num_channels // self._tp_size
            self.weight.set_data(self.weight[self._rank * size : (self._rank + 1) * size])
            self.bias.set_data(self.bias[self._rank * size : (self._rank + 1) * size])
        return self


class Conv3dTPCol(mint.nn.Conv3d):
    def split_weights(self) -> "Conv3dTPCol":
        """
        Split cell weights across tensor parallel groups for distributed execution.
        """
        tp_group, tp_size, rank = initialize_parallel_group()
        if tp_group is not None:
            size = self.out_channels // tp_size
            self.weight.set_data(self.weight[rank * size : (rank + 1) * size])
            if self.bias is not None:
                self.bias.set_data(self.bias[rank * size : (rank + 1) * size])
        return self


class Conv3dTPRow(mint.nn.Conv3d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int, int]],
        stride: Union[int, tuple[int, int, int]] = 1,
        padding: Union[int, tuple[int, int, int], Literal["same", "valid"]] = 0,
        dilation: Union[int, tuple[int, int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: Literal["zeros", "reflect", "circular", "replicate"] = "zeros",
        dtype: Optional[mstype.Type] = None,
        split_output: bool = False,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, dtype
        )
        self._tp_group, self._tp_size, self._rank = initialize_parallel_group()
        self._split_output, self._reduce = False, None
        if self._tp_group is not None:
            self._reduce = ops.AllReduce(op=ops.ReduceOp.SUM, group=self._tp_group)
            self._split_output = split_output
            if split_output:
                self._split_forward_gather_backward = SplitForwardGatherBackward(dim=1, group=self._tp_group)

    def split_weights(self) -> "Conv3dTPRow":
        """
        Split cell weights across tensor parallel groups for distributed execution.
        """
        if self._tp_group is not None:
            size = self.in_channels // self._tp_size
            self.weight.set_data(self.weight[:, self._rank * size : (self._rank + 1) * size])
        return self

    def construct(self, x: Tensor) -> Tensor:
        out = self.conv3d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        if self._reduce is not None:
            out = self._reduce(out)
        if self.bias is not None:
            out = out + self.bias[:, None, None, None]
        if self._split_output:
            out = self._split_forward_gather_backward(out)
        return out
