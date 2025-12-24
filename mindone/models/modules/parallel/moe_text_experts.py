from typing import Literal, Optional

import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import mint, nn
from mindspore.communication import get_group_size, get_rank
from mindspore.communication.management import GlobalComm
from mindspore.context import ParallelMode
from mindspore.parallel._utils import _get_parallel_mode

from .param_wrapper import ZeroParamWrapper


class MoeTextExperts(nn.Cell):
    def __init__(
        self,
        net: nn.Cell,
        zero_stage: Literal[0, 1, 2, 3] = 0,
        optimizer_parallel_group: str = GlobalComm.WORLD_COMM_GROUP,
        cell_type: Optional[mstype.Type] = None,
    ):
        super().__init__(auto_prefix=False)
        self.net = net
        self.set_param_wrapper(zero_stage, optimizer_parallel_group, cell_type)

    def set_param_wrapper(self, zero_stage, optimizer_parallel_group, cell_type=None):
        self.param_wrapper_gate_up_proj = nn.Identity()
        self.param_wrapper_down_proj = nn.Identity()
        if zero_stage == 3:
            # Init parallel settings
            is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
            op_group_size = get_group_size(optimizer_parallel_group) if is_parallel else 1
            op_rank_id = get_rank(optimizer_parallel_group) if is_parallel else 0
            self.op_group_size = op_group_size
            self.op_rank_id = op_rank_id
            self.param_wrapper_gate_up_proj = ZeroParamWrapper(
                self.net.gate_up_proj, zero_stage, optimizer_parallel_group, cell_type
            )
            if self.param_wrapper_gate_up_proj.need_rewrite:
                self.net.gate_up_proj.assign_value(
                    Tensor.from_numpy(
                        self.net.gate_up_proj.numpy().reshape(op_group_size, -1, *self.net.gate_up_proj.shape[1:])[
                            op_rank_id
                        ]
                    )
                )
            self.param_wrapper_down_proj = ZeroParamWrapper(
                self.net.down_proj, zero_stage, optimizer_parallel_group, cell_type
            )
            if self.param_wrapper_down_proj.need_rewrite:
                self.net.down_proj.assign_value(
                    Tensor.from_numpy(
                        self.net.down_proj.numpy().reshape(op_group_size, -1, *self.net.down_proj.shape[1:])[op_rank_id]
                    )
                )

    def construct(self, hidden_states, routing_weights, router_indices):
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.net.hidden_size)  # (num_tokens, hidden_size)

        hidden_states = hidden_states.repeat(self.net.num_experts, 1)
        hidden_states = hidden_states.view(self.net.num_experts, -1, self.net.hidden_size)

        gate_up = mint.bmm(hidden_states, self.param_wrapper_gate_up_proj(self.net.gate_up_proj))
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = mint.bmm((up * self.net.act_fn(gate)), self.param_wrapper_down_proj(self.net.down_proj))
        next_states = next_states.reshape(self.net.num_experts, batch_size, -1, self.net.hidden_size)
        next_states = next_states * routing_weights.swapaxes(0, 1).view(self.net.num_experts, batch_size, -1)[..., None]
        next_states = next_states.sum(dim=0)
        return next_states


class Llama4TextExpertsWrapper(nn.Cell):
    def __init__(
        self,
        net: nn.Cell,
        zero_stage: Literal[0, 1, 2, 3] = 0,
        optimizer_parallel_group: str = GlobalComm.WORLD_COMM_GROUP,
        cell_type: Optional[mstype.Type] = None,
    ):
        super().__init__(auto_prefix=False)
        self.net = net
        self.set_param_wrapper(zero_stage, optimizer_parallel_group, cell_type)

    def set_param_wrapper(self, zero_stage, optimizer_parallel_group, cell_type=None):
        self.param_wrapper_gate_up_proj = nn.Identity()
        self.param_wrapper_down_proj = nn.Identity()
        if zero_stage == 3:
            # Init parallel settings
            is_parallel = _get_parallel_mode() == ParallelMode.DATA_PARALLEL
            op_group_size = get_group_size(optimizer_parallel_group) if is_parallel else 1
            op_rank_id = get_rank(optimizer_parallel_group) if is_parallel else 0
            self.op_group_size = op_group_size
            self.op_rank_id = op_rank_id
            self.param_wrapper_gate_up_proj = ZeroParamWrapper(
                self.net.gate_up_proj, zero_stage, optimizer_parallel_group, cell_type
            )
            if self.param_wrapper_gate_up_proj.need_rewrite:
                self.net.gate_up_proj.assign_value(
                    Tensor.from_numpy(
                        self.net.gate_up_proj.numpy().reshape(op_group_size, -1, *self.net.gate_up_proj.shape[1:])[
                            op_rank_id
                        ]
                    )
                )
            self.param_wrapper_down_proj = ZeroParamWrapper(
                self.net.down_proj, zero_stage, optimizer_parallel_group, cell_type
            )
            if self.param_wrapper_down_proj.need_rewrite:
                self.net.down_proj.assign_value(
                    Tensor.from_numpy(
                        self.net.down_proj.numpy().reshape(op_group_size, -1, *self.net.down_proj.shape[1:])[op_rank_id]
                    )
                )

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        """
        This should really not be run on a single machine, as we are reaching compute bound:
        - the inputs are expected to be "sorted" per expert already.
        - the weights are viewed with another dim, to match num_expert, 1, shape * num_tokens, shape

        Args:
            hidden_states (ms.Tensor): (batch_size * token_num, hidden_size)
            selected_experts (ms.Tensor): (batch_size * token_num, top_k)
            routing_weights (ms.Tensor): (batch_size * token_num, top_k)
        Returns:
            ms.Tensor
        """
        hidden_states = hidden_states.view(
            self.param_wrapper_gate_up_proj(self.net.gate_up_proj).shape[0], -1, self.net.hidden_size
        )
        gate_up = mint.bmm(hidden_states, self.param_wrapper_gate_up_proj(self.net.gate_up_proj))
        gate, up = gate_up.chunk(2, dim=-1)  # not supported for DTensors
        next_states = mint.bmm((up * self.net.act_fn(gate)), self.param_wrapper_down_proj(self.net.down_proj))
        next_states = next_states.view(-1, self.net.hidden_size)
        return next_states
