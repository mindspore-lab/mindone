# Copyright 2024 Huawei Technologies Co., Ltd
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
# ======================

"""Model and data parallel groups."""
import mindspore
from mindspore.communication import create_group, destroy_group, get_group_size, get_rank
from mindspore import hal

_GLOBAL_STREAM = None
_SP_SEND_STREAM = None
_SP_RECV_STREAM = None
_SP_SEND_OML_STREAM = None
_SP_RECV_OML_STREAM = None

group_info_maps = {}

# special_groups has a different initialization process compared to normal_groups
normal_groups = ['tp', 'dp', 'pp', 'cp', 'dp-cp', 'tp-pp', 'tp-dp-cp', 'tp-dp', 'tp-cp']
special_groups = ['ep', 'tp-ep', 'dp-independent_ep', 'vpp', 'embedding', 'position_embedding']
valid_groups = normal_groups + special_groups

class GroupInfo:
    """ Comm Group Info """
    def __init__(self):
        self.group = None
        self.world_size = None
        self.rank = None
        self.global_ranks = None
        self.is_group_created = False

    def reset(self):
        if self.group is not None and self.is_group_created:
            destroy_group(self.group)
        self.group = None
        self.world_size = None
        self.rank = None
        self.global_ranks = None
        self.is_group_created = False

def get_group_info(mode):
    global group_info_maps
    if mode not in group_info_maps:
        assert mode in valid_groups
        group_info_maps[mode] = GroupInfo()
    return group_info_maps[mode]

class CreateCommGroups():
    '''Generate ranks for each parallel type.'''

    def __init__(self, tp, ep, dp, pp, cp, order):
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }
        self.rank = get_rank()
        self.order = order

        for name, size in self.name_to_size.items():
            if name not in order:
                if size == 1:
                    order = order + '-' + name
                else:
                    raise RuntimeError(
                        f"The size of ({name}) is ({size}), \
                        but you haven't specified the order ({self.order})."
                    )

        self.order_w_ep = order
        self.order_wo_ep = '-'.join([token for token in order.split('-') if token != 'ep'])
        self.ordered_size_wo_ep = []
        self.ordered_size_w_ep = []

        for token in order.split('-'):
            if token == 'dp':
                self.ordered_size_w_ep.append(self.dp // self.ep)
                self.ordered_size_wo_ep.append(self.dp)
            elif token == 'ep':
                self.ordered_size_w_ep.append(self.ep)
            else:
                self.ordered_size_w_ep.append(self.name_to_size[token])
                self.ordered_size_wo_ep.append(self.name_to_size[token])

    def get_mask(self, order, token):
        ordered_token = order.split('-')
        token = token.split('-')
        mask = [False] * len(ordered_token)
        for t in token:
            mask[ordered_token.index(t)] = True
        return mask

    def get_ranks(self, token, independent_ep=False):
        '''Get rank group by input token.

        Arguments:
            token (str): Specify the ranks type that want to get. Use a hyphen '-' to separate multiple parallel types.
            independent_ep (bool): Whether to treat EP and DP independently. Default: False.
        '''
        if independent_ep:
            parallel_size = self.ordered_size_w_ep
            order = self.order_w_ep
        else:
            parallel_size = self.ordered_size_wo_ep
            order = self.order_wo_ep
        mask = self.get_mask(order, token)
        ranks = self._dispatch_comm_ranks(self.world_size, parallel_size, mask)
        return ranks

    def init_group(self, input_mode, independent_ep=False):
        '''Create data parallel group.'''
        mode = input_mode + '-independent_ep' if input_mode == 'dp' and independent_ep else input_mode
        comm_group = get_group_info(mode)

        assert comm_group.group is None, f'{mode} parallel group is already initialized'

        for ranks in self.get_ranks(input_mode, independent_ep=independent_ep):
            if self.rank in ranks:
                group = mode + '-' + '-'.join([str(i) for i in ranks])
                comm_group.group = group
                comm_group.global_ranks = ranks
                comm_group.world_size = len(ranks)

    def init_embedding_group(self, pipeline_model_parallel_split_rank):
        '''Init pipeline parallel group.'''
        embedding_group = get_group_info('embedding')
        position_embedding_group = get_group_info('position_embedding')
        assert embedding_group.group is None, 'embedding group is already initialized'
        assert position_embedding_group.group is None, 'position embedding group is already initialized'
        for ranks in self.get_ranks('pp'):
            # Setup embedding group (to exchange gradients between
            # first and last stages).
            if len(ranks) > 1:
                embedding_ranks = [ranks[0], ranks[-1]]
                position_embedding_ranks = [ranks[0]]
                if pipeline_model_parallel_split_rank is not None:
                    if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                        embedding_ranks = [
                            ranks[0],
                            ranks[pipeline_model_parallel_split_rank],
                            ranks[-1],
                        ]
                    if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                        position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
            else:
                embedding_ranks = ranks
                position_embedding_ranks = ranks

            if self.rank in embedding_ranks:
                group = 'embedding-' + '-'.join([str(i) for i in embedding_ranks])
                embedding_group.group = group
                embedding_group.global_ranks = embedding_ranks
            if self.rank in position_embedding_ranks:
                group = 'position_embedding-' + '-'.join([str(i) for i in position_embedding_ranks])
                position_embedding_group.group = group
                position_embedding_group.global_ranks = position_embedding_ranks

    def _dispatch_comm_ranks(self, world_size, parallel_size, mask):
        """dispatch comm ranks"""
        def prefix_product(a, init=1):
            r = [init]
            for v in a:
                init = init * v
                r.append(init)
            return r

        def modulo(index, shape, stride=None):
            if stride is None:
                stride = prefix_product(shape)
            idx = [(index // d) % s for s, d in zip(shape, stride)]
            assert (
                sum([x * y for x, y in zip(idx, stride[:-1])]) == index
            ), "idx {} with shape {} mismatch the return idx {}".format(index, shape, idx)
            return idx

        masked_shape = [s for s, m in zip(parallel_size, mask) if m]
        unmasked_shape = [s for s, m in zip(parallel_size, mask) if not m]

        global_stride = prefix_product(parallel_size)
        masked_stride = [d for d, m in zip(global_stride, mask) if m]
        unmasked_stride = [d for d, m in zip(global_stride, mask) if not m]

        group_size = prefix_product(masked_shape)[-1]
        num_of_group = world_size // group_size

        ranks = []
        for group_index in range(num_of_group):
            # get indices from unmaksed for group_index.
            decomposed_group_idx = modulo(group_index, unmasked_shape)
            rank = []
            for rank_in_group in range(group_size):
                # get indices from masked for rank_in_group.
                decomposed_rank_idx = modulo(rank_in_group, masked_shape)
                masked_inner_product = sum([x * y for x, y in zip(decomposed_rank_idx, masked_stride)])
                unmasked_inner_product = sum([x * y for x, y in zip(decomposed_group_idx, unmasked_stride)])
                rank.append(masked_inner_product + unmasked_inner_product)
            ranks.append(rank)
        return ranks

# pylint: disable=W0613
def initialize_model_parallel(tensor_model_parallel_size=1,
                              pipeline_model_parallel_size=1,
                              virtual_pipeline_model_parallel_size=None,
                              pipeline_model_parallel_split_rank=None,
                              context_parallel_size=1,
                              expert_model_parallel_size=1,
                              order="tp-cp-ep-dp-pp",
                              communicator_config_path=None,
                              **kwargs):
    """Initialize model data parallel groups.
    """

    # pylint: disable=W0212
    assert mindspore.communication._comm_helper._is_initialized()
    world_size = get_group_size()

    minimum_world_size = (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)

    if world_size % minimum_world_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size = world_size // minimum_world_size

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    if virtual_pipeline_model_parallel_size is not None:
        if pipeline_model_parallel_size < 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        vpp_group = get_group_info('vpp')
        vpp_group.rank = 0
        vpp_group.world_size = virtual_pipeline_model_parallel_size

    order = order.lower()
    order_list = order.split('-')
    if not order:
        raise RuntimeError(f"order can not be empty.")
    if len(set(order_list)) != len(order_list):
        raise RuntimeError(f"Duplicate elements in order ({order}).")
    if 'ep' in order:
        if 'ep-dp' not in order and 'dp-ep' not in order:
            raise RuntimeError(f"The ep and dp must be adjacent in order ({order}).")

    rank_generator = CreateCommGroups(tp=tensor_model_parallel_size,\
                                      ep=expert_model_parallel_size, \
                                      dp=data_parallel_size, pp=pipeline_model_parallel_size, \
                                      cp=context_parallel_size, order=order)

    # Build the basic parallel groups.
    for mode in normal_groups:
        rank_generator.init_group(mode)

    # Build the expert-parallel groups which share ranks with DP.
    rank_generator.init_group('ep', independent_ep=True)
    rank_generator.init_group('tp-ep', independent_ep=True)
    rank_generator.init_group('dp', independent_ep=True)

    # Build the pipeline-parallel related groups.
    rank_generator.init_embedding_group(pipeline_model_parallel_split_rank)

    global _GLOBAL_STREAM
    assert (_GLOBAL_STREAM is None), 'Global stream is already initialized'
    _GLOBAL_STREAM = hal.Stream()

    global _SP_SEND_STREAM
    global _SP_RECV_STREAM
    global _SP_SEND_OML_STREAM
    global _SP_RECV_OML_STREAM
    if context_parallel_size > 1:
        _SP_SEND_STREAM = hal.Stream()
        # _SP_RECV_STREAM = hal.Stream()
        _SP_SEND_OML_STREAM = hal.Stream()
        _SP_RECV_OML_STREAM = hal.Stream()

    # a temporary workaround for dp group failure initialization in ms.dataset
    if get_dp_world_size() > 1:
        get_dp_group()


### get group
# pylint: disable=C0330
def _get_group_helper(mode):
    comm_group = get_group_info(mode)
    assert comm_group.group is not None, \
        (f"{mode} parallel group is not initialized. Please check whether communication "
         f"is initialized and {mode} in order.")
    if not comm_group.is_group_created:
        create_group(comm_group.group, comm_group.global_ranks)
        comm_group.is_group_created = True
    return comm_group.group

def get_tp_group():
    """Get the tensor model parallel group the caller rank belongs to."""
    return _get_group_helper('tp')

def get_cp_group():
    """Get the context parallel group the caller rank belongs to."""
    return _get_group_helper('cp')

def get_ep_group():
    return _get_group_helper('ep')

def get_dp_group(with_context_parallel=False):
    """Get the data parallel group the caller rank belongs to."""
    return _get_group_helper('dp-cp') if with_context_parallel else _get_group_helper('dp')

def get_pp_group():
    """Get the pipeline model parallel group the caller rank belongs to."""
    return _get_group_helper('pp')

def get_embedding_group():
    """Get the embedding group the caller rank belongs to."""
    return _get_group_helper('embedding')

def get_tensor_and_expert_parallel_group():
    return _get_group_helper('tp-ep')

def get_data_modulo_expert_parallel_group():
    return _get_group_helper('dp-independent_ep')

def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    return _get_group_helper('tp-pp')

def get_tensor_and_context_parallel_group():
    return _get_group_helper('tp-cp')


### get global ranks
def _get_global_ranks_helper(mode, check_initialized=True):
    comm_group = get_group_info(mode)
    if check_initialized:
        assert comm_group.global_ranks is not None, \
            (f"{mode} parallel group is not initialized. Please check whether communication "
             f"is initialized and {mode} in order.")
    return comm_group.group

# pylint: disable=C0330
def get_cp_global_ranks(check_initialized=True):
    """Get all global ranks of the context parallel group that the caller rank belongs to."""
    return _get_global_ranks_helper('cp', check_initialized)


### get world size
def _get_world_size_helper(mode):
    comm_group = get_group_info(mode)
    return comm_group.world_size

def get_tp_world_size():
    """Return world size for the tensor model parallel group."""
    return _get_world_size_helper('tp')

def get_cp_world_size():
    """Return world size for the context parallel group."""
    return _get_world_size_helper('cp')

def get_ep_world_size():
    """Return world size for the expert model parallel group"""
    tensor_and_expert_parallel_world_size = _get_world_size_helper('tp-ep')
    return tensor_and_expert_parallel_world_size // get_tp_world_size()

def get_dp_world_size(with_context_parallel=False):
    """Return world size for the data parallel group."""
    return _get_world_size_helper('dp-cp') if with_context_parallel else _get_world_size_helper('dp')

def get_pp_world_size():
    """Return world size for the pipeline model parallel group."""
    return _get_world_size_helper('pp')

def get_vpp_world_size():
    """Return world size for the virtual pipeline model parallel group."""
    return _get_world_size_helper('vpp')

def get_tensor_and_expert_parallel_world_size():
    """Return world size for the expert model parallel group times model parallel group.
       Currently, each expert will also be distributed across TP group by default.
    """
    return _get_world_size_helper('tp-ep')

def get_tensor_and_context_parallel_world_size():
    """Return world size for the tensor parallel group and context parallel group."""
    return _get_world_size_helper('tp-cp')


### get rank
def _get_rank_helper(mode):
    comm_group = get_group_info(mode)
    if comm_group.rank is not None:
        return comm_group.rank
    comm_group.rank = 0 if _get_world_size_helper(mode) == 1 else get_rank(group=_get_group_helper(mode))
    return comm_group.rank

def get_tp_rank():
    """Return my rank for the tensor model parallel group."""
    return _get_rank_helper('tp')

def get_cp_rank():
    """Return my rank for the context parallel group."""
    return _get_rank_helper('cp')

def get_ep_rank():
    """Return my rank for the expert parallel group"""
    tensor_and_expert_parallel_rank = _get_rank_helper('tp-ep')
    return tensor_and_expert_parallel_rank // get_tp_world_size()

def get_dp_rank(with_context_parallel=False):
    """Return my rank for the data parallel group."""
    return _get_rank_helper('dp-cp') if with_context_parallel else _get_rank_helper('dp')

def get_pp_rank():
    """Return my rank for the pipeline model parallel group."""
    return _get_rank_helper('pp')

def is_pipeline_first_stage(ignore_virtual=False):
    """Return True if in the first pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        if get_vpp_world_size() is not None and get_vpp_rank() != 0:
            return False
    return get_pp_rank() == 0

def is_pipeline_last_stage(ignore_virtual=False):
    """Return True if in the last pipeline model-parallel stage, False otherwise."""
    if not ignore_virtual:
        vpp_world_size = get_vpp_world_size()
        if vpp_world_size is not None and get_vpp_rank() != (vpp_world_size - 1):
            return False
    return get_pp_rank() == (get_pp_world_size() - 1)

def is_rank_in_embedding_group(ignore_virtual=False):
    """Return true if current rank is in embedding group, False otherwise."""
    ret = False
    rank = get_rank()
    embedding_group = get_group_info('embedding')
    global_ranks = embedding_group.global_ranks
    if global_ranks is None:
        return False
    if ignore_virtual:
        return rank in global_ranks
    if rank in global_ranks:
        if rank == global_ranks[0]:
            ret = is_pipeline_first_stage(ignore_virtual=False)
        elif rank == global_ranks[-1]:
            ret = is_pipeline_last_stage(ignore_virtual=False)
        else:
            ret = True
    return ret

def get_vpp_rank():
    """Get the virtual pipeline-parallel rank."""
    comm_group = get_group_info('vpp')
    return comm_group.rank

def set_vpp_rank(rank):
    """Set the virtual pipeline-parallel rank."""
    comm_group = get_group_info('vpp')
    comm_group.rank = rank

def set_ep_rank(rank):
    """Set expert model parallel rank."""
    comm_group = get_group_info('ep')
    comm_group.rank = rank


def get_stream():
    """Return global stream. There is only one stream for each npu."""
    assert _GLOBAL_STREAM is not None, "Global stream is not initialized"
    return _GLOBAL_STREAM


def get_sp_send_stream():
    """Return send stream for sequence parallel."""
    assert _SP_SEND_STREAM is not None, "Sp send stream is not initialized"
    return _SP_SEND_STREAM


def get_sp_recv_stream():
    """Return recv stream for sequence parallel."""
    assert _SP_RECV_STREAM is not None, "Sp receive stream is not initialized"
    return _SP_RECV_STREAM


def get_sp_send_oml_stream():
    """Return send stream for sequence parallel."""
    assert _SP_SEND_OML_STREAM is not None, "Sp send oml stream is not initialized"
    return _SP_SEND_OML_STREAM


def get_sp_recv_oml_stream():
    """Return recv stream for sequence parallel."""
    assert _SP_RECV_OML_STREAM is not None, "Sp receive oml stream is not initialized"
    return _SP_RECV_OML_STREAM


def destroy_model_parallel():
    """Set the groups to none."""
    global group_info_maps
    for _, comm_group in group_info_maps.items():
        comm_group.reset()
    global _GLOBAL_STREAM
    _GLOBAL_STREAM = None
    global _SP_SEND_STREAM
    _SP_SEND_STREAM = None
    global _SP_RECV_STREAM
    _SP_RECV_STREAM = None
    global _SP_SEND_OML_STREAM
    _SP_SEND_OML_STREAM = None
    global _SP_RECV_OML_STREAM
    _SP_RECV_OML_STREAM = None
