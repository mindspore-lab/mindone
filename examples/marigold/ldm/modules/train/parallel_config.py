# Copyright 2022 Huawei Technologies Co., Ltd
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
"""Transformer Networks"""

from mindspore.context import ParallelMode


class ParallelConfig:
    r"""
    ParallelConfig for the setting the global data parallel, model parallel and fusion group.
    """
    dp = 8
    mp = 1
    pipeline_stage = 1
    recompute = False
    optimizer_shard = False
    fusion_group = 1
    parallel_mode = ParallelMode.SEMI_AUTO_PARALLEL
    vocab_emb_dp = False
    ep = dp
    capacity_factor = 1.5
    expert_num = 32
    aux_loss_factor = 0.01

    @staticmethod
    def set_global_parallel_config(
        dp=1,
        mp=1,
        recompute=True,
        stages=1,
        optimizer_shard=True,
        fusion_group=4,
        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
        vocab_emb_dp=True,
    ):
        r"""
        The parallel configure setting

        Args:
            dp (int): The data parallel way. Default: 1
            mp (int): The model parallel way. Default: 1
            stages (int): The number of the pipeline stage. Should be a positive value. Default: 1.
            optimizer_shard (bool): Enable optimizer state sharding or not. Default: True.
            fusion_group (int): The fusion group size of the optimizer state sharding. Default: 4.
            recompute (bool): Enable recomputation of the transformer block or not. Default: False.
            parallel_mode (ParallelMode): Can be SEMI_AUTO_PARALLEL, DATA_AUTO_PARALLEL or AUTO_PARALLEL.
            vocab_emb_dp (bool): Shard embedding in model parallel or data parallel. Default: True

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> ParallelConfig(dp=1, mp=1)
            >>> ParallelConfig(stages=4)
        """
        ParallelConfig.dp = dp
        ParallelConfig.mp = mp
        ParallelConfig.pipeline_stage = stages
        ParallelConfig.optimizer_shard = optimizer_shard
        ParallelConfig.fusion_group = fusion_group
        ParallelConfig.recompute = recompute
        ParallelConfig.parallel_mode = parallel_mode
        ParallelConfig.vocab_emb_dp = vocab_emb_dp
