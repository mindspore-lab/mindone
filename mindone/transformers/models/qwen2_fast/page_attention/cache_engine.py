# Copyright 2023 Huawei Technologies Co., Ltd
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
"""manage simple kv cache for paged attention."""
import logging
from typing import List


class BlockMemPool:
    """
    block-wise memory pool
    reuse memory at the granularity of block.

    memory view is as follows:

    the whole memory area(big enough) would be split to 'num_blocks' blocks,
    each contains

    +------+------+------+------+
    | b0s0 | b0s1 | b0s2 | b0s3 |
    +------+------+------+------+
    | b1s0 | b1s1 | b1s2 | b1s3 |
    +------+------+------+------+
    | b2s0 | b2s1 | b2s2 | b2s3 |
    +------+------+------+------+

    """
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = [i for i in range(num_blocks)]
        self.used_blocks = []

    def allocate_block(self, num_new_block: int):
        if len(self.free_blocks) < num_new_block:
            raise RuntimeError('block pool is out of memory')

        new_blocks = self.free_blocks[0:num_new_block]
        self.used_blocks += new_blocks
        self.free_blocks = self.free_blocks[num_new_block:]
        logging.info("free block num in pool: %s", len(self.free_blocks))
        return new_blocks

    def free_block(self, block_indices: List[int]):
        for idx in block_indices:
            if idx not in self.used_blocks:
                raise RuntimeError(f"bad block idx, {idx} is not in the used block list.")
            self.free_blocks.append(idx)
            self.used_blocks.remove(idx)


class CacheEngine:
    """allocate a big chunk memory."""
    # pylint: disable=C0326
    def __init__(self, block_size: int, pool: BlockMemPool = None):
        self.block_size = block_size
        self.pool = pool
        self.num_token = 0
        self.block_table = []
        logging.info("use block size: %s", block_size)

    def prepare_cache(self, num_new_token):
        """prepare cache for paged attention."""
        num_blocks = len(self.block_table)
        remained_token = num_blocks * self.block_size - self.num_token

        if remained_token < num_new_token:
            # free block slot is not enough, allocate more blocks.
            num_new_block = (num_new_token - remained_token + self.block_size - 1) // self.block_size
            new_block = self.pool.allocate_block(num_new_block)
            self.block_table += new_block

        # update token num
        self.num_token += num_new_token

    def release_cache(self):
        self.pool.free_block(self.block_table)
        self.block_table = []
        self.num_token = 0
