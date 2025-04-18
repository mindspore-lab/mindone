"""manage simple block tables for paged attention."""
from typing import List

import numpy as np
from transformers import logging

from .cache_engine import BlockMemPool, CacheEngine

logger = logging.get_logger(__name__)


class BlockTables:
    """
    The Block Table records on which physical block the key and value of each seq are distributed.
    By dividing the cache of each seq's key and value into fixed size physical blocks,
    each block contains the key and value of several tokens in each sentence.
    Paged Attention obtains the corresponding key and value through the block table and calculates the attention.

    Args:
        num_blocks (int): The count of block.
        block_size (int): The size of block.
        seq_length (int): The seq length.

        Examples:
            >>> num_blocks = 1024
            >>> block_size = 16
            >>> seq_length = 1024
            >>> batch_size = 1
            >>> block_mgr = BlockTables(num_blocks, block_size, seq_length)
            >>> block_mgr.init_cache_engine(batch_size)
    """

    def __init__(self, num_blocks, block_size, seq_length):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.seq_length = seq_length
        self.max_num_blocks_per_seq = self.seq_length // self.block_size
        self.block_mem_pool = BlockMemPool(self.num_blocks, self.block_size)
        self.cache_engines = []

    def init_cache_engine(self, batch_size):
        """Init cache engine, allocate block memory bool."""
        self.cache_engines.clear()
        if batch_size * self.seq_length // self.block_size > self.num_blocks:
            logger.warning(
                "Argument `num blocks` is less than the maximum possible block numbers. "
                "May cause `block pool is out of memory` error. "
                "Please make sure batch_size * seq_length <= block_size * num_blocks. "
            )
        for _ in range(batch_size):
            self.cache_engines.append(CacheEngine(self.block_size, self.block_mem_pool))
        logger.info("init cache engine success.")

    def assemble_pa_full_inputs(self, max_input_length, batch_valid_length: np.array, is_finished: List[bool]):
        """Prepare prefill inputs for Paged Attention."""
        bs = batch_valid_length.shape[0]
        block_tables = []
        for i in range(bs):
            if not is_finished[i]:
                logger.debug("prepare cache for full: %s", batch_valid_length[i])
                self.cache_engines[i].prepare_cache(batch_valid_length[i])
            padded_table = self.cache_engines[i].block_table + [-1] * (
                self.max_num_blocks_per_seq - len(self.cache_engines[i].block_table)
            )
            block_tables.append(padded_table)
        block_tables = np.array(block_tables, dtype=np.int32)

        # new method to generate slot mapping, to improve the performance
        batch_valid_np = -1 * np.ones((bs, max_input_length), dtype=np.int32)
        batch_valid_np[:] = np.arange(max_input_length)
        batch_valid_mask = batch_valid_np.copy()
        for i in range(bs):
            batch_valid_mask[i] = batch_valid_mask[i] < batch_valid_length[i]
        batch_valid_mask = batch_valid_mask.astype(np.bool_)
        valid_index_np = batch_valid_np.copy()
        valid_index_np[batch_valid_mask] = batch_valid_np[batch_valid_mask] // self.block_size
        block_tables_np = -1 * np.ones((bs, max_input_length), dtype=np.int32)
        min_block_table_length = min(block_tables_np.shape[1], block_tables.shape[1])
        block_tables_np[:, :min_block_table_length] = block_tables[:, :min_block_table_length]
        for i in range(bs):
            block_tables_np[i] = block_tables_np[i, valid_index_np[i]]
        block_tables_np[batch_valid_mask] *= self.block_size
        batch_valid_np[batch_valid_mask] %= self.block_size
        block_tables_np[batch_valid_mask] += batch_valid_np[batch_valid_mask]
        slot_mapping = block_tables_np.flatten()
        return block_tables, slot_mapping

    def assemble_pa_inc_inputs(self, batch_valid_length: np.array, is_finished: List[bool]):
        """Prepare incremental inputs for Paged Attention."""
        bs = batch_valid_length.shape[0]

        block_tables = []
        slot_mapping = []
        for i in range(bs):
            if not is_finished[i]:
                logger.debug("prepare cache for inc: %s", batch_valid_length[i])
                self.cache_engines[i].prepare_cache(1)

            block_table = self.cache_engines[i].block_table
            padded_table = block_table + [-1] * (self.max_num_blocks_per_seq - len(self.cache_engines[i].block_table))
            block_tables.append(padded_table)

            curent_idx = batch_valid_length[i] - 1
            index = curent_idx // self.block_size
            if index >= len(block_table):
                index = len(block_table) - 1
            slots = [block_table[index] * self.block_size + curent_idx % self.block_size]
            slot_mapping = slot_mapping + slots
        block_tables = np.array(block_tables, dtype=np.int32)
        slot_mapping = np.array(slot_mapping, dtype=np.int32)
        return block_tables, slot_mapping

    def clear_cache(self):
        for cache_engine in self.cache_engines:
            cache_engine.release_cache()
        logger.info("Clear block table cache engines.")
