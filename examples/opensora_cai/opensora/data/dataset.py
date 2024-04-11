import copy
import csv
import logging
import os
import random

import albumentations
import cv2
import imageio
import numpy as np
from decord import VideoReader
from PIL import Image, ImageSequence

import mindspore as ms

logger = logging.getLogger()

class DumpEmbeddingDataet:
    def __init__(
        self,
        sample_size=256,
        sample_n_frames=16,
        space_compress=8,
        time_compress=1,
        vae_embed_dim=4,
        text_embed_dim=4096,
        num_tokens=120,
        dataset_size=200,
    ):
        self.num_tokens = num_tokens 
        self.h = self.w = sample_size // space_compress
        self.t = sample_n_frames // time_compress
        self.dim_vae = vae_embed_dim
        self.dim_text = text_embed_dim

        self.length= dataset_size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            tuple (video_embed, text_embed, text_mask), input to network
                - video: (d t h w) 
                - text: (n_tokens d_t)   
                - text_mask: (n_tokens)
        """
        video_emb = np.random.normal(size=(self.dim_vae, self.t, self.h , self.w)).astype(np.float32)
        
        y_len = random.randint(3, self.num_tokens)
        text_mask = np.zeros(shape=[self.num_tokens]).astype(np.uint8)
        text_mask[:y_len] = np.ones(y_len)
        text_emb = np.random.normal(size=(self.num_tokens, self.dim_text)).astype(np.float32)

        return video_emb, text_emb, text_mask


def create_dataloader(config, batch_size, shuffle=True, device_num=1, rank_id=0):
    dataset = DumpEmbeddingDataet(**config)

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=[
            "video",
            "text",
            "text_mask",
        ],
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=shuffle,
    )

    dl = dataloader.batch(
        batch_size,
        drop_remainder=True,
    )

    return dl

