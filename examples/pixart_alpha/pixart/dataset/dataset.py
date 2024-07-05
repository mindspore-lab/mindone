import csv
import logging
import os
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LatentDataset:
    def __init__(self, csv_path: str, latent_dir: str, text_emb_dir: str, path_column: str = "dir") -> None:
        logger.info(f"loading annotations from {csv_path} ...")
        with open(csv_path, "r") as csvfile:
            self.dataset = list(csv.DictReader(csvfile))

        self.length = len(self.dataset)
        logger.info(f"Num data samples: {self.length}")

        self.latent_dir = latent_dir
        self.text_emb_dir = text_emb_dir
        self.path_column = path_column

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        row = self.dataset[idx]
        filename = os.path.basename(row[self.path_column])
        filename = os.path.splitext(filename)[0] + ".npz"

        vae_latent_file = os.path.join(self.latent_dir, filename)
        text_emb_file = os.path.join(self.text_emb_dir, filename)

        vae_latent = self.read_vae_latent(vae_latent_file)
        text_emb, text_mask = self.read_text_emb(text_emb_file)
        return vae_latent, text_emb, text_mask

    @staticmethod
    def read_vae_latent(path: str) -> np.ndarray:
        latent_npy = np.load(path)
        mean, std = latent_npy["latent_mean"], latent_npy["latent_std"]
        latent = mean + std * np.random.randn(*mean.shape)
        return latent

    @staticmethod
    def read_text_emb(path: str) -> Tuple[np.ndarray, np.ndarray]:
        emb_npy = np.load(path)
        emb, mask = emb_npy["text_emb"], emb_npy["text_mask"]
        return emb, mask
