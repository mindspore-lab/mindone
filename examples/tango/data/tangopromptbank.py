import json

import numpy as np


class TangoPromptBank:
    def __init__(self, data_path="tpb.json", is_train=True):
        self.bins = []
        self.data_path = data_path
        self.is_train = is_train
        self.collect_data()
        super().__init__()

    def collect_data(self):
        with open(self.data_path) as file:
            for line in file.readlines():
                entries = json.loads(line)
                self.bins.append([entries["location"], entries["captions"]])
        print("[TPB] one of data:", self.bins[0])
        if self.is_train:
            np.random.seed(0)
            np.random.shuffle(self.bins)

    def __getitem__(self, index):
        audio_path, text = self.bins[index]
        return audio_path, text

    def __len__(self):
        return len(self.bins)
