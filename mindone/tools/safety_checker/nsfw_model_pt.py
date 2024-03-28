"""
PyTorch implementation of NSFW classifier
"""
import torch
from torch import nn


class Normalization(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("variance", torch.ones(shape))

    def forward(self, x):
        return (x - self.mean) / self.variance.sqrt()


class NSFWModelPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = Normalization([768])
        self.linear_1 = nn.Linear(768, 64)
        self.linear_2 = nn.Linear(64, 512)
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 1)
        self.act = nn.ReLU()
        self.act_out = nn.Sigmoid()

    def forward(self, x):
        x = self.norm(x)
        x = self.act(self.linear_1(x))
        x = self.act(self.linear_2(x))
        x = self.act(self.linear_3(x))
        x = self.act_out(self.linear_4(x))
        return x
