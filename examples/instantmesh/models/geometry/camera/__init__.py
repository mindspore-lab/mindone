# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# This code is adapted from https://github.com/TencentARC/InstantMesh to work with MindSpore.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

from mindspore import nn


class Camera(nn.Cell):
    def __init__(self):
        super(Camera, self).__init__()
        pass
