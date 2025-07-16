# Copyright 2023 The videogvt Authors.
#
# This code is adapted from https://github.com/google-research/magvit/
# with modifications to run mgvit on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Configs for the VQGAN-3D on the UCF101.

"""

import ml_collections


def get_config():
    """Returns the base experiment configuration."""

    config = ml_collections.ConfigDict()

    # Model: vqvae
    config.vqvae = ml_collections.ConfigDict()
    config.vqvae.channels = 3
    config.vqvae.embedding_dim = 18
    config.vqvae.codebook_size = 262144  # 2^18
    config.vqvae.filters = 128
    config.vqvae.activation_fn = "swish"
    config.vqvae.num_enc_res_blocks = 4
    config.vqvae.num_dec_res_blocks = 4
    config.vqvae.channel_multipliers = (1, 2, 2, 4)
    config.vqvae.spatial_downsample = (True, True, True)
    config.vqvae.temporal_downsample = (True, True, False)
    config.vqvae.num_groups = 32
    config.vqvae.num_frames = 17

    config.discriminator = ml_collections.ConfigDict()
    config.discriminator.filters = config.vqvae.get_oneway_ref("filters")
    config.discriminator.channel_multipliers = (2, 4, 4, 4, 4)

    # Loss
    config.lr_configs = ml_collections.ConfigDict()
    config.lr_configs.perceptual_weight = 0.1
    config.lr_configs.entropy_weight = 0.1
    config.lr_configs.commit_weight = 0.25
    config.lr_configs.recons_weight = 10.0
    config.lr_configs.disc_weight = 0.1
    config.lr_configs.disc_start = 1

    # LFQ
    config.lfq = ml_collections.ConfigDict()
    config.lfq.dim = config.vqvae.embedding_dim
    config.lfq.codebook_size = config.vqvae.codebook_size
    config.lfq.entropy_loss_weight = config.lr_configs.entropy_weight
    config.lfq.commitment_loss_weight = config.lr_configs.commit_weight
    config.lfq.diversity_gamma = 1.0
    config.lfq.straight_through_activation = "identity"
    config.lfq.num_codebooks = 1
    config.lfq.keep_num_codebooks_dim = None
    config.lfq.codebook_scale = 1.0  # for residual LFQ, codebook scaled down by 2x at each layer
    config.lfq.frac_per_sample_entropy = (
        1.0  # make less than 1. to only use a random fraction of the probs for per sample entropy
    )
    config.lfq.inv_temperature = 100.0
    config.lfq.soft_clamp_input_value = None
    config.lfq.cosine_sim_project_in = False
    config.lfq.cosine_sim_project_in_scale = None

    # Pretrained models on ImageNet.
    config.init_from = ml_collections.ConfigDict()
    config.init_from.inflation = "2d->3d"

    return config
