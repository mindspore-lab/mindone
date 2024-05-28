r"""Configs for the VQGAN-3D on the UCF101.

"""

import ml_collections

from videogvt.config import vqgan2d_ucf101_config

VARIANT = "VQGAN/3D"


def get_config(config_str="MAGVIT-V2"):
    """Returns the base experiment configuration."""
    version, *options = config_str.split("-")

    config = vqgan2d_ucf101_config.get_config(config_str)
    config.experiment_name = f"UCF101_{VARIANT}"
    model_class, model_type = VARIANT.split("/")

    # Overall
    config.batch_size = 128
    config.eval_batch_size = config.get_ref("batch_size") // 4
    config.num_training_epochs = 500

    # Dataset.
    del config.num_train_sampled_frames

    # Model: vqvae
    config.model_class = model_class

    config.pretrained_image_model = True  # TODO(Lijun-Yu): 3d perceptual loss

    config.vqgan.model_type = model_type

    config.vqvae.architecture = "3dcnn"
    config.vqvae.video_contains_first_frame = True
    config.vqvae.separate_first_frame_encoding = True
    config.vqvae.channels = 3
    config.vqvae.middle_channles = 18
    config.vqvae.codebook_size = 1024
    config.vqvae.filters = 128
    config.vqvae.upsample = "nearest+conv" # nearest+conv, deconv
    config.vqvae.num_enc_res_blocks = 4
    config.vqvae.num_dec_res_blocks = 4
    config.vqvae.channel_multipliers = (1, 2, 2, 4, 4)
    config.vqvae.temporal_downsample = (True, True, True, False, False)
    config.vqvae.embedding_dim = 256
    config.vqvae.conv_downsample = False
    config.vqvae.deconv_upsample = False

    config.discriminator = ml_collections.ConfigDict()
    config.discriminator.filters = config.vqvae.get_oneway_ref("filters")
    config.discriminator.channel_multipliers = (2, 4, 4, 4, 4)

    # Save memory
    config.vqvae.num_enc_remat_blocks = 0
    config.vqvae.num_dec_remat_blocks = config.vqvae.get_ref("num_enc_remat_blocks")
    config.discriminator.num_remat_blocks = config.vqvae.get_ref("num_enc_remat_blocks")

    # Loss
    config.lr_configs.disc_weight = 0.1
    config.lr_configs.disc_start = 1

    # Pretrained models on ImageNet.
    config.init_from = ml_collections.ConfigDict()
    config.init_from.inflation = "2d->3d"

    # Standalone evaluation.
    if "eval" in options:
        config.eval_only = True
        config.eval_from.checkpoint_path = {
            "B": "gs://magvit/models/ucf_3d_base",
            "L": "gs://magvit/models/ucf_3d_large",
        }[version]
        config.eval_from.step = -1
        config.eval_from.legacy_checkpoint = True

    if "runlocal" in options:
        config.batch_size = 16
        config.num_training_epochs = 10
        # gets a small model for debugging
        config.vqvae.filters = 32
        config.vqvae.embedding_dim = 16
        config.vqvae.num_enc_res_blocks = 1
        config.vqvae.num_dec_res_blocks = 1
        config.discriminator.filters = 1
        config.discriminator.channel_multipliers = (1,)
        config.vqvae.channel_multipliers = (1,)
        config.vqvae.codebook_size = 128

    return config
