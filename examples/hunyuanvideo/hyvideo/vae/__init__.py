from pathlib import Path

import mindspore as ms

from mindone.safetensors.mindspore import load_file
from mindone.utils.amp import auto_mixed_precision

from ..constants import PRECISION_TO_TYPE, VAE_PATH
from ..utils.helpers import set_model_param_dtype
from .autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .unet_causal_3d_blocks import GroupNorm, MSInterpolate, MSPad


def load_vae(
    type: str = "884-16c-hy",
    precision: str = None,
    sample_size: tuple = None,
    tiling: bool = False,
    path: str = None,
    logger=None,
    checkpoint=None,
    trainable: bool = False,
):
    """the fucntion to load the 3D VAE model

    Args:
        type (str): the type of the 3D VAE model. Defaults to "884-16c-hy".
        precision (str, optional): the precision to load vae. Defaults to None.
        sample_size (tuple, optional): the tiling size. Defaults to None.
        tiling (bool, optional): the tiling mode. Defaults to False.
        path (str, optional): the path to vae. Defaults to None.
        logger (_type_, optional): logger. Defaults to None.
        checkpoint (str, optional): the checkpoint to load vae. Defaults to None and use default path.
        trainable (bool, optional): set vae trainable
    """
    if path is None:
        path = VAE_PATH[type]

    if logger is not None:
        logger.info(f"Loading 3D VAE model ({type}) (trainable={trainable}, tiling={tiling}) from: {path}")
    config = AutoencoderKLCausal3D.load_config(path)
    if sample_size:
        vae = AutoencoderKLCausal3D.from_config(config, sample_size=sample_size)
    else:
        vae = AutoencoderKLCausal3D.from_config(config)
    if checkpoint is None:
        vae_ckpt = Path(path) / "model.safetensors"
        logger.info(f"Load from default checkpoint {vae_ckpt}")
        # assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"

        if vae_ckpt.exists():
            ckpt = load_file(vae_ckpt)
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            if any(k.startswith("vae.") for k in ckpt.keys()):
                ckpt = {k.replace("vae.", ""): v for k, v in ckpt.items() if k.startswith("vae.")}
            vae.load_state_dict(ckpt)
        else:
            print("No vae ckpt is loaded")
    else:
        if isinstance(checkpoint, str):
            logger.info(f"Load from checkpoint {checkpoint}")
            assert checkpoint.exists(), f"The provided checkpoint {checkpoint} does not exist!"
            state_dict = ms.load_checkpoint(checkpoint)
            state_dict = dict(
                [k.replace("autoencoder.", "") if k.startswith("autoencoder.") else k, v] for k, v in state_dict.items()
            )
            vae.load_state_dict(state_dict)
        elif isinstance(checkpoint, dict):
            logger.info("Load from state dictionary")
            vae.load_state_dict(checkpoint)
        else:
            raise ValueError(f"The provided checkpoint {checkpoint} is not a valid checkpoint!")

    spatial_compression_ratio = vae.config.spatial_compression_ratio
    time_compression_ratio = vae.config.time_compression_ratio

    # set mixed precision
    if precision is not None:
        if precision != "fp32":
            dtype = PRECISION_TO_TYPE[precision]
            if dtype == ms.float16:
                custom_fp32_cells = [GroupNorm]
            elif dtype == ms.bfloat16:
                custom_fp32_cells = [MSPad, MSInterpolate]
            else:
                raise ValueError

            # half param to save memory
            if dtype != ms.float32:
                set_model_param_dtype(vae, dtype=dtype)

            # TODO: try 'auto' in ms.amp.auto_mixed_precision
            amp_level = "O2"
            vae = auto_mixed_precision(vae, amp_level=amp_level, dtype=dtype, custom_fp32_cells=custom_fp32_cells)
            logger.info(
                f"Set vae mixed precision to {amp_level} with dtype={dtype}, custom fp32_cells {custom_fp32_cells}"
            )

    vae.set_train(trainable)
    for param in vae.trainable_params():
        param.requires_grad = trainable

    if logger is not None:
        logger.info(f"VAE param dtype: {vae.dtype}")

    if tiling:
        vae.enable_tiling()

    return vae, path, spatial_compression_ratio, time_compression_ratio
