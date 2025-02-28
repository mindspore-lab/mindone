from typing import Optional, Tuple, Union

import numpy as np

from mindspore import nn

from mindone.diffusers.utils.logging import get_logger

logger = get_logger(__name__)


def get_optimizer(
    params_to_optimize,
    optimizer_name: str = "adam",
    learning_rate: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.95,
    beta3: float = 0.98,
    epsilon: float = 1e-8,
    weight_decay: float = 1e-4,
    prodigy_decouple: bool = False,
    prodigy_use_bias_correction: bool = False,
    prodigy_safeguard_warmup: bool = False,
    use_8bit: bool = False,
    use_4bit: bool = False,
    use_torchao: bool = False,
    use_deepspeed: bool = False,
    use_cpu_offload_optimizer: bool = False,
    offload_gradients: bool = False,
) -> nn.Optimizer:
    optimizer_name = optimizer_name.lower()

    # Optimizer creation
    supported_optimizers = ["adam", "adamw"]  # "prodigy", "came"
    if optimizer_name not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {optimizer_name}. Supported optimizers include {supported_optimizers}. Defaulting to `AdamW`."
        )
        optimizer_name = "adamw"

    if optimizer_name == "adamw":
        optimizer_class = nn.optim.AdamWeightDecay

        init_kwargs = {
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    elif optimizer_name == "adam":
        optimizer_class = nn.optim.Adam

        init_kwargs = {
            "learning_rate": learning_rate,
            "beta1": beta1,
            "beta2": beta2,
            "eps": epsilon,
            "weight_decay": weight_decay,
        }

    optimizer = optimizer_class(params_to_optimize, **init_kwargs)

    return optimizer


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    patch_size_t: int = None,
    attention_head_dim: int = 64,
    base_height: int = 480,
    base_width: int = 720,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return numpy.ndarray for MindData, All computations are completed by numpy operators.
    """
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)
    if patch_size_t is None:
        # CogVideoX 1.0
        grid_crops_coords = get_resize_crop_region_for_grid(
            (grid_height, grid_width), base_size_width, base_size_height
        )
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=grid_crops_coords,
            grid_size=(grid_height, grid_width),
            temporal_size=num_frames,
        )
    else:
        # CogVideoX 1.5
        base_num_frames = (num_frames + patch_size_t - 1) // patch_size_t
        freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
            embed_dim=attention_head_dim,
            crops_coords=None,
            grid_size=(grid_height, grid_width),
            temporal_size=base_num_frames,
            grid_type="slice",
            max_size=(base_size_height, base_size_width),
        )

    return freqs_cos, freqs_sin


# Adapted from diffusers.models.embeddings.get_3d_rotary_pos_embed
def get_3d_rotary_pos_embed(
    embed_dim,
    crops_coords,
    grid_size,
    temporal_size,
    theta: int = 10000,
    use_real: bool = True,
    grid_type: str = "linspace",
    max_size: Optional[Tuple[int, int]] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    RoPE for video tokens with 3D structure.

    Args:
    embed_dim: (`int`):
        The embedding dimension size, corresponding to hidden_size_head.
    crops_coords (`Tuple[int]`):
        The top-left and bottom-right coordinates of the crop.
    grid_size (`Tuple[int]`):
        The grid size of the spatial positional embedding (height, width).
    temporal_size (`int`):
        The size of the temporal dimension.
    theta (`float`):
        Scaling factor for frequency computation.
    grid_type (`str`):
        Whether to use "linspace" or "slice" to compute grids.

    Returns:
        `np.ndarray`: positional embedding with shape `(temporal_size * grid_size[0] * grid_size[1], embed_dim/2)`.
    """

    if use_real is not True:
        raise ValueError(" `use_real = False` is not currently supported for get_3d_rotary_pos_embed")

    if grid_type == "linspace":
        start, stop = crops_coords
        grid_size_h, grid_size_w = grid_size
        grid_h = np.linspace(start[0], stop[0], grid_size_h, endpoint=False, dtype=np.float32)
        grid_w = np.linspace(start[1], stop[1], grid_size_w, endpoint=False, dtype=np.float32)
        grid_t = np.linspace(0, temporal_size, temporal_size, endpoint=False, dtype=np.float32)
    elif grid_type == "slice":
        max_h, max_w = max_size
        grid_size_h, grid_size_w = grid_size
        grid_h = np.arange(max_h, dtype=np.float32)
        grid_w = np.arange(max_w, dtype=np.float32)
        grid_t = np.arange(temporal_size, dtype=np.float32)
    else:
        raise ValueError("Invalid value passed for `grid_type`.")

    # Compute dimensions for each axis
    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3

    # Temporal frequencies
    freqs_t = get_1d_rotary_pos_embed(dim_t, grid_t, use_real=True)
    # Spatial frequencies for height and width
    freqs_h = get_1d_rotary_pos_embed(dim_h, grid_h, use_real=True)
    freqs_w = get_1d_rotary_pos_embed(dim_w, grid_w, use_real=True)

    # BroadCast and concatenate temporal and spaial frequencie (height and width) into a 3d ndarray
    def combine_time_height_width(freqs_t, freqs_h, freqs_w):
        freqs_t = np.broadcast_to(
            freqs_t[:, None, None, :], (freqs_t.shape[0], grid_size_h, grid_size_w, freqs_t.shape[1])
        )  # temporal_size, grid_size_h, grid_size_w, dim_t
        freqs_h = np.broadcast_to(
            freqs_h[None, :, None, :], (temporal_size, freqs_h.shape[0], grid_size_w, freqs_h.shape[1])
        )  # temporal_size, grid_size_h, grid_size_2, dim_h
        freqs_w = np.broadcast_to(
            freqs_w[None, None, :, :], (temporal_size, grid_size_h, freqs_w.shape[0], freqs_w.shape[1])
        )  # temporal_size, grid_size_h, grid_size_2, dim_w

        freqs = np.concatenate(
            [freqs_t, freqs_h, freqs_w], axis=-1
        )  # temporal_size, grid_size_h, grid_size_w, (dim_t + dim_h + dim_w)
        freqs = freqs.reshape(
            temporal_size * grid_size_h * grid_size_w, -1
        )  # (temporal_size * grid_size_h * grid_size_w), (dim_t + dim_h + dim_w)
        return freqs

    t_cos, t_sin = freqs_t  # both t_cos and t_sin has shape: temporal_size, dim_t
    h_cos, h_sin = freqs_h  # both h_cos and h_sin has shape: grid_size_h, dim_h
    w_cos, w_sin = freqs_w  # both w_cos and w_sin has shape: grid_size_w, dim_w

    if grid_type == "slice":
        t_cos, t_sin = t_cos[:temporal_size], t_sin[:temporal_size]
        h_cos, h_sin = h_cos[:grid_size_h], h_sin[:grid_size_h]
        w_cos, w_sin = w_cos[:grid_size_w], w_sin[:grid_size_w]

    cos = combine_time_height_width(t_cos, h_cos, w_cos)
    sin = combine_time_height_width(t_sin, h_sin, w_sin)
    return cos, sin


# Adapted from diffusers.models.embeddings.get_1d_rotary_pos_embed
def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=np.float32,  # numpy.float32, numpy.float64 (flux)
):
    r"""
    Precompute the frequency ndarray for complex exponentials (cis) with given dimensions.

    This function calculates a frequency ndarray with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned ndarray contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency ndarray.
        pos (`np.ndarray` or `int`): Position indices for the frequency ndarray. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`numpy.float32` or `numpy.float64`, *optional*, defaults to `numpy.float32`):
            the dtype of the frequency ndarray.
    Returns:
        `np.ndarray`: Precomputed frequency ndarray with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = np.arange(pos)

    theta = theta * ntk_factor
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=freqs_dtype)[: (dim // 2)] / dim)) / linear_factor  # [D/2]
    freqs = np.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = np.cos(freqs).repeat(2, axis=1).astype(np.float32)  # [S, D]
        freqs_sin = np.sin(freqs).repeat(2, axis=1).astype(np.float32)  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio
        freqs_cos = np.concatenate([np.cos(freqs), np.cos(freqs)], axis=-1).astype(np.float32)  # [S, D]
        freqs_sin = np.concatenate([np.sin(freqs), np.sin(freqs)], axis=-1).astype(np.float32)  # [S, D]
        return freqs_cos, freqs_sin
    else:
        raise NotImplementedError("'use_real' in `get_1d_rotary_pos_embed` must be True.")


# Adapted from (THUMD/CogVideo) cogvideo.sat.data_video.pad_last_frame
def pad_last_frame(array: np.ndarray, num_frames: int):
    # T, H, W, C
    if len(array) < num_frames:
        pad_length = num_frames - len(array)
        # Use the last frame to pad instead of zero
        last_frame = array[-1]
        pad_array = np.expand_dims(last_frame, axis=0)
        pad_array = np.broadcast_to(pad_array, shape=(pad_length,) + array.shape[1:])
        padded_array = np.concatenate([array, pad_array], axis=0)
        return padded_array
    else:
        return array[:num_frames]
