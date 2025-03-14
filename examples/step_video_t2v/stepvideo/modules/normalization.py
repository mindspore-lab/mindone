import math
from typing import Dict, Optional, Tuple

import mindspore as ms
from mindspore import Parameter, Tensor, mint, nn


class RMSNorm(nn.Cell):
    def __init__(
        self,
        dim: int,
        elementwise_affine=True,
        eps: float = 1e-6,
        dtype=None,
    ):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps

        self.weight = None
        if elementwise_affine:
            self.weight = Parameter(mint.ones(dim, dtype=dtype))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (mindspore.Tensor): The input tensor.

        Returns:
            mindspore.Tensor: The normalized tensor.

        """
        return x * mint.rsqrt(x.pow(2).mean(-1, keep_dims=True) + self.eps)

    def construct(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (mindspore.Tensor): The input tensor.

        Returns:
            mindspore.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        # if hasattr(self, "weight"):
        if self.weight is not None:
            output = output * self.weight
        return output


ACTIVATION_FUNCTIONS = {
    "swish": mint.nn.SiLU(),
    "silu": mint.nn.SiLU(),
    "mish": mint.nn.Mish(),
    "gelu": mint.nn.GELU(),
    "relu": mint.nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Cell:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Cell: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert len(timesteps.shape) == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2
    exponent = -math.log(max_period) * mint.arange(start=0, end=half_dim, dtype=ms.float32)
    exponent = exponent / (half_dim - downscale_freq_shift)

    emb = mint.exp(exponent)
    emb = timesteps[:, None].float() * emb[None, :]

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    emb = mint.cat([mint.sin(emb), mint.cos(emb)], dim=-1)

    # flip sine and cosine embeddings
    if flip_sin_to_cos:
        emb = mint.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)

    # zero pad
    if embedding_dim % 2 == 1:
        emb = mint.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


class Timesteps(nn.Cell):
    def __init__(self, num_channels: int, flip_sin_to_cos: bool, downscale_freq_shift: float):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift

    def construct(self, timesteps):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
        )
        return t_emb


class TimestepEmbedding(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()
        linear_cls = mint.nn.Linear

        self.linear_1 = linear_cls(
            in_channels,
            time_embed_dim,
            bias=sample_proj_bias,
        )

        if cond_proj_dim is not None:
            self.cond_proj = linear_cls(
                cond_proj_dim,
                in_channels,
                bias=False,
            )
        else:
            self.cond_proj = None

        self.act = get_activation(act_fn)

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim

        self.linear_2 = linear_cls(
            time_embed_dim,
            time_embed_dim_out,
            bias=sample_proj_bias,
        )

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)

    def construct(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Cell):
    def __init__(self, embedding_dim, size_emb_dim, use_additional_conditions: bool = False):
        super().__init__()

        self.outdim = size_emb_dim
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

        self.use_additional_conditions = use_additional_conditions
        if self.use_additional_conditions:
            self.additional_condition_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=0)
            self.resolution_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=size_emb_dim)
            self.nframe_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
            self.fps_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)

    def construct(self, timestep, resolution=None, nframe=None, fps=None):
        # hidden_dtype = next(self.timestep_embedder.parameters()).dtype
        hidden_dtype = self.timestep_embedder.linear_1.weight.dtype

        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj.to(dtype=hidden_dtype))  # (N, D)

        if self.use_additional_conditions:
            batch_size = timestep.shape[0]
            resolution_emb = self.additional_condition_proj(resolution.flatten(start_dim=0)).to(hidden_dtype)
            resolution_emb = self.resolution_embedder(resolution_emb).reshape(batch_size, -1)
            nframe_emb = self.additional_condition_proj(nframe.flatten(start_dim=0)).to(hidden_dtype)
            nframe_emb = self.nframe_embedder(nframe_emb).reshape(batch_size, -1)
            conditioning = timesteps_emb + resolution_emb + nframe_emb

            if fps is not None:
                fps_emb = self.additional_condition_proj(fps.flatten(start_dim=0)).to(hidden_dtype)
                fps_emb = self.fps_embedder(fps_emb).reshape(batch_size, -1)
                conditioning = conditioning + fps_emb
        else:
            conditioning = timesteps_emb

        return conditioning


class AdaLayerNormSingle(nn.Cell):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(self, embedding_dim: int, use_additional_conditions: bool = False, time_step_rescale=1000):
        super().__init__()

        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim, size_emb_dim=embedding_dim // 2, use_additional_conditions=use_additional_conditions
        )

        self.silu = mint.nn.SiLU()
        self.linear = mint.nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)

        self.time_step_rescale = (
            time_step_rescale  # timestep usually in [0, 1], we rescale it to [0,1000] for stability
        )

    def construct(
        self,
        timestep: Tensor,
        added_cond_kwargs: Dict[str, Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        embedded_timestep = self.emb(timestep * self.time_step_rescale, **added_cond_kwargs)

        out = self.linear(self.silu(embedded_timestep))

        return out, embedded_timestep


class PixArtAlphaTextProjection(nn.Cell):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.linear_1 = mint.nn.Linear(
            in_features,
            hidden_size,
            bias=True,
        )
        # self.act_1 = nn.GELU(approximate="tanh")
        self.linear_2 = mint.nn.Linear(
            hidden_size,
            hidden_size,
            bias=True,
        )

    def construct(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = mint.nn.functional.gelu(hidden_states, approximate="tanh")  # act_1
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
