import numpy as np

from mindspore import Parameter, Tensor, nn

from ...stable_diffusion_v2.ldm.modules.attention import BasicTransformerBlock
from ...stable_diffusion_v2.ldm.modules.diffusionmodules.openaimodel import UNetModel


def _positional_encoding(length: int, dim: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.

    Args:
        length: The length of the sequence.
        dim: The dimension of the positional encodings.

    Returns:
        A numpy array of shape (length, dim) containing the positional encodings.
    """
    encodings = np.zeros((length, dim))
    positions = np.arange(0, length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, dim, 2) * -(np.log(10000.0) / dim))
    encodings[:, 0::2] = np.sin(positions * div_term)
    encodings[:, 1::2] = np.cos(positions * div_term)
    return encodings


class GroupNorm3D(nn.GroupNorm):
    """
    MindSpore supports (N, C, H, W) input only
    """

    def construct(self, x: Tensor) -> Tensor:
        return super().construct(x.view(x.shape[0], x.shape[1], x.shape[2], -1)).view(x.shape)


class Conv3DLayer(nn.Cell):
    def __init__(self, in_channels: int, out_channels: int, num_frames: int):
        super().__init__()
        self._num_frames = num_frames

        self.conv3d = nn.SequentialCell(
            GroupNorm3D(32, in_channels),
            nn.SiLU(),
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), has_bias=True),
            GroupNorm3D(32, out_channels),
            nn.SiLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), has_bias=True),
        )

        self.alpha = Parameter(1.0)

    def construct(self, x: Tensor, *args) -> Tensor:
        # (b t) c h w -> b c t h w
        h = x.reshape(-1, self._num_frames, x.shape[1], x.shape[2], x.shape[3]).swapaxes(1, 2)
        h = self.conv3d(h)
        h = h.swapaxes(1, 2).reshape(-1, x.shape[1], x.shape[2], x.shape[3])  # b c t h w -> (b t) c h w

        # TODO: limit alpha with no grad
        return self.alpha * x + (1 - self.alpha) * h


class TemporalTransformer(nn.Cell):
    def __init__(
        self,
        in_channels: int,
        num_frames: int,
        n_heads: int = 16,
        d_head: int = 88,
        depth: int = 1,
        dropout: float = 0.0,
        context_dim: int = None,
        enable_flash_attention=False,
    ):
        super().__init__()
        self._num_frames = num_frames

        self.in_channels = in_channels
        inner_dim = n_heads * d_head

        self._pe = Tensor(_positional_encoding(num_frames, inner_dim))

        self.norm = GroupNorm3D(32, in_channels, eps=1e-6)
        self.proj_in = nn.Dense(in_channels, inner_dim)

        # 3. Define transformers blocks
        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    enable_flash_attention=False,   # FIXME: add FA support
                )
                for _ in range(depth)
            ]
        )

        self.proj_out = nn.Dense(inner_dim, in_channels)
        self.alpha = Parameter(1.0)

    def construct(self, x: Tensor, emb=None, context=None):
        residual = x

        # 1. Input
        _, c, h, w = x.shape

        x = x.reshape(-1, self._num_frames, c, h, w).swapaxes(1, 2)  # (b t) c h w -> b c t h w
        x = self.norm(x)
        x = x.transpose(0, 3, 4, 2, 1).reshape(-1, h * w, self._num_frames, c)  # b c t h w -> b (h w) t c

        x = x + self._pe
        x = self.proj_in(x)

        # 2. Blocks
        context = context[:: self._num_frames].expand_dims(1)  # sample context for each video
        for block in self.transformer_blocks:
            x = block(x, context=context)

        # 3. Output
        x = self.proj_out(x)

        x = x.reshape(-1, h, w, self._num_frames, c).permute(0, 3, 4, 1, 2)  # b (h w) t c -> b t c h w
        x = x.reshape(-1, c, h, w)  # b t c h w -> (b t) c h w

        # TODO: limit alpha with no grad
        return self.alpha * residual + (1 - self.alpha) * x


class VideoLDMUNetModel(UNetModel):
    def __init__(self, num_frames=5, **kwargs):
        super().__init__(**kwargs)
        self._temporal_params = []  # temporal parameters names
        self._params_map = {}  # map old parameter names to new names (after injecting 3D layers): {old_name: new_name}

        # inject 3D convolutions and temporal attention into input and output blocks
        for blocks, b_name in zip([self.input_blocks, self.output_blocks], ["input_blocks", "output_blocks"]):
            for b_id in range(len(blocks)):
                names = [block.cls_name for block in blocks[b_id]]
                if "ResBlock" in names and "SpatialTransformer" in names:
                    out_channels = blocks[b_id][0].out_channels
                    conv3d = Conv3DLayer(out_channels, out_channels, num_frames)
                    tt = TemporalTransformer(
                        out_channels,
                        num_frames,
                        n_heads=out_channels // kwargs["num_head_channels"],
                        d_head=kwargs["num_head_channels"],
                        dropout=self.dropout,
                        context_dim=kwargs["context_dim"],
                        depth=kwargs["transformer_depth"],  # TODO: verify the depth
                        enable_flash_attention=kwargs["enable_flash_attention"],
                    )

                    blocks[b_id].insert(1, conv3d)
                    if len(blocks[b_id]) == 3:
                        blocks[b_id].append(tt)
                    else:
                        blocks[b_id].insert(3, tt)

                    self._temporal_params.extend(
                        [name for name, _ in conv3d.parameters_and_names(name_prefix=f"{b_name}.{b_id}.1")]
                    )
                    self._temporal_params.extend(
                        [name for name, _ in tt.parameters_and_names(name_prefix=f"{b_name}.{b_id}.3")]
                    )

                    self._params_map.update(  # Spatial Transformer
                        {
                            f"{b_name}.{b_id}.1." + name: f"{b_name}.{b_id}.2." + name
                            for name, _ in blocks[b_id][2].parameters_and_names()
                        }
                    )
                    for i in range(4, len(blocks[b_id])):  # remaining layers after Temporal Transformer
                        self._params_map.update(
                            {
                                f"{b_name}.{b_id}.{i - 2}." + name: f"{b_name}.{b_id}.{i}." + name
                                for name, _ in blocks[b_id][i].parameters_and_names()
                            }
                        )

    def get_temporal_params(self, prefix: str = "") -> set:
        return {prefix + name for name in self._temporal_params}

    def get_weights_map(self, prefix: str = "") -> dict:
        return {prefix + old_name: prefix + new_name for old_name, new_name in self._params_map.items()}
