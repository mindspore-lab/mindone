import math

import mindspore as ms
from mindspore import nn, ops

from .modules import Conv2_5d, Decoder, Encoder

SDXL_CONFIG = {
    "double_z": True,
    "z_channels": 16,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
    "use_post_quant_conv": True,
    "use_quant_conv": True,
}

# modify based on SD3d5_CONFIG
TAE_CONFIG = {
    "double_z": True,
    "z_channels": 16,
    "resolution": 256,
    "in_channels": 3,
    "out_ch": 3,
    "ch": 128,
    "ch_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_resolutions": [],
    "dropout": 0.0,
    "scaling_factor": 1.5305,
    "shift_factor": 0.0609,
    "use_post_quant_conv": False,
    "use_quant_conv": False,
    "attn_type": "vanilla",
    "temporal_downsample_level": [0, 1, 2],
    "temporal_upsample_level": [3, 2, 1],
}


class TemporalAutoencoder(nn.Cell):
    r"""
    TAE

    Parameters:
        config (`dict`): config dict
        pretrained (`str`): checkpoint path
    """

    def __init__(
        self,
        config: dict = TAE_CONFIG,
        pretrained: str = None,
        use_recompute: bool = False,
        sample_deterministic: bool = False,
        use_tile: bool = False,
        encode_tile: int = 32,
        encode_overlap: int = 0,
        decode_tile: int = 32,
        decode_overlap: int = 16,
    ):
        super().__init__()

        # encoder
        self.encoder = Encoder(**config)

        # quant and post quant
        embed_dim = config["z_channels"]
        if config["use_quant_conv"]:
            self.quant_conv = Conv2_5d(2 * embed_dim, 2 * embed_dim, 1, pad_mode="valid", has_bias=True)
        if config["use_post_quant_conv"]:
            self.post_quant_conv = Conv2_5d(embed_dim, embed_dim, 1, pad_mode="valid", has_bias=True)

        self.use_quant_conv = config["use_quant_conv"]
        self.use_post_quant_conv = config["use_post_quant_conv"]

        # decoder
        self.decoder = Decoder(**config)

        self.exp = ops.Exp()
        self.stdnormal = ops.StandardNormal()
        self.split = ms.ops.split

        self.sample_deterministic = sample_deterministic
        self.discard_spurious_frames = True

        # tile
        self.encode_tile = encode_tile
        self.time_compress = 2 ** len(config["temporal_downsample_level"])  # 8
        self.encode_overlap = encode_overlap
        self.use_tile = use_tile
        if use_tile:
            assert (self.encode_tile % self.time_compress == 0) and (
                self.encode_tile > 0
            ), f"num tile frames should be divisable by {self.time_compress} and non-zero"
            assert (
                self.encode_overlap % self.time_compress == 0
            ), f"overlap frames should be divisable by {self.time_compress}"
            assert self.encode_overlap == 0, "not supported"

        self.decode_tile = decode_tile
        self.decode_overlap = decode_overlap

        # recompute
        if use_recompute:
            self.recompute(self.encoder)
            self.recompute(self.decoder)

        if pretrained is not None:
            self.load_pretrained(pretrained)

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute()
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        else:
            b.add_flags(output_no_recompute=True)

    def _encode(self, x):
        # return latent distribution, N(mean, logvar)
        h = self.encoder(x)
        if self.use_quant_conv:
            moments = self.quant_conv(h)
        else:
            moments = h
        mean, logvar = self.split(moments, moments.shape[1] // 2, 1)

        return mean, logvar

    def sample(self, mean, logvar):
        # sample z from latent distribution
        logvar = ops.clip_by_value(logvar, -30.0, 20.0)
        std = self.exp(0.5 * logvar)
        z = mean + std * self.stdnormal(mean.shape)

        return z

    def encode(self, x: ms.Tensor) -> ms.Tensor:
        r"""
        Encode a batch of videos into latents

        Args:
            x (Tensor): input video tensor of shape (b c t h w)

        Returns:
            z (Tensor): the sampled latent tensor, shape (b z t' h' w')
            posterior_mean (Tensor): mean of latent distribution
            posterior_logvar (Tensor): logvar of latent distribution
        """

        posterior_mean, posterior_logvar = self._encode(x)
        if self.sample_deterministic:
            return posterior_mean
        z = self.sample(posterior_mean, posterior_logvar)

        return z, posterior_mean, posterior_logvar

    def decode(self, z: ms.Tensor, target_num_frames: int = None) -> ms.Tensor:
        r"""
        Decode a batch of latents to videos

        Args:
            x (Tensor): input latent tensor of shape (b z t' h' w')
            target_num_frames (int): target number of frames for output, if None, all the decoded frames will be reserved. \
                    Otherwise, the previous this number of frames will be reserved.

        Returns:
            z (Tensor): the decoded videos of shape (b c t h w)
        """

        if self.use_post_quant_conv:
            z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if target_num_frames is not None:
            dec = dec[:, :, :target_num_frames]

        return dec

    def encode_with_tile(self, x: ms.Tensor) -> ms.Tensor:
        r"""
        Encode a batch of videos into latents with tiling

        Args:
            x (Tensor): input video tensor of shape (b c t h w)

        Returns:
            z (Tensor): the sampled latent tensor, shape (b z t/8 h/8 w/8)
            posterior_mean (Tensor): mean of latent distribution
            posterior_logvar (Tensor): logvar of latent distribution
        """

        tf = self.encode_tile

        z_out, mean, logvar = self.encode(x[:, :, :tf])

        for i in range(tf, x.shape[2], tf):
            z_cur, mean, logvar = self.encode(x[:, :, i : i + tf])
            z_out = ops.cat((z_out, z_cur), axis=2)

        # TODO: merge mean, logvar for different slices for training tae with tile
        return z_out, mean, logvar

    def decode_with_tile(self, z: ms.Tensor, target_num_frames: int = None) -> ms.Tensor:
        r"""
        Decode a batch of latents to videos with tiling

        Args:
            x (Tensor): input latent tensor of shape (b z t' h' w')
            target_num_frames (int): target number of frames for output, if None, all the decoded frames will be reserved. \
                    Otherwise, the previous this number of frames will be reserved.

        Returns:
            z (Tensor): the decoded videos of shape (b c t h w)
        """

        tl = self.decode_tile // self.time_compress  # tile len
        ol = self.decode_overlap // self.time_compress  # overlap len
        stride = tl - ol
        in_len = z.shape[2]
        num_slices = (in_len - tl) // stride + 1
        if (in_len - tl) % stride != 0 and (in_len - tl) + stride < in_len:
            num_slices += 1

        # ms graph mode requires an init x_out
        x_out = self.decode(z[:, :, :tl])

        visited = tl
        i = stride  # start position
        while visited < in_len:
            x_cur = self.decode(z[:, :, i : i + tl])
            x_out = ops.cat((x_out, x_cur), axis=2)

            visited = i + tl
            i += stride

        # linear blend the overlapp part
        if self.decode_overlap > 0:
            x_out = self.blend_slices(x_out, self.decode_tile, self.decode_overlap)

        if target_num_frames is not None:
            x_out = x_out[:, :, :target_num_frames]

        return x_out

    def blend_slices(self, x: ms.Tensor, slice_len=32, overlap_len=16):
        """
        Blend decoded latent slices, used with decode_with_tile

        Args:
            x: (b c t h w) is the concatenation of the decoded slices,
            slice_len: slice length; for decoding, it's the latent tile size mulitplied by temporal upsampling ratio. default is 4*8 for moviegen tae.
            overlap_len: overlap between slices. for decoding, default is 2*8 for movie gen tae

            Note that the length of the last slice can be shorter than slice_len.

        Returns:
            ms.Tensor
        """

        B, C, in_len, H, W = x.shape
        num_slices = math.ceil(in_len / slice_len)
        stride = slice_len - overlap_len

        out_len = ((num_slices - 1) * slice_len) - (num_slices - 2) * overlap_len
        last_slice_len = in_len - (num_slices - 1) * slice_len
        out_len += last_slice_len - overlap_len

        out_tensor = ops.zeros((B, C, out_len, H, W), ms.float32)
        out_cnt = ops.zeros((B, C, out_len, H, W), ms.float32)

        for i in range(num_slices):
            # get the slice form the concatnated latent
            cur_slice = x[:, :, i * slice_len : (i + 1) * slice_len]
            cur_len = cur_slice.shape[2]

            # put the slice into the right position of output tensor
            start = i * stride
            out_tensor[:, :, start : start + cur_len] += cur_slice
            out_cnt[:, :, start : start + cur_len] += 1

        out_tensor = out_tensor / out_cnt

        return out_tensor

    def construct(self, x: ms.Tensor) -> ms.Tensor:
        r"""
        Video reconstruction

        Args:
            x: a batch of videos of shape (b c t h w)

        Returns:
            recons (Tensor): the reconstructed videos of shape (b c t h w)
            z (Tensor): the latent tensor, shape (b z t' h' w')
            posterior_mean (Tensor): mean of latent distribution
            posterior_logvar (Tensor): logvar of latent distribution
        """

        if self.use_tile:
            z, posterior_mean, posterior_logvar = self.encode_with_tile(x)
        else:
            posterior_mean, posterior_logvar = self._encode(x)
            z = self.sample(posterior_mean, posterior_logvar)

        if self.use_tile:
            recons = self.decode_with_tile(z)
        else:
            recons = self.decode(z)

        if self.discard_spurious_frames and (recons.shape[-3] != x.shape[-3]):
            recons = recons[:, :, : x.shape[-3], :, :]

        return recons, z, posterior_mean, posterior_logvar

    def load_pretrained(self, ckpt_path: str):
        if ckpt_path.endswith("safetensors"):
            # load vae parameters from safetensors into my mindspore model
            import safetensors

            ckpt = safetensors.safe_open(ckpt_path, framework="pt")
            state_dict = {}
            for key in ckpt.keys():
                state_dict[key] = ckpt.get_tensor(key)
            raise NotImplementedError
        else:
            param_dict = ms.load_checkpoint(ckpt_path)

            # remove the added prefix in the trained checkpoint
            pnames = list(param_dict.keys())
            for pn in pnames:
                new_pn = pn.replace("autoencoder.", "").replace("_backbone.", "")
                param_dict[new_pn] = param_dict.pop(pn)

            param_not_load, ckpt_not_load = ms.load_param_into_net(self, param_dict, strict_load=True)

            if param_not_load or ckpt_not_load:
                print(f"{param_not_load} in network is not loaded")
                print(f"{ckpt_not_load} in checkpoint is not loaded!")

        print("TAE checkpoint loaded")
