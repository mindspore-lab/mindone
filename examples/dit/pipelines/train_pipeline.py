from diffusion import create_diffusion

import mindspore as ms
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore import nn, ops


class DiTWithLoss(nn.Cell):
    """

    Args:
        dit (nn.Cell): A `DiT` to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
    """

    def __init__(
        self,
        dit,
        vae,
        text_encoder=None,
        scale_factor=1.0,
        condition="class",
    ):
        super().__init__()
        self.dit = dit
        self.num_classes = self.dit.num_classes
        self.text_encoder = text_encoder
        self.condition = condition
        self.vae = vae
        self.scale_factor = scale_factor
        self.uniform_int = ops.UniformInt()
        self.diffusion = create_diffusion(timestep_respacing="")  # sampling_steps=1000
        self.num_timesteps = 1000

    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        image_latents = image_latents * self.scale_factor
        return image_latents.astype(ms.float16)

    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        b, c, h, w = x.shape

        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def vae_decode_video(self, x):
        """
        Args:
            x: (b f c h w), denoised latent
        Return:
            y: (b f H W 3), batch of images, normalized to [0, 1]
        """
        b, f, c, h, w = x.shape
        x = x.reshape((b * f, c, h, w))

        y = self.vae_decode(x)
        _, h, w, c = y.shape
        y = y.reshape((b, f, h, w, c))

        return y

    def get_latents(self, x):
        # "b f c h w -> (b f) c h w"
        B, F, C, H, W = x.shape
        if C != 3:
            raise ValueError("Expect input shape (b f 3 h w), but get {}".format(x.shape))
        x = ops.reshape(x, (-1, C, H, W))

        z = ops.stop_gradient(self.vae_encode(x))

        # (b*f c h w) -> (b f c h w)
        z = ops.reshape(z, (B, F, z.shape[1], z.shape[2], z.shape[3]))

        return z

    def get_condition_embeddings(self, text_tokens, control=None):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        if self.cond_stage_trainable:
            text_emb = self.text_encoder(text_tokens)
        else:
            text_emb = ops.stop_gradient(self.text_encoder(text_tokens))

        return text_emb

    def construct(self, x: ms.Tensor, text_tokens: ms.Tensor = None, labels: ms.Tensor = None, control=None, **kwargs):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape [bs, F, 3, 256, 256]
            text: text tokens padded to fixed shape [bs, 77]
            control: other conditions for future extension

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume unet3d input/output shape: (b c f h w)
                unet2d input/output shape: (b c h w)
        """

        # 1. get image/video latents z using vae
        z = self.get_latents(x)
        t = self.uniform_int(
            (x.shape[0],), Tensor(0, dtype=mstype.int32), Tensor(self.num_timesteps, dtype=mstype.int32)
        )
        # 2. get conditions
        if text_tokens is not None:
            assert self.condition == "text", "When text tokens inputs are not None, expect the condition type is `text`"
            f"but got {self.condition}!"
            text_embed = self.get_condition_embeddings(text_tokens, control)
        else:
            text_embed = None

        if labels is not None:
            assert self.condition == "class", "When labels inputs are not None, expect the condition type is `class`"
            f"but got {self.condition}!"
            y = labels
        else:
            y = None
        model_kwargs = {"y": y, "text_embed": text_embed}
        # 3.  loss computation
        loss = self.diffusion.training_losses(self.dit, z, t, model_kwargs)["loss"]
        return loss
