import mindspore as ms

from mindone.diffusers.models import AutoencoderKL


class HFVAEWrapper:
    def __init__(self, hfvae="mse"):
        super(HFVAEWrapper, self).__init__()
        self.vae = AutoencoderKL.from_pretrained(hfvae, cache_dir="cache_dir")

    @ms.jit
    def encode(self, x):  # b c h w
        t = 0
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            # b c t h w -> (b t) c h w
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        x = self.vae.encode(x).latent_dist.sample() * (0.18215)
        if t != 0:
            # (b t) c h w -> b c t h w
            _, c, h, w = x.shape
            x = x.reshape(-1, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x

    @ms.jit
    def decode(self, x):
        t = 0
        if x.ndim == 5:
            b, c, t, h, w = x.shape
            # b c t h w -> (b t) c h w
            x = x.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        x = self.vae.decode(x / 0.18215).sample
        if t != 0:
            # (b t) c h w -> b t c h w
            _, c, h, w = x.shape
            x = x.reshape(-1, t, c, h, w).permute(0, 2, 1, 3, 4)
        return x


class SDVAEWrapper:
    def __init__(self):
        super(SDVAEWrapper, self).__init__()
        raise NotImplementedError

    @ms.jit
    def encode(self, x):  # b c h w
        raise NotImplementedError

    @ms.jit
    def decode(self, x):
        raise NotImplementedError
