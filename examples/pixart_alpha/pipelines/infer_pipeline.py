from typing import Any, Dict, List, Union

from diffusion import create_diffusion
from modules.pixart import PixArt

import mindspore as ms
from mindspore import Tensor, nn, ops

from mindone.models.modules.pos_embed import get_2d_sincos_pos_embed


class PixArtInferPipeline:
    """
    Args:
        network (nn.Cell): A `DiT` to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        scale_factor (float): scale_factor for vae.
        guidance_rescale (float): A higher guidance scale value for noise rescale.
        num_inference_steps: (int): The number of denoising steps.
    """

    def __init__(
        self,
        network: PixArt,
        vae: nn.Cell,
        text_encoder: nn.Cell,
        scale_factor: float = 1.0,
        guidance_rescale: float = 0.0,
        num_inference_steps: int = 50,
        ddim_sampling: bool = True,
        multi_scale: bool = False,
        model_config: Dict[str, Any] = {},
    ):
        super().__init__()
        self.network = network
        self.vae = vae
        self.text_encoder = text_encoder
        self.scale_factor = scale_factor
        self.guidance_rescale = guidance_rescale
        self.diffusion = create_diffusion(str(num_inference_steps))
        self.multi_scale = multi_scale
        self.model_config = model_config
        if ddim_sampling:
            self.sampling_func = self.diffusion.ddim_sample_loop
        else:
            self.sampling_func = self.diffusion.p_sample_loop

        # freeze all components
        self.network.set_train(False)
        for param in self.network.trainable_params():
            param.requires_grad = False

        self.vae.set_train(False)
        for param in self.vae.trainable_params():
            param.requires_grad = False

        self.text_encoder.set_train(False)
        for param in self.text_encoder.trainable_params():
            param.requires_grad = False

    @ms.jit
    def vae_decode(self, x):
        """
        Args:
            x: (b c h w), denoised latent
        Return:
            y: (b H W 3), batch of images, normalized to [0, 1]
        """
        y = self.vae.decode(x / self.scale_factor)
        y = ops.clip_by_value((y + 1.0) / 2.0, clip_value_min=0.0, clip_value_max=1.0)

        # (b 3 H W) -> (b H W 3)
        y = ops.transpose(y, (0, 2, 3, 1))

        return y

    def get_condition_embeddings(self, text: Union[str, List[str]]):
        text_emb, text_mask = self.text_encoder.get_text_embeddings(text)
        return text_emb, text_mask.to(ms.bool_)

    def data_prepare(self, inputs):
        x = inputs["noise"]
        y, mask_y = self.get_condition_embeddings(inputs["y"])
        y_null, mask_y_null = self.get_condition_embeddings(inputs["y_null"])

        if y.shape[0] == 1 and y_null.shape[0] == 1:
            N = x.shape[0]
            y = ops.tile(y, (N, 1, 1))
            mask_y = ops.tile(mask_y, (N, 1))
            y_null = ops.tile(y_null, (N, 1, 1))
            mask_y_null = ops.tile(mask_y_null, (N, 1))

        y = ops.concat([y, y_null], axis=0)
        mask_y = ops.concat([mask_y, mask_y_null], axis=0)
        x_in = ops.concat([x] * 2, axis=0)
        assert y.shape[0] == x_in.shape[0], "shape mismatch!"
        return x_in, y, mask_y

    def extra_args_prepare(self, bs: int):
        height = self.model_config["height"]
        width = self.model_config["width"]
        sample_size = self.model_config["sample_size"]
        hidden_size = self.model_config["hidden_size"]
        patch_size = self.model_config["patch_size"]
        lewei_scale = self.model_config["lewei_scale"]

        csize = Tensor([[height, width]], dtype=ms.float32)
        ar = Tensor([[height / width]], dtype=ms.float32)
        base_size = sample_size // patch_size
        pos_embed = Tensor(
            get_2d_sincos_pos_embed(
                hidden_size,
                nh=height // patch_size // 8,
                nw=width // patch_size // 8,
                scale=lewei_scale,
                base_size=base_size,
            )[None, ...]
        )

        csize = ops.tile(csize, (bs, 1))
        ar = ops.tile(ar, (bs, 1))

        csize = ops.concat([csize] * 2, axis=0)
        ar = ops.concat([ar] * 2, axis=0)
        return dict(csize=csize, ar=ar, pos_embed=pos_embed)

    def __call__(self, inputs):
        """
        args:
            inputs: dict

        return:
            images (b H W 3)
        """
        z, y, mask_y = self.data_prepare(inputs)
        model_kwargs = dict(y=y, mask_y=mask_y, cfg_scale=Tensor(self.guidance_rescale, dtype=ms.float32))
        if self.multi_scale:
            extra_model_kwargs = self.extra_args_prepare(inputs["noise"].shape[0])
            model_kwargs.update(extra_model_kwargs)
        latents = self.sampling_func(
            self.network.construct_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
        )
        latents, _ = latents.chunk(2, axis=0)
        assert latents.dim() == 4, f"Expect to have 4-dim latents, but got {latents.shape}"

        images = self.vae_decode(latents)

        return images
