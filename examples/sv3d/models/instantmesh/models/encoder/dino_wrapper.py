from loguru import logger

import mindspore as ms
import mindspore.nn as nn

# from einops import rearrange
from .dino import ViTModel


class DinoWrapper(nn.Cell):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """

    def __init__(self, model_name: str, freeze: bool = True):
        super().__init__()
        self.model = self._build_dino(model_name)
        self.camera_embedder = nn.SequentialCell(
            nn.Dense(16, self.model.config.hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(self.model.config.hidden_size, self.model.config.hidden_size, has_bias=True),
        )

    def construct(self, images: ms.Tensor, camera: ms.Tensor):  # because img processor only takes np img
        # image: [B, N, C, H, W]
        # camera: [B, N, D]
        logger.info(f"input np image shape is {images.shape}")
        if images.ndim == 5:
            # image = rearrange(image, 'b n c h w -> (b n) c h w')  # NOW ITS ALREADY NCHW
            (B, N, C, H, W) = images.shape
            images = images.reshape(B * N, C, H, W)

        # embed camera
        N = camera.shape[1]
        camera_embeddings = self.camera_embedder(camera)
        # camera_embeddings = rearrange(camera_embeddings, 'b n d -> (b n) d')
        cam_emb_shape = camera_embeddings.shape
        camera_embeddings = camera_embeddings.reshape(cam_emb_shape[0] * cam_emb_shape[1], cam_emb_shape[2])
        embeddings = camera_embeddings
        logger.info(f"emd shape {embeddings.shape}")

        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model(pixel_values=images, adaln_input=embeddings, interpolate_pos_encoding=True)
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests

        try:
            model = ViTModel.from_pretrained(
                model_name,
                mindspore_dtype=ms.float32,
                add_pooling_layer=False,
                local_files_only=True,  # this will make the model name has to be full path
                use_safetensors=True,
            )
            # processor = ViTImageProcessor.from_pretrained(model_name)

            return model
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time

                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err
