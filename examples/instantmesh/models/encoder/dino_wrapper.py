import mindspore as ms
import mindspore.nn as nn

from .dino import ViTModel


class DinoWrapper(nn.Cell):
    """
    Dino v1 wrapper using huggingface transformer implementation.
    """

    def __init__(self, model_name: str, use_recompute: bool = False):
        super().__init__()
        self.model = self._build_dino(model_name)
        self.camera_embedder = nn.SequentialCell(
            nn.Dense(16, self.model.config.hidden_size, has_bias=True),
            nn.SiLU(),
            nn.Dense(self.model.config.hidden_size, self.model.config.hidden_size, has_bias=True),
        )
        if use_recompute:
            self.camera_embedder.recompute()
            self.model.encoder.recompute()
            self.model.layernorm.recompute()  # recompute layernorm causes gram leackage?

    # @ms.jit, for now don't make it graph mode, as the vit encoder output dict will be none weirdly
    def construct(self, images: ms.Tensor, camera: ms.Tensor):  # because img processor only takes np img
        # image: [B, N, C, H, W]
        # camera: [B, N, D]
        # logger.info(f'input np image shape is {images.shape}')
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
        # logger.info(f'emd shape {embeddings.shape}')

        # This resampling of positional embedding uses bicubic interpolation
        outputs = self.model(pixel_values=images, adaln_input=embeddings, interpolate_pos_encoding=True)[0]
        # last_hidden_states = outputs.last_hidden_state
        return outputs

    @staticmethod
    def _build_dino(model_name: str, proxy_error_retries: int = 3, proxy_error_cooldown: int = 5):
        import requests

        try:
            model = ViTModel.from_pretrained(
                model_name,
                mindspore_dtype=ms.float32,
                add_pooling_layer=False,
                local_files_only=True,
                use_safetensors=True,
            )

            return model
        except requests.exceptions.ProxyError as err:
            if proxy_error_retries > 0:
                print(f"Huggingface ProxyError: Retrying in {proxy_error_cooldown} seconds...")
                import time

                time.sleep(proxy_error_cooldown)
                return DinoWrapper._build_dino(model_name, proxy_error_retries - 1, proxy_error_cooldown)
            else:
                raise err
