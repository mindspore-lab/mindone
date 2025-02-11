import logging

from hyvideo.utils.communications import prepare_parallel_data
from hyvideo.utils.ms_utils import no_grad
from hyvideo.utils.parallel_states import get_sequence_parallel_state, hccl_info

import mindspore as ms
from mindspore import mint, nn, ops

__all__ = ["DiffusionWithLoss"]

logger = logging.getLogger(__name__)


class DiffusionWithLoss(nn.Cell):
    """An training pipeline for diffusion model

    Args:
        model (nn.Cell): A noise prediction model to denoise the encoded image latents.
        vae (nn.Cell): Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        noise_scheduler: (object): A class for noise scheduler, such as DDPM scheduler
        text_encoder / text_encoder_2 (nn.Cell): A text encoding model which accepts token ids and returns text embeddings in shape (T, D).
            T is the number of tokens, and D is the embedding dimension.
        train_with_embed (bool): whether to train with embeddings (no need vae and text encoder to extract latent features and text embeddings)
    """

    def __init__(
        self,
        network: nn.Cell,
        vae: nn.Cell = None,
        text_encoder: nn.Cell = None,
        text_encoder_2: nn.Cell = None,  # not to use yet
        text_emb_cached: bool = True,
        video_emb_cached: bool = False,
        use_image_num: int = 0,
        dtype=ms.float32,
        rank_id: int = 0,
        device_num: int = 1,
        embedded_guidance_scale: float = 6.0,
    ):
        super().__init__()
        # TODO: is set_grad() necessary?
        self.network = network.set_grad()
        self.vae = vae

        self.rank_id = rank_id
        self.device_num = device_num
        self.embedded_guidance_scale = embedded_guidance_scale
        if self.network.guidance_embed:
            assert (
                self.embedded_guidance_scale is not None
            ), "embedded_guidance_scale should be set when using guidance embed"

        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.dtype = dtype

        self.text_emb_cached = text_emb_cached
        self.video_emb_cached = video_emb_cached

        if self.text_emb_cached:
            self.text_encoder = None
            self.text_encoder_2 = None
            logger.info("Train with text embedding inputs")

        self.use_image_num = use_image_num
        self.reduce_t = None if not get_sequence_parallel_state() else ops.AllReduce(group=hccl_info.group)
        self.sp_size = 1 if not get_sequence_parallel_state() else hccl_info.world_size
        self.all_gather = None if not get_sequence_parallel_state() else ops.AllGather(group=hccl_info.group)

    def get_condition_embeddings(self, text_tokens, encoder_attention_mask, index=0):
        # text conditions inputs for cross-attention
        # optional: for some conditions, concat to latents, or add to time embedding
        # use for loop to avoid OOM?
        B, frame, L = text_tokens.shape  # B T+num_images L = b 1+4, L
        text_emb = []
        assert index in [0, 1], "index should be 0 or 1"
        text_encoder = self.text_encoder if index == 0 else self.text_encoder_2
        for i in range(frame):
            t = text_encoder(text_tokens[:, i], encoder_attention_mask[:, i])
            text_emb.append(t)
        text_emb = ops.stack(text_emb, axis=1)
        return text_emb

    def vae_encode(self, x):
        image_latents = self.vae.encode(x)
        return image_latents

    def get_latents(self, x):
        if x.dim() == 5:
            B, C, F, H, W = x.shape
            if C != 3:
                raise ValueError("Expect input shape (b 3 f h w), but get {}".format(x.shape))
            if self.use_image_num == 0:
                z = self.vae_encode(x)  # (b, c, f, h, w)
            else:
                videos, images = x[:, :, : -self.use_image_num], x[:, :, -self.use_image_num :]
                videos = self.vae_encode(videos)  # (b, c, f, h, w)
                # (b, c, f, h, w) -> (b, f, c, h, w) -> (b*f, c, h, w) -> (b*f, c, 1, h, w)
                images = images.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W).unsqueeze(2)
                images = self.vae_encode(images)  # (b*f, c, 1, h, w)
                # (b*f, c, 1, h, w) -> (b*f, c, h, w) -> (b, f, c, h, w) -> (b, c, f, h, w)
                _, c, _, h, w = images.shape
                images = images.squeeze(2).reshape(B, self.use_image_num, c, h, w).permute(0, 2, 1, 3, 4)
                z = mint.cat([videos, images], dim=2)  # b c 16+4, h, w
        else:
            raise ValueError("Incorrect Dimensions of x")
        return z

    def construct(
        self,
        x: ms.Tensor,
        attention_mask: ms.Tensor,
        text_tokens: ms.Tensor,
        encoder_attention_mask: ms.Tensor = None,
        text_tokens_2: ms.Tensor = None,
        encoder_attention_mask_2: ms.Tensor = None,
    ):
        """
        Video diffusion model forward and loss computation for training

        Args:
            x: pixel values of video frames, resized and normalized to shape (b c f+num_img h w)
            attention_mask: the mask for latent features of shape (b t' h' w'), where t' h' w' are the shape of latent features after vae's encoding.
            text_tokens: text tokens padded to fixed shape (B F L) or text embedding of shape (B F L D) if using text embedding cache
            encoder_attention_mask: the mask for text tokens/embeddings of a fixed shape (B F L)

        Returns:
            loss

        Notes:
            - inputs should matches dataloder output order
            - assume model input/output shape: (b c f+num_img h w)
        """
        # 1. get image/video latents z using vae
        x = x.to(self.dtype)
        with no_grad():
            if not self.video_emb_cached:
                x = ops.stop_gradient(self.get_latents(x))

            # 2. get conditions
            if not self.text_emb_cached:
                text_embed = ops.stop_gradient(
                    self.get_condition_embeddings(text_tokens, encoder_attention_mask, index=0)
                )
                if text_tokens_2 is not None:
                    text_embed_2 = ops.stop_gradient(
                        self.get_condition_embeddings(text_tokens_2, encoder_attention_mask_2, index=1)
                    )
                else:
                    text_embed_2 = None
            else:
                text_embed = text_tokens
                if text_tokens_2 is not None:
                    text_embed_2 = text_tokens_2
                else:
                    text_embed_2 = None
        loss = self.compute_loss(
            x, attention_mask, text_embed, encoder_attention_mask, text_embed_2, encoder_attention_mask_2
        )
        return loss

    def compute_loss(
        self, x, attention_mask, text_embed, encoder_attention_mask, text_embed_2, encoder_attention_mask_2
    ):
        use_image_num = self.use_image_num
        bsz = x.shape[0]

        current_step_frame = x.shape[2]
        if get_sequence_parallel_state() and current_step_frame > 1:
            x = self.all_gather(x[None])[0]
            (
                x,
                text_embed,
                text_embed_2,
                attention_mask,
                encoder_attention_mask,
                encoder_attention_mask_2,
                use_image_num,
            ) = prepare_parallel_data(
                x,
                text_embed,
                text_embed_2,
                attention_mask,
                encoder_attention_mask,
                encoder_attention_mask_2,
                use_image_num,
            )
        if get_sequence_parallel_state():
            assert (attention_mask.bool()).all()
            # assert attention_mask is None
            attention_mask = None

        # latte forward input match
        # text embed: (b n_tokens  d) -> (b  1 n_tokens d)
        # text_embed = ops.expand_dims(text_embed, axis=1)
        guidance_expand = (
            ms.Tensor(
                [self.embedded_guidance_scale] * bsz,
                dtype=ms.float32,
            ).to(x.dtype)
            * 1000.0
            if self.embedded_guidance_scale is not None
            else None
        )
        loss = self.network(
            x,
            text_states=text_embed,
            text_mask=encoder_attention_mask,
            text_states_2=text_embed_2,
            guidance=guidance_expand,
        )

        # (b c t h w),
        bsz, c, _, _, _ = loss.shape
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).float().repeat(c, axis=1)  # b t h w -> b c t h w
            attention_mask = attention_mask.reshape(bsz, -1)

        loss = loss.reshape(bsz, -1)
        if attention_mask is not None:
            loss = (loss * attention_mask).sum() / attention_mask.sum()  # mean loss on unpad patches
        else:
            loss = loss.mean()

        return loss
