import logging

import numpy as np
from ldm.models.clip.simple_tokenizer import get_tokenizer
from ldm.modules.diffusionmodules.openaimodel import Timestep
from ldm.modules.diffusionmodules.upscaling import ImageConcatWithNoiseAugmentation

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

from .image_encoder import ImageEncoder
from .text_encoder import OpenClipTextEncoder, TextEncoder

_logger = logging.getLogger(__name__)


class FrozenCLIPEmbedder(nn.Cell):
    def __init__(
        self,
        use_fp16=False,
        tokenizer_name="WordpieceTokenizer",
        context_length=77,
        vocab_size=49408,
        output_dim=768,
        width=768,
        layers=12,
        heads=12,
        epsilon=1e-5,
        use_quick_gelu=False,
        upcast_attn=False,
    ):
        super(FrozenCLIPEmbedder, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.context_length = context_length
        self.tokenizer = get_tokenizer(tokenizer_name)
        setattr(self.tokenizer, "context_length", context_length)

        self.transformer = TextEncoder(
            context_length=context_length,
            vocab_size=vocab_size,
            output_dim=output_dim,
            width=width,
            layers=layers,
            heads=heads,
            epsilon=epsilon,
            use_quick_gelu=use_quick_gelu,
            dtype=self.dtype,
            upcast_attn=upcast_attn,
        )

    def tokenize(self, texts):
        SOT_TEXT = self.tokenizer.sot_text
        EOT_TEXT = self.tokenizer.eot_text
        CONTEXT_LEN = self.context_length

        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = np.zeros((len(all_tokens), CONTEXT_LEN), np.int64)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > CONTEXT_LEN:
                tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]

            result[i, : len(tokens)] = np.array(tokens, np.int64)

        return Tensor(result)

    def encode(self, tokenized_text):
        outputs = self.transformer(tokenized_text)
        return outputs

    def construct(self, c):
        outputs = self.transformer(c)
        return outputs


class FrozenOpenCLIPEmbedder(FrozenCLIPEmbedder):
    def __init__(
        self,
        use_fp16=False,
        tokenizer_name="WordpieceTokenizer",
        context_length=77,
        vocab_size=49408,
        output_dim=768,
        width=768,
        layers=12,
        heads=12,
        upcast_attn=False,
    ):
        super(FrozenCLIPEmbedder, self).__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.context_length = context_length
        self.tokenizer = get_tokenizer(tokenizer_name)
        setattr(self.tokenizer, "context_length", context_length)

        self.model = OpenClipTextEncoder(
            context_length=context_length,
            vocab_size=vocab_size,
            output_dim=output_dim,
            width=width,
            layers=layers,
            heads=heads,
            epsilon=1e-5,
            use_quick_gelu=False,
            dtype=self.dtype,
            upcast_attn=upcast_attn,
        )

    def encode(self, tokenized_text):
        outputs = self.model(tokenized_text)
        return outputs

    def construct(self, c):
        outputs = self.model(c)
        return outputs


class CLIPImageEmbedder(nn.Cell):
    def __init__(
        self,
        use_fp16=False,
        embed_dim=1024,
        image_resolution=224,
        vision_layers=32,
        vision_width=1024,
        vision_patch_size=14,
        vision_head_width=64,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.use_fp16 = use_fp16
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.model = ImageEncoder(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            vision_head_width=vision_head_width,
            epsilon=1e-5,
            use_quick_gelu=True,
            mlp_ratio=mlp_ratio,
            dtype=self.dtype,
        )

        self.mean = ms.Tensor([0.48145466, 0.4578275, 0.40821073], dtype=self.dtype)
        self.std = ms.Tensor([0.26862954, 0.26130258, 0.27577711], dtype=self.dtype)

    def preprocess(self, x: Tensor) -> Tensor:
        x = ops.interpolate(x, (224, 224), mode="bicubic", align_corners=True)
        # normalize to [0,1]
        x = (x + 1.0) / 2.0
        # re-normalize according to clip
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return x

    def encode(self, x: Tensor) -> Tensor:
        # x should be a CLIP preproceesed tensor
        return self.model.encode_image(x)

    def construct(self, x: Tensor) -> Tensor:
        # x should be a normalzized tensor with range (-1, 1)
        x = self.preprocess(x)
        out = self.model.encode_image(x)
        return out


class FrozenOpenCLIPImageEmbedder(CLIPImageEmbedder):
    def __init__(
        self,
        use_fp16=False,
        embed_dim=1024,
        image_resolution=224,
        vision_layers=32,
        vision_width=1024,
        vision_patch_size=14,
        vision_head_width=64,
        mlp_ratio=4.0,
    ):
        super(CLIPImageEmbedder, self).__init__()
        self.use_fp16 = use_fp16
        self.dtype = ms.float16 if use_fp16 else ms.float32
        self.model = ImageEncoder(
            embed_dim=embed_dim,
            image_resolution=image_resolution,
            vision_layers=vision_layers,
            vision_width=vision_width,
            vision_patch_size=vision_patch_size,
            vision_head_width=vision_head_width,
            epsilon=1e-5,
            use_quick_gelu=False,
            mlp_ratio=mlp_ratio,
            dtype=self.dtype,
        )

        self.mean = ms.Tensor([0.48145466, 0.4578275, 0.40821073], dtype=self.dtype)
        self.std = ms.Tensor([0.26862954, 0.26130258, 0.27577711], dtype=self.dtype)


class CLIPEmbeddingNoiseAugmentation(ImageConcatWithNoiseAugmentation):
    def __init__(self, *args, clip_stats_path=None, timestep_dim=1024, use_fp16=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = ms.float16 if use_fp16 else ms.float32

        if clip_stats_path is None:
            clip_mean, clip_std = ops.zeros(timestep_dim), ops.ones(timestep_dim)
        else:
            _logger.info(f"Loading CLIP stats from {clip_stats_path}")
            clip = ms.load_checkpoint(clip_stats_path)
            clip_mean, clip_std = clip["mean"], clip["std"]

        self.data_mean = clip_mean[None, :]
        self.data_std = clip_std[None, :]
        self.time_embed = Timestep(timestep_dim).to_float(self.dtype)

    def scale(self, x):
        # re-normalize to centered mean and unit variance
        x = (x - self.data_mean) * 1.0 / self.data_std
        return x

    def unscale(self, x):
        # back to original data stats
        x = (x * self.data_std) + self.data_mean
        return x

    def construct(self, x, noise_level=None):
        if noise_level is None:
            noise_level = ms.numpy.randint(0, self.max_noise_level, (x.shape[0],), dtype=ms.int32)
        x = self.scale(x)
        z = self.q_sample(x, noise_level)
        z = self.unscale(z)
        noise_level = self.time_embed(noise_level)
        z = ops.cast(z, self.dtype)
        noise_level = ops.cast(noise_level, self.dtype)
        return z, noise_level
