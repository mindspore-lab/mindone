import os

from lvdm.modules.encoders.clip import CLIPModel, CLIPTokenizer, parse, support_list
from transformers import T5Tokenizer
from utils.download import download_checkpoint
from utils.utils import count_params

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.transformers import CLIPTextModel, T5EncoderModel

_CKPT_URL = {
    "open_clip_vit_h_14": "https://download.mindspore.cn/toolkits/mindone/videocomposer/model_weights/open_clip_vit_h_14-9bb07a10.ckpt"
}


def load_clip_model(arch, pretrained_ckpt_path, dtype):
    """
    Load CLIP model.

    Args:
        arch (str): Model architecture.
        pretrained_ckpt_path (str): Path of the pretrained checkpoint.
    Returns:
        model (CLIPModel): CLIP model.
    """
    if arch.lower() not in support_list:
        raise ValueError(f"arch {arch} is not supported")
    config_path = support_list[arch.lower()]
    # download the clip model if
    if arch.lower() != "open_clip_vit_h_14":
        raise ValueError(f"currently not support {arch.lower()}")
    if not os.path.exists(pretrained_ckpt_path):
        download_checkpoint(_CKPT_URL[arch.lower()], "model_cache/")
        pretrained_ckpt_path = "model_cache/" + _CKPT_URL[arch.lower()].split("/")[-1]
    if not os.path.exists(pretrained_ckpt_path):
        raise ValueError(
            f"Maybe download failed. Please download it manually from {_CKPT_URL[arch.lower()]} and place it under `model_cache/`"
        )

    config = parse(config_path, pretrained_ckpt_path)
    config.dtype = dtype
    model = CLIPModel(config)
    return model


def load_ckpt_tokenizer(tokenizer_path):
    text_processor = CLIPTokenizer(tokenizer_path, pad_token="!")
    return text_processor


class AbstractEncoder(nn.Cell):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class IdentityEncoder(AbstractEncoder):
    def encode(self, x):
        return x


class ClassEmbedder(nn.Cell):
    def __init__(self, embed_dim, n_classes=1000, key="class", ucg_rate=0.1):
        super().__init__()
        self.key = key
        self.embedding = nn.Embedding(n_classes, embed_dim)
        self.n_classes = n_classes
        self.ucg_rate = ucg_rate

    def construct(self, batch, key=None, disable_dropout=False):
        if key is None:
            key = self.key
        # this is for use in crossattn
        c = batch[key][:, None]
        if self.ucg_rate > 0.0 and not disable_dropout:
            mask = 1.0 - ops.bernoulli(mint.ones_like(c) * self.ucg_rate)
            c = mask * c + (1 - mask) * mint.ones_like(c) * (self.n_classes - 1)
            c = c.long()
        c = self.embedding(c)
        return c

    def get_unconditional_conditioning(self, bs, device="cuda"):
        uc_class = self.n_classes - 1  # 1000 classes --> 0 ... 999, one extra class for ucg (class 1000)
        uc = mint.ones((bs,)) * uc_class
        uc = {self.key: uc}
        return uc


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class FrozenT5Embedder(AbstractEncoder):
    """Uses the T5 transformer encoder for text"""

    def __init__(
        self, version="google/t5-v1_1-large", device="cuda", max_length=77, freeze=True
    ):  # others are google/t5-v1_1-xl and google/t5-v1_1-xxl
        super().__init__()
        self.tokenizer = T5Tokenizer.from_pretrained(version)
        self.transformer = T5EncoderModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length  # TODO: typical value?
        if freeze:
            self.freeze()

    def freeze(self):
        self.transformer.set_train(False)
        # self.train = disabled_train
        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"]
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer.set_train(False)
        # self.train = disabled_train
        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"]
        outputs = self.transformer(input_ids=tokens, output_hidden_states=self.layer == "hidden")
        if self.layer == "last":
            z = outputs.last_hidden_state
        elif self.layer == "pooled":
            z = outputs.pooler_output[:, None, :]
        else:
            z = outputs.hidden_states[self.layer_idx]
        return z

    def encode(self, text):
        return self(text)


class ClipImageEmbedder(nn.Cell):
    def __init__(
        self,
        model,
        pretrained_ckpt_path="open_clip_vit_h_14-9bb07a10.ckpt",
        jit=False,
        antialias=True,
        ucg_rate=0.0,
    ):
        super().__init__()
        # from clip import load as load_clip

        # self.model, _ = load_clip(name=model, jit=jit)
        model = load_clip_model(model, pretrained_ckpt_path, str(self.dtype).lower())

        self.antialias = antialias

        self.mean = ms.Tensor([0.48145466, 0.4578275, 0.40821073])
        self.std = ms.Tensor([0.26862954, 0.26130258, 0.27577711])
        self.ucg_rate = ucg_rate

    def preprocess(self, x: ms.Tensor) -> ms.Tensor:
        x = ops.interpolate(x, (224, 224), mode="bicubic", align_corners=True)
        # normalize to [0,1]
        x = (x + 1.0) / 2.0
        # re-normalize according to clip
        x = (x - self.mean[None, :, None, None]) / self.std[None, :, None, None]
        return x

    def encode(self, x: ms.Tensor) -> ms.Tensor:
        # x should be a CLIP preproceesed tensor
        return self.model.encode_image(x)

    def construct(self, x, no_dropout=False):
        # x is assumed to be in range [-1,1]
        out = self.model.encode_image(self.preprocess(x))
        out = out.to(x.dtype)
        if self.ucg_rate > 0.0 and not no_dropout:
            out = ops.bernoulli((1.0 - self.ucg_rate) * mint.ones(out.shape[0]))[:, None] * out
        return out


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="open_clip_vit_h_14",
        version="laion2b_s32b_b79k",
        pretrained_ckpt_path="open_clip_vit_h_14-9bb07a10.ckpt",
        tokenizer_path="bpe_simple_vocab_16e6.txt.gz",
        max_length=77,
        freeze=True,
        layer="last",
        dtype=ms.float32,
    ):
        super().__init__()
        assert layer in self.LAYERS
        # model, _, _ = open_clip.create_model_and_transforms(
        #     arch, device=torch.device("cpu")
        # )
        self.dtype = dtype
        model = load_clip_model(arch, pretrained_ckpt_path, str(self.dtype).lower())
        del model.visual

        self.model = model

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model.set_train(False)
        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, tokens):
        # tokens = self.tokenizer(text, padding="max_length", max_length=77)["input_ids"]
        # tokens = ms.Tensor(tokens, ms.int32).unsqueeze(0)
        z = self.encode_with_transformer(tokens)
        return z

    def encode_with_transformer(self, tokens):
        x = self.model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: ms.Tensor):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            x = r(x)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedder(ClipImageEmbedder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="open_clip_vit_h_14",
        version="laion2b_s32b_b79k",
        pretrained_ckpt_path="open_clip_vit_h_14-9bb07a10.ckpt",
        max_length=77,
        freeze=True,
        layer="pooled",
        antialias=True,
        ucg_rate=0.0,
        dtype=ms.float32,
    ):
        super().__init__()
        self.dtype = dtype
        model = load_clip_model(arch, pretrained_ckpt_path, str(self.dtype).lower())
        del model.transformer
        self.model = model

        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias

        self.mean = ms.Tensor([0.48145466, 0.4578275, 0.40821073])
        self.std = ms.Tensor([0.26862954, 0.26130258, 0.27577711])
        self.ucg_rate = ucg_rate

    def freeze(self):
        self.model.set_train(False)
        for param in self.parameters():
            param.requires_grad = False

    def construct(self, image, no_dropout=False):
        z = self.encode_with_vision_transformer(image)
        if self.ucg_rate > 0.0 and not no_dropout:
            z = ops.bernoulli((1.0 - self.ucg_rate) * mint.ones(z.shape[0]))[:, None] * z
        return z

    def encode_with_vision_transformer(self, img):
        img = self.preprocess(img)
        x = self.model.visual(img)
        return x

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPImageEmbedderV2(ClipImageEmbedder):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="open_clip_vit_h_14",
        version="laion2b_s32b_b79k",
        pretrained_ckpt_path="open_clip_vit_h_14-9bb07a10.ckpt",
        freeze=True,
        layer="pooled",
        antialias=True,
    ):
        super().__init__()
        model = load_clip_model(arch, pretrained_ckpt_path, str(self.dtype).lower())
        del model.transformer
        self.model = model

        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "penultimate":
            raise NotImplementedError()
            self.layer_idx = 1

        self.antialias = antialias
        self.mean = ms.Tensor([0.48145466, 0.4578275, 0.40821073])
        self.std = ms.Tensor([0.26862954, 0.26130258, 0.27577711])

    def freeze(self):
        self.model.set_train(False)
        for param in self.model.get_parameters():
            param.requires_grad = False

    def construct(self, image, no_dropout=False):
        # image: b c h w
        z = self.encode_with_vision_transformer(image)
        return z

    def encode_with_vision_transformer(self, x):
        x = self.preprocess(x)

        # to patches - whether to use dual patchnorm - https://arxiv.org/abs/2302.01327v1
        if self.model.visual.input_patchnorm:
            # einops - rearrange(x, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)')
            x = x.reshape(
                x.shape[0],
                x.shape[1],
                self.model.visual.grid_size[0],
                self.model.visual.patch_size[0],
                self.model.visual.grid_size[1],
                self.model.visual.patch_size[1],
            )
            x = x.permute(0, 2, 4, 1, 3, 5)
            x = x.reshape(
                x.shape[0],
                self.model.visual.grid_size[0] * self.model.visual.grid_size[1],
                -1,
            )
            x = self.model.visual.patchnorm_pre_ln(x)
            x = self.model.visual.conv1(x)
        else:
            x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = mint.cat(
            [
                self.model.visual.class_embedding.to(x.dtype) + mint.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype),
                x,
            ],
            dim=1,
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.model.visual.patch_dropout(x)
        x = self.model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.model.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        return x


class FrozenCLIPT5Encoder(AbstractEncoder):
    def __init__(
        self,
        clip_version="openai/clip-vit-large-patch14",
        t5_version="google/t5-v1_1-xl",
        clip_max_length=77,
        t5_max_length=77,
    ):
        super().__init__()
        self.clip_encoder = FrozenCLIPEmbedder(clip_version, max_length=clip_max_length)
        self.t5_encoder = FrozenT5Embedder(t5_version, max_length=t5_max_length)
        print(
            f"{self.clip_encoder.__class__.__name__} has {count_params(self.clip_encoder) * 1.e-6:.2f} M parameters, "
            f"{self.t5_encoder.__class__.__name__} comes with {count_params(self.t5_encoder) * 1.e-6:.2f} M params."
        )

    def encode(self, text):
        return self(text)

    def construct(self, text):
        clip_z = self.clip_encoder.encode(text)
        t5_z = self.t5_encoder.encode(text)
        return [clip_z, t5_z]
