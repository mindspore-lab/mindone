# Reference to https://github.com/mlfoundations/open_clip

from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple, Union

import numpy as np
from sgm.modules.util import normalize as normalize_func

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

from .modified_resnet import ModifiedResNet
from .transformer import LayerNormFp32, TextTransformer, VisionTransformer


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224

    ls_init_value: Optional[float] = None  # layer scale initial value
    # whether to use dual patchnorm - would only apply the input layernorm on each patch,
    # as post-layernorm already exist in original clip vit design
    input_patchnorm: bool = False
    # whether to global average pool the last embedding layer,
    # instead of using CLS token (https://arxiv.org/abs/2205.01580)
    global_average_pool: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    hf_model_pretrained: bool = True
    proj: str = "mlp"
    pooler_type: str = "mean_pooler"
    embed_cls: bool = False
    pad_id: int = 0


def _build_vision_tower(embed_dim: int, vision_cfg: CLIPVisionCfg, cast_dtype=None):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    act_layer = partial(nn.GELU, False)

    if isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (ms.float16,) else nn.LayerNorm
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    return visual


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
    cast_dtype=None,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    act_layer = partial(nn.GELU, False)
    norm_layer = LayerNormFp32 if cast_dtype in (ms.float16,) else nn.LayerNorm

    text = TextTransformer(
        context_length=text_cfg.context_length,
        vocab_size=text_cfg.vocab_size,
        width=text_cfg.width,
        heads=text_cfg.heads,
        layers=text_cfg.layers,
        ls_init_value=text_cfg.ls_init_value,
        output_dim=embed_dim,
        embed_cls=text_cfg.embed_cls,
        pad_id=text_cfg.pad_id,
        act_layer=act_layer,
        norm_layer=norm_layer,
    )

    return text


class CLIP(nn.Cell):
    def __init__(
        self,
        embed_dim: int,
        text_cfg: CLIPTextCfg,
        vision_cfg: CLIPVisionCfg = None,
        cast_dtype=None,
    ):
        super().__init__()
        self.visual = _build_vision_tower(embed_dim, vision_cfg, cast_dtype) if vision_cfg is not None else None

        text = _build_text_tower(embed_dim, text_cfg, cast_dtype)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection

        self.attn_mask = Tensor(text.attn_mask)

        self.logit_scale = Parameter(Tensor(np.log(1 / 0.07), ms.float32))

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        if normalize:
            features = normalize_func(features, dim=-1)
        return features

    def encode_text(self, text, normalize: bool = False):
        cast_dtype = ms.float32
        x = self.token_embedding(text).astype(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.astype(cast_dtype)
        x = x.transpose(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.transpose(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)

        # x = x[ops.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        x = x[ops.arange(x.shape[0]), text.argmax(dim=-1)]
        x = ops.matmul(x, self.text_projection)

        if normalize:
            x = normalize_func(x, dim=-1)

        return x

    def construct(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Tensor] = None,
    ):
        image_features, text_features = None, None

        if image is not None:
            image_features = self.encode_image(image, normalize=True)
        if text is not None:
            text_features = self.encode_text(text, normalize=True)

        return image_features, text_features, self.logit_scale.exp()

    def get_input_embeddings(self) -> nn.Cell:
        return self.token_embedding

    def set_input_embeddings(self, value):
        self.token_embedding = value

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        """
        Resizes input token embeddings matrix of the `CLIPTextTransformer` if `new_num_tokens != config.vocab_size`.

        Arguments:
            new_num_tokens (`int`, *optional*):
                The number of new tokens in the embedding matrix. Increasing the size will add newly initialized
                vectors at the end. Reducing the size will remove vectors from the end. If not provided or `None`, just
                returns a pointer to the input tokens `torch.nn.Embedding` module of the model without doing anything.

        Return:
            `torch.nn.Embedding`: Pointer to the input tokens Embeddings Module of the model.
        """
        model_embeds = self._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.vocab_size = model_embeds.embedding_table.shape[0]

        return model_embeds

    def _resize_token_embeddings(self, new_num_tokens) -> nn.Embedding:
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

        return self.get_input_embeddings()

    def _get_resized_embeddings(
        self,
        old_embeddings,
        new_num_tokens: Optional[int] = None,
    ) -> nn.Embedding:
        """Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``mindspore.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.embedding_table.shape
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        old_dtype = old_embeddings.embedding_table.dtype
        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim, dtype=old_dtype)

        # # initialize all new embeddings (in particular added tokens)
        # self._init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.embedding_table.data[:num_tokens_to_copy, :] = old_embeddings.embedding_table.data[
            :num_tokens_to_copy, :
        ]

        # align the parameter status
        old_name = old_embeddings.embedding_table.name
        old_requires_grad = old_embeddings.embedding_table.requires_grad
        new_embeddings.embedding_table.name = old_name
        new_embeddings.embedding_table.requires_grad = old_requires_grad
        return new_embeddings
