import functools
import logging
import os
from collections import OrderedDict

import mindspore as ms
from mindspore import nn, ops

from .simple_tokenizer import SimpleTokenizer as _Tokenizer

logger = logging.getLogger(__name__)


# On P1, model extracted from https://huggingface.co/laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K
MODEL_PATH = "https://huggingface.co/laion"
_MODELS = {
    "ViT-L/14": os.path.join(MODEL_PATH, "CLIP-ViT-L-14-DataComp.XL-s13B-b90K", "vit_l14_text.pth"),
    "ViT-B/16": os.path.join(MODEL_PATH, "CLIP-ViT-B-16-DataComp.XL-s13B-b90K", "vit_b16_text.pth"),
}


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def construct(self, x: ms.Tensor):
        orig_type = x.dtype
        ret = super().construct(x)
        return ret.type(orig_type)


class QuickGELU(nn.Cell):
    def construct(self, x: ms.Tensor):
        return x * ops.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Cell):
    def __init__(self, d_model: int, n_head: int, attn_mask: ms.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.SequentialCell(
            OrderedDict(
                [
                    ("c_fc", nn.Dense(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Dense(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: ms.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def construct(self, x: ms.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Cell):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: ms.Tensor = None, checkpoint_num: int = 0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.SequentialCell(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

        self.checkpoint_num = checkpoint_num

    def construct(self, x: ms.Tensor):
        if self.checkpoint_num > 0:
            segments = min(self.checkpoint_num, len(self.resblocks))
            return ms.recompute(self.resblocks, segments, x)
        else:
            return self.resblocks(x)


class CLIP_TEXT(nn.Cell):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        checkpoint_num: int,
    ):
        super().__init__()

        self.context_length = context_length
        self._tokenizer = _Tokenizer()

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            checkpoint_num=checkpoint_num,
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = ms.Parameter(ops.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = ms.Parameter(ops.empty(transformer_width, embed_dim))

    def no_weight_decay(self):
        return {"token_embedding", "positional_embedding"}

    @functools.lru_cache(maxsize=None)
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = ops.empty(self.context_length, self.context_length)
        mask = mask.fill(float("-inf"))
        mask = mask.triu(1)  # zero out the lower diagonal
        return mask

    def tokenize(self, texts, context_length=77, truncate=True):
        """
        Returns the tokenized representation of given input string(s)
        Parameters
        ----------
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize
        context_length : int
            The context length to use; all CLIP models use 77 as the context length
        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
        Returns
        -------
        A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
        """
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self._tokenizer.encode(text) + [eot_token] for text in texts]
        result = ops.zeros(len(all_tokens), context_length, dtype=ms.int)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                if truncate:
                    tokens = tokens[:context_length]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
            result[i, : len(tokens)] = ms.Tensor(tokens)

        return result

    def construct(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[ops.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def clip_text_b16(
    embed_dim=512,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
    checkpoint_num=0,
    pretrained=True,
):
    # raise NotImplementedError
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        checkpoint_num,
    )

    if pretrained:
        if isinstance(pretrained, str) and pretrained != "bert-base-uncased":
            pretrained = _MODELS[pretrained]
        else:
            pretrained = _MODELS["ViT-B/16"]
        logger.info(f"Load pretrained weights from {pretrained}")
        state_dict = ms.load_checkpoint(pretrained)
        if context_length != state_dict["positional_embedding"].shape[0]:
            # assert context_length < state_dict["positional_embedding"].size(0), "Cannot increase context length."
            print(f"Resize positional embedding from {state_dict['positional_embedding'].shape[0]} to {context_length}")
            if context_length < state_dict["positional_embedding"].shape[0]:
                state_dict["positional_embedding"] = state_dict["positional_embedding"][:context_length]
            else:
                state_dict["positional_embedding"] = ops.pad(
                    state_dict["positional_embedding"],
                    (0, 0, 0, context_length - state_dict["positional_embedding"].size(0)),
                    value=0,
                )

        ms.load_param_into_net(model, state_dict)
        model.load_state_dict(state_dict, strict_load=False)
        print(f"Load pretrained weights from {pretrained}.")
        model.set_train(False)
    return model


def clip_text_l14(
    embed_dim=768,
    context_length=77,
    vocab_size=49408,
    transformer_width=768,
    transformer_heads=12,
    transformer_layers=12,
    checkpoint_num=0,
    pretrained=True,
):
    model = CLIP_TEXT(
        embed_dim,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        checkpoint_num,
    )
    if pretrained:
        if isinstance(pretrained, str) and pretrained != "bert-base-uncased":
            pretrained = _MODELS[pretrained]
        else:
            pretrained = _MODELS["ViT-L/14"]
        logger.info(f"Load pretrained weights from {pretrained}")
        state_dict = ms.load_checkpoint(pretrained)
        if context_length != state_dict["positional_embedding"].shape[0]:
            # assert context_length < state_dict["positional_embedding"].size(0), "Cannot increase context length."
            print(f"Resize positional embedding from {state_dict['positional_embedding'].shape[0]} to {context_length}")
            if context_length < state_dict["positional_embedding"].shape[0]:
                state_dict["positional_embedding"] = state_dict["positional_embedding"][:context_length]
            else:
                state_dict["positional_embedding"] = ops.pad(
                    state_dict["positional_embedding"],
                    (0, 0, 0, context_length - state_dict["positional_embedding"].shape[0]),
                    value=0,
                )

        ms.load_param_into_net(model, state_dict)
        print(f"Load pretrained weights from {pretrained}.")
        model.set_train(False)
    return model


def clip_text_l14_336(
    embed_dim=768,
    context_length=77,
    vocab_size=49408,
    transformer_width=768,
    transformer_heads=12,
    transformer_layers=12,
):
    raise NotImplementedError


def build_clip(config):
    model_cls = config.text_encoder.clip_teacher
    model = eval(model_cls)()
    return model
