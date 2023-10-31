# reference to https://github.com/Stability-AI/generative-models

from typing import Dict, List, Optional, Union

import numpy as np
from gm.modules.diffusionmodules.openaimodel import Timestep
from gm.modules.embedders.clip import CLIPTextModel

# OpenCLIP model
from gm.modules.embedders.open_clip import create_model as openclip_create_model
from gm.modules.embedders.open_clip import tokenize as openclip_tokenize
from gm.util import count_params, expand_dims_like, instantiate_from_config
from omegaconf import ListConfig

# CLIP model
from transformers import CLIPTokenizer

import mindspore as ms
from mindspore import Tensor, nn, ops


class AbstractEmbModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    def tokenize(self, x):
        raise NotImplementedError

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key


class GeneralConditioner(nn.Cell):
    def __init__(self, emb_models: Union[List, ListConfig]):
        super().__init__()
        embedders = []
        for n, embconfig in enumerate(emb_models):
            embedder = instantiate_from_config(embconfig)
            assert isinstance(
                embedder, AbstractEmbModel
            ), f"embedder model {embedder.__class__.__name__} has to inherit from AbstractEmbModel"
            embedder.is_trainable = embconfig.get("is_trainable", False)
            embedder.ucg_rate = embconfig.get("ucg_rate", 0.0)
            if not embedder.is_trainable:
                embedder.set_train(False)
                embedder.set_grad(False)
                for _, param in embedder.parameters_and_names():
                    param.requires_grad = False
            print(
                f"Initialized embedder #{n}: {embedder.__class__.__name__} "
                f"with {count_params(embedder, False)} params. Trainable: {embedder.is_trainable}"
            )

            if "input_key" in embconfig:
                embedder.input_key = embconfig["input_key"]
            elif "input_keys" in embconfig:
                embedder.input_keys = embconfig["input_keys"]
            else:
                raise KeyError(f"need either 'input_key' or 'input_keys' for embedder {embedder.__class__.__name__}")

            embedder.legacy_ucg_val = embconfig.get("legacy_ucg_value", None)
            if embedder.legacy_ucg_val is not None:
                embedder.ucg_prng = np.random.RandomState()

            embedders.append(embedder)
        self.embedders = nn.CellList(embedders)

    def possibly_get_ucg_val(self, embedder: AbstractEmbModel, batch: Dict) -> Dict:
        assert embedder.legacy_ucg_val is not None
        p = embedder.ucg_rate
        val = embedder.legacy_ucg_val
        for i in range(len(batch[embedder.input_key])):
            if embedder.ucg_prng.choice(2, p=[1 - p, p]):
                batch[embedder.input_key][i] = val
        return batch

    def tokenize(self, batch: Dict):
        tokens, lengths = [], []
        for embedder in self.embedders:
            if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                if embedder.legacy_ucg_val is not None:
                    batch = self.possibly_get_ucg_val(embedder, batch)
                emb_token, emb_length = embedder.tokenize(batch[embedder.input_key])
            elif hasattr(embedder, "input_keys"):
                emb_token, emb_length = embedder.tokenize(*[batch[k] for k in embedder.input_keys])
            else:
                raise AttributeError("embedder does not have attribute input_key/input_keys.")

            assert isinstance(
                emb_token, (Tensor, np.ndarray, list, tuple)
            ), f"tokens must be Tensor, np.ndarray or a sequence, but got {type(emb_token)}"
            assert isinstance(
                emb_length, (np.ndarray, type(None))
            ), f"length must be np.ndarray or None, but got {type(emb_token)}"

            tokens.append(emb_token)
            lengths.append(emb_length)
        return tokens, lengths

    def embedding(self, *tokens, force_zero_embeddings=None):
        assert len(tokens) == len(self.embedders), (
            f"tokens and self.embedders length is not equal, " f"{len(tokens)}, {len(self.embedders)}"
        )

        vector, crossattn, concat = None, None, None

        if force_zero_embeddings is None:
            force_zero_embeddings = ()
        for i in range(len(self.embedders)):
            embedder = self.embedders[i]
            token = tokens[i]
            token = token if isinstance(token, (list, tuple)) else (token,)
            emb_out = embedder(*token)

            if not isinstance(emb_out, (list, tuple)):
                emb_out = (emb_out,)
            for emb in emb_out:
                if embedder.ucg_rate > 0.0 and embedder.legacy_ucg_val is None:
                    emb = (
                        expand_dims_like(
                            ops.bernoulli((1.0 - embedder.ucg_rate) * ops.ones(emb.shape[0], dtype=emb.dtype)),
                            emb,
                        )
                        * emb
                    )
                if hasattr(embedder, "input_key") and embedder.input_key in force_zero_embeddings:
                    emb = ops.zeros_like(emb)

                # CONCAT
                # OUTPUT_DIM2KEYS = {2: "vector", 3: "crossattn", 4: "concat", 5: "concat"}
                # KEY2CATDIM = {"vector": 1, "crossattn": 2, "concat": 1}
                assert emb.dim() in (2, 3, 4, 5)
                if emb.dim() == 2:  # vector
                    if vector is None:
                        vector = emb
                    else:
                        vector = ops.concat((vector, emb), 1)
                elif emb.dim() == 3:  # crossattn
                    if crossattn is None:
                        crossattn = emb
                    else:
                        crossattn = ops.concat((crossattn, emb), 2)
                else:  # concat
                    if concat is None:
                        concat = emb
                    else:
                        concat = ops.concat((concat, emb), 1)

        return vector, crossattn, concat

    def __call__(self, batch: Dict, force_zero_embeddings: Optional[List] = None) -> Dict:
        # tokenize
        tokens, _ = self.tokenize(batch)
        tokens = [Tensor(t) for t in tokens]

        # embeddings
        vector, crossattn, concat = self.embedding(*tokens, force_zero_embeddings=force_zero_embeddings)
        embeddings_dict = {}
        for k, v in zip(("vector", "crossattn", "concat"), (vector, crossattn, concat)):
            if v is not None:
                embeddings_dict[k] = v

        return embeddings_dict

    def construct(self, *tokens, force_zero_embeddings: Optional[List] = None):
        vector, crossattn, concat = self.embedding(*tokens, force_zero_embeddings=force_zero_embeddings)
        return vector, crossattn, concat

    def get_unconditional_conditioning(self, batch_c, batch_uc=None, force_uc_zero_embeddings=None):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = []
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self(batch_c)
        uc = self(batch_c if batch_uc is None else batch_uc, force_uc_zero_embeddings)

        for embedder, rate in zip(self.embedders, ucg_rates):
            embedder.ucg_rate = rate
        return c, uc


class FrozenCLIPEmbedder(AbstractEmbModel):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        pretrained=None,
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        always_return_pooled=False,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel(config_path="openai/clip-vit-large-patch14", weight=pretrained)

        if freeze:
            self.freeze()

        self.layer = layer
        self.layer_idx = layer_idx
        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer.set_train(False)
        self.transformer.set_grad(False)

        for _, p in self.parameters_and_names():
            p.requires_grad = False

    def tokenize(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
        )
        tokens = np.array(batch_encoding["input_ids"], np.int32)
        length = np.array(batch_encoding["length"], np.int32)
        return tokens, length

    @ms.jit
    def construct(self, tokens):
        (last_hidden_state, pooler_output, hidden_states, attentions) = self.embedding(
            input_ids=tokens, output_hidden_states=(self.layer == "hidden")
        )

        if self.layer == "last":
            z = last_hidden_state
        elif self.layer == "pooled":
            z = pooler_output[:, None, :]
        else:
            z = hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, pooler_output
        return z

    def embedding(self, input_ids, output_hidden_states):
        return self.transformer(input_ids=input_ids, output_hidden_states=output_hidden_states)

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder2(AbstractEmbModel):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = ["pooled", "last", "penultimate"]

    def __init__(
        self,
        arch="ViT-H-14-Text",
        pretrained=None,
        require_pretrained=True,
        max_length=77,
        freeze=True,
        layer="last",
        always_return_pooled=False,
        legacy=True,
    ):
        super().__init__()
        assert layer in self.LAYERS
        self.model = openclip_create_model(arch, pretrained=pretrained, require_pretrained=require_pretrained)

        self.max_length = max_length
        self.return_pooled = always_return_pooled
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()
        self.legacy = legacy

    def freeze(self):
        self.model.set_train(False)
        self.model.set_grad(False)
        for _, p in self.parameters_and_names():
            p.requires_grad = False

    def tokenize(self, text):
        tokens, lengths = openclip_tokenize(text)
        tokens = np.array(tokens, dtype=np.int32)
        lengths = np.array(lengths, dtype=np.int32)
        return tokens, lengths

    @ms.jit
    def construct(self, tokens):
        z = self.encode_with_transformer(tokens)
        if not self.return_pooled and self.legacy:
            return z
        if self.return_pooled:
            assert not self.legacy
            return z[self.layer_idx], z[-1]  # last/penultimate, pooled
        return z[self.layer_idx]

    def encode_with_transformer(self, tokens):
        x = self.model.token_embedding(tokens)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.transpose(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)  # x: last, penultimate
        if self.legacy:
            x = x[self.layer_idx]
            x = self.model.ln_final(x)
            return x
        else:
            # x is a dict and will stay a dict
            o = x[0]  # x["last"]
            o = self.model.ln_final(o)
            pooled = self.pool(o, tokens)
            return x[0], x[1], pooled  # last, penultimate, pooled

    def pool(self, x, tokens):
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        _dtype = x.dtype

        # x = x[ops.arange(x.shape[0]), tokens.argmax(axis=-1)]
        indices = ops.stack((ops.arange(x.shape[0]), tokens.argmax(axis=-1)), axis=-1)
        x = ops.gather_nd(x, indices)

        x = ops.matmul(x, ops.cast(self.model.text_projection, x.dtype)).astype(_dtype)

        return x

    def text_transformer_forward(self, x: Tensor, attn_mask=None):
        penultimate = None
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - 1:
                penultimate = x.transpose(1, 0, 2)  # LND -> NLD
            x = r(x, attn_mask=attn_mask)
        last = x.transpose(1, 0, 2)  # LND -> NLD

        return last, penultimate

    def encode(self, text):
        return self(text)


class ConcatTimestepEmbedderND(AbstractEmbModel):
    """embeds each dimension independently and concatenates them"""

    def __init__(self, outdim):
        super().__init__()
        self.timestep = Timestep(outdim)
        self.outdim = outdim

    @ms.jit
    def construct(self, x):
        if x.ndim == 1:
            x = x[:, None]
        assert len(x.shape) == 2
        b, dims = x.shape[0], x.shape[1]

        # x = rearrange(x, "b d -> (b d)")
        x = x.view(-1)

        emb = self.timestep(x)

        # emb = rearrange(emb, "(b d) d2 -> b (d d2)", b=b, d=dims, d2=self.outdim)
        emb = emb.view(b, dims, self.outdim).view(b, -1)

        return emb

    def tokenize(self, x):
        return x, None


if __name__ == "__main__":
    # 1. check timestep embedder
    cond_model = ConcatTimestepEmbedderND(outdim=256)
    cond_input = Tensor(np.tile(np.array([1024, 1024]), [2, 1]), ms.float16)
    emb_cond = cond_model(cond_input)
    print(f"ConcatTimestepEmbedderND, emb.shape: {emb_cond.shape}, emb.dtype: {emb_cond.dtype}")

    # 2. check clip embedder
    clip_model = FrozenCLIPEmbedder(layer="hidden", layer_idx=11, version="openai/clip-vit-large-patch14")
    ms.amp.auto_mixed_precision(clip_model, "O2")
    tokens, _ = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
    emb1 = clip_model(Tensor(tokens))
    print(f"FrozenCLIPEmbedder, emb.shape: {emb1.shape}, emb.dtype: {emb1.dtype}")

    # 3. check openclip embedder
    open_clip_model = FrozenOpenCLIPEmbedder2(
        arch="ViT-bigG-14-Text",
        freeze=True,
        layer="penultimate",
        always_return_pooled=True,
        legacy=False,
        require_pretrained=False,
    )
    ms.amp.auto_mixed_precision(open_clip_model, "O2")
    tokens, _ = open_clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
    emb2 = open_clip_model(Tensor(tokens))
    if isinstance(emb2, (tuple, list)):
        print(f"FrozenOpenCLIPEmbedder2, emb.shape: {[e.shape for e in emb2]}, emb.dtype: {[e.dtype for e in emb2]}")
    else:
        print(f"FrozenOpenCLIPEmbedder2, emb.shape: {emb2.shape}, emb.dtype: {emb2.dtype}")
