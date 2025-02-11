# reference to https://github.com/Stability-AI/generative-models
from functools import partial
from json import load as json_load
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from gm.modules.diffusionmodules.openaimodel import Timestep
from gm.modules.embedders.chinese_clip import BertConfig, BertModel
from gm.modules.embedders.clip import CLIPTextModel

# OpenCLIP model
from gm.modules.embedders.open_clip import create_model as openclip_create_model
from gm.modules.embedders.open_clip import lpw_tokenize2 as lpw_openclip_tokenize2
from gm.modules.embedders.open_clip import tokenize as openclip_tokenize
from gm.util import count_params, expand_dims_like, instantiate_from_config
from omegaconf import ListConfig

# CLIP & Chinese-CLIP model
from transformers import BertTokenizer, CLIPTokenizer

import mindspore as ms
from mindspore import Tensor, nn, ops


class AbstractEmbModel(nn.Cell):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None

    def tokenize(self, x, lpw=False, max_embeddings_multiples=4):
        return x, None

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

    def freeze(self):
        self.set_train(False)
        self.set_grad(False)
        for _, p in self.parameters_and_names():
            p.requires_grad = False


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
            else:
                if hasattr(embedder, "set_recompute"):
                    embedder.set_recompute()
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

    def tokenize(self, batch: Dict, lpw=False, max_embeddings_multiples=4):
        tokens, lengths = [], []
        for embedder in self.embedders:
            if hasattr(embedder, "input_key") and (embedder.input_key is not None):
                if embedder.legacy_ucg_val is not None:
                    batch = self.possibly_get_ucg_val(embedder, batch)
                emb_token, emb_length = embedder.tokenize(
                    batch[embedder.input_key], lpw=lpw, max_embeddings_multiples=max_embeddings_multiples
                )
            elif hasattr(embedder, "input_keys"):
                emb_token, emb_length = embedder.tokenize(
                    *[batch[k] for k in embedder.input_keys], lpw=lpw, max_embeddings_multiples=max_embeddings_multiples
                )
            else:
                raise AttributeError("embedder does not have attribute input_key/input_keys.")

            assert isinstance(
                emb_token, (Tensor, np.ndarray, list, tuple, type(None))
            ), f"tokens must be Tensor, np.ndarray, a sequence or None, but got {type(emb_token)}"
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

                if not embedder.is_trainable:
                    emb = ops.stop_gradient(emb)

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
                        if crossattn.shape[1] == emb.shape[1]:
                            crossattn = ops.concat((crossattn, emb), 2)
                        else:
                            # for image/text emb fusion
                            if emb.shape[0] == 1:
                                emb = ops.tile(emb, (crossattn.shape[0], 1, 1))
                            crossattn = ops.concat((crossattn, emb), 1)
                else:  # concat
                    if concat is None:
                        concat = emb
                    else:
                        concat = ops.concat((concat, emb), 1)

        return vector, crossattn, concat

    def tokenize_embedding(
        self, batch: Dict, force_zero_embeddings: Optional[List] = None, lpw=False, max_embeddings_multiples=4
    ) -> Dict:
        # tokenize
        tokens, _ = self.tokenize(batch, lpw=lpw, max_embeddings_multiples=max_embeddings_multiples)
        tokens = [Tensor(t) if t is not None else t for t in tokens]

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

    def get_unconditional_conditioning(
        self, batch_c, batch_uc=None, force_uc_zero_embeddings=None, lpw=False, max_embeddings_multiples=4
    ):
        if force_uc_zero_embeddings is None:
            force_uc_zero_embeddings = []
        ucg_rates = []
        for embedder in self.embedders:
            ucg_rates.append(embedder.ucg_rate)
            embedder.ucg_rate = 0.0
        c = self.tokenize_embedding(batch_c, lpw=lpw, max_embeddings_multiples=max_embeddings_multiples)
        uc = self.tokenize_embedding(
            batch_c if batch_uc is None else batch_uc,
            force_uc_zero_embeddings,
            lpw=lpw,
            max_embeddings_multiples=max_embeddings_multiples,
        )

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
        self.transformer = CLIPTextModel(config_path=version, weight=pretrained)

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

    def tokenize(self, text, lpw=False, max_embeddings_multiples=4):
        if lpw:
            tokens, length = self.get_text_index(self.tokenizer, text, max_embeddings_multiples)
        else:
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

    def get_unweighted_text_embeddings_SDXL1(self, tokens):
        max_embeddings_multiples = (tokens.shape[1] - 2) // (self.max_length - 2)
        if max_embeddings_multiples > 1:
            last_hidden_state_all = []
            hidden_states_all = []
            last_hidden_state, pooler_output, hidden_states = None, None, None
            for i in range(max_embeddings_multiples):
                # extract the i-th chunk
                text_input_chunk = tokens[:, i * (self.max_length - 2) : (i + 1) * (self.max_length - 2) + 2].copy()
                # cover the head and the tail by the starting and the ending tokens
                text_input_chunk[:, 0] = tokens[0, 0]
                text_input_chunk[:, -1] = tokens[0, -1]
                (last_hidden_state, pooler_output, hidden_states, attentions) = self.embedding(
                    input_ids=text_input_chunk, output_hidden_states=(self.layer == "hidden")
                )
                # no_boseos_middle
                if i == 0:
                    # discard the ending token
                    last_hidden_state = last_hidden_state[:, :-1]
                    if self.layer == "hidden":
                        hidden_states = hidden_states[self.layer_idx][:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    last_hidden_state = last_hidden_state[:, 1:]
                    if self.layer == "hidden":
                        hidden_states = hidden_states[self.layer_idx][:, 1:]
                else:
                    # discard both starting and ending tokens
                    last_hidden_state = last_hidden_state[:, 1:-1]
                    if self.layer == "hidden":
                        hidden_states = hidden_states[self.layer_idx][:, 1:-1]
                last_hidden_state_all.append(last_hidden_state)
                if self.layer == "hidden":
                    hidden_states_all.append(hidden_states)
            last_hidden_state = ops.concat(last_hidden_state_all, axis=1)
            if self.layer == "hidden":
                hidden_states = ops.concat(hidden_states_all, axis=1)
            pooler_output = last_hidden_state[
                ops.arange(last_hidden_state.shape[0]),
                tokens.argmax(axis=-1),
            ]
        else:
            (last_hidden_state, pooler_output, hidden_states, attentions) = self.embedding(
                input_ids=tokens, output_hidden_states=(self.layer == "hidden")
            )
        return last_hidden_state, pooler_output, hidden_states, max_embeddings_multiples

    @ms.jit
    def construct(self, tokens):
        (
            last_hidden_state,
            pooler_output,
            hidden_states,
            max_embeddings_multiples,
        ) = self.get_unweighted_text_embeddings_SDXL1(tokens)
        if self.layer == "last":
            z = last_hidden_state
        elif self.layer == "pooled":
            z = pooler_output[:, None, :]
        else:
            z = hidden_states if max_embeddings_multiples > 1 else hidden_states[self.layer_idx]
        if self.return_pooled:
            return z, pooler_output
        return z

    def embedding(self, input_ids, output_hidden_states):
        return self.transformer(input_ids=input_ids, output_hidden_states=output_hidden_states)

    def encode(self, text):
        return self(text)

    def set_recompute(self):
        self.transformer.text_model.embeddings.recompute()
        for i in range(len(self.transformer.text_model.encoder.layers)):
            if i != 7:
                self.transformer.text_model.encoder.layers[i].recompute()
        # self.transformer.text_model.final_layer_norm.recompute()

    def pad_tokens(self, tokens, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
        r"""
        Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
        """
        for i in range(len(tokens)):
            tokens[i] = [bos] + tokens[i] + [pad] * (max_length - 1 - len(tokens[i]) - 1) + [eos]

        return tokens

    def get_text_index(
        self,
        tokenizer,
        prompt: Union[str, List[str]],
        max_embeddings_multiples: Optional[int] = 4,
        no_boseos_middle: Optional[bool] = False,
    ):
        max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
        if isinstance(prompt, str):
            prompt = [prompt]

        prompt_tokens = [token[1:-1] for token in tokenizer(prompt, max_length=max_length, truncation=True).input_ids]
        prompt_tokens_length = np.array([len(p) + 2 for p in prompt_tokens], np.int32)
        # round up the longest length of tokens to a multiple of (model_max_length - 2)

        # max_length = max([len(token) for token in prompt_tokens])
        # max_embeddings_multiples = min(
        #     max_embeddings_multiples,
        #     (max_length - 1) // (tokenizer.model_max_length - 2) + 1,
        # )
        # max_embeddings_multiples = max(1, max_embeddings_multiples)
        max_length = (tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

        # pad the length of tokens and weights
        bos = tokenizer.bos_token_id
        eos = tokenizer.eos_token_id
        pad = getattr(tokenizer, "pad_token_id", eos)
        prompt_tokens = self.pad_tokens(
            prompt_tokens,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=tokenizer.model_max_length,
        )
        prompt_tokens = np.array(prompt_tokens, np.int32)
        return prompt_tokens, prompt_tokens_length


class FrozenCLIPEmbedder_lora(FrozenCLIPEmbedder):
    """Lora injection to clip embedder."""

    def __init__(self, *, lora_dim=4, lora_alpha=None, lora_dropout=0.0, lora_merge_weights=True, **kwargs):
        super(FrozenCLIPEmbedder_lora, self).__init__(**kwargs)
        from gm.modules.embedders.clip import CLIPAttention
        from gm.modules.lora import Dense as Dense_lora
        from gm.modules.lora import mark_only_lora_as_trainable

        for cell_name, cell in self.cells_and_names():
            if isinstance(cell, CLIPAttention):
                assert hasattr(cell, "k_proj")
                query_dim, inner_dim = cell.k_proj.in_channels, cell.k_proj.out_channels
                cell.k_proj = Dense_lora(
                    query_dim,
                    inner_dim,
                    r=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                _ = [_ for _ in map(partial(self._prefix_param, cell_name), cell.k_proj.get_parameters())]

                assert hasattr(cell, "v_proj")
                context_dim, inner_dim = cell.v_proj.in_channels, cell.v_proj.out_channels
                cell.v_proj = Dense_lora(
                    context_dim,
                    inner_dim,
                    r=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                _ = [_ for _ in map(partial(self._prefix_param, cell_name), cell.v_proj.get_parameters())]

                assert hasattr(cell, "q_proj")
                context_dim, inner_dim = cell.q_proj.in_channels, cell.q_proj.out_channels
                cell.q_proj = Dense_lora(
                    context_dim,
                    inner_dim,
                    r=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                _ = [_ for _ in map(partial(self._prefix_param, cell_name), cell.q_proj.get_parameters())]

                assert hasattr(cell, "out_proj")
                inner_dim, query_dim = cell.out_proj.in_channels, cell.out_proj.out_channels
                cell.out_proj = Dense_lora(
                    inner_dim,
                    query_dim,
                    r=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                _ = [_ for _ in map(partial(self._prefix_param, cell_name), cell.out_proj.get_parameters())]

        mark_only_lora_as_trainable(self, bias="none")

        num_param = sum([p.size for _, p in self.parameters_and_names()])
        num_param_trainable = sum([p.size for p in self.trainable_params()])
        print(
            f"FrozenCLIPEmbedder_lora total params: {float(num_param) / 1e9}B, "
            f"trainable params: {float(num_param_trainable) / 1e6}M."
        )

    @staticmethod
    def _prefix_param(prefix, param):
        if not param.name.startswith(prefix):
            param.name = f"{prefix}.{param.name}"


class FrozenCnCLIPEmbedder(AbstractEmbModel):
    """Uses the Bert transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="./chinese_clip/model_configs/chinese_clip_L14.json",
        tokenizer_version="bert-base-chinese",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
    ):  # Chinese-CLIP-ViT-L14
        super().__init__()

        with open(version, "r") as f:
            version = json_load(f)
        version = version.get("text_config")

        assert layer in self.LAYERS
        if layer == "hidden":
            version["output_hidden_states"] = True

        config = BertConfig(**version)
        self.bert = BertModel(config)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_version, do_lower_case=True)

        if freeze:
            self.freeze()

        self.layer = layer
        self.layer_idx = layer_idx
        self.max_length = max_length
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= config.num_hidden_layers

    def freeze(self):
        self.bert.set_train(False)
        self.bert.set_grad(False)

        for _, p in self.parameters_and_names():
            p.requires_grad = False

    def tokenize(self, text, **kwargs):
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
        (last_hidden_state, _, hidden_states, _) = self.embedding(input_ids=tokens)

        if self.layer == "last":
            z = last_hidden_state
        else:
            z = hidden_states[self.layer_idx]
        return z

    def embedding(self, input_ids):
        return self.bert(input_ids=input_ids)

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
        self.model = openclip_create_model(arch, pretrained=pretrained)

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

    def tokenize(self, text, lpw=False, max_embeddings_multiples=4):
        if lpw:
            tokens, lengths = lpw_openclip_tokenize2(text, max_embeddings_multiples=max_embeddings_multiples)
        else:
            tokens, lengths = openclip_tokenize(text)
        tokens = np.array(tokens, dtype=np.int32)
        lengths = np.array(lengths, dtype=np.int32)
        return tokens, lengths

    @ms.jit
    def construct(self, tokens):
        max_embeddings_multiples = (tokens.shape[1] - 2) // (self.max_length - 2)
        if max_embeddings_multiples > 1:
            z = self.get_unweighted_text_embeddings_SDXL3(tokens, max_embeddings_multiples)
            return z
        else:
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

    def get_unweighted_text_embeddings_SDXL2(self, tokens, max_embeddings_multiples):
        tokens_embeds_all = []
        last_embeds_all = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = tokens[:, i * (self.max_length - 2) : (i + 1) * (self.max_length - 2) + 2].copy()
            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = tokens[0, 0]
            text_input_chunk[:, -1] = tokens[0, -1]
            z = self.encode_with_transformer(text_input_chunk)
            if not self.return_pooled and self.legacy:
                tokens_embeds = z
            elif not self.return_pooled and not self.legacy:
                tokens_embeds = z[self.layer_idx]
            else:
                assert not self.legacy
                tokens_embeds = z[self.layer_idx]
                last_embeds = z[0]
                last_embeds = self.model.ln_final(last_embeds)
            # no_boseos_middle
            if i == 0:
                # discard the ending token
                tokens_embeds = tokens_embeds[:, :-1]
                if self.return_pooled and not self.legacy:
                    last_embeds = last_embeds[:, :-1]
            elif i == max_embeddings_multiples - 1:
                # discard the starting token
                tokens_embeds = tokens_embeds[:, 1:]
                if self.return_pooled and not self.legacy:
                    last_embeds = last_embeds[:, 1:]
            else:
                # discard both starting and ending tokens
                tokens_embeds = tokens_embeds[:, 1:-1]
                if self.return_pooled and not self.legacy:
                    last_embeds = last_embeds[:, 1:-1]
            tokens_embeds_all.append(tokens_embeds)
            if self.return_pooled and not self.legacy:
                last_embeds_all.append(last_embeds)
        tokens_embeds = ops.concat(tokens_embeds_all, axis=1)
        if self.return_pooled and not self.legacy:
            last_embeds = ops.concat(last_embeds_all, axis=1)
            pooled = self.pool(last_embeds, tokens)
            return tokens_embeds, pooled
        return tokens_embeds

    def get_unweighted_text_embeddings_SDXL3(self, tokens, max_embeddings_multiples):
        max_ids, text_embeddings = None, None

        tokens_embeds_all = []
        text_embeddings_all = []
        weight_all = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = tokens[:, i * (self.max_length - 2) : (i + 1) * (self.max_length - 2) + 2].copy()
            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = tokens[0, 0]
            text_input_chunk[:, -1] = tokens[0, -1]
            z = self.encode_with_transformer(text_input_chunk)

            assert self.return_pooled and (not self.legacy)
            tokens_embeds = z[self.layer_idx]
            pooled = z[-1]

            # last_embeds = z[0]
            # last_embeds = self.model.ln_final(last_embeds)
            # # last_embeds = ops.concat(last_embeds_all, axis=1)
            # pooled = self.pool(last_embeds, tokens)

            # no_boseos_middle: True
            if i == 0:
                # discard the ending token
                tokens_embeds = tokens_embeds[:, :-1]
                # if self.return_pooled and not self.legacy:
                #     last_embeds = last_embeds[:, :-1]
            elif i == max_embeddings_multiples - 1:
                # discard the starting token
                tokens_embeds = tokens_embeds[:, 1:]
                # if self.return_pooled and not self.legacy:
                #     last_embeds = last_embeds[:, 1:]
            else:
                # discard both starting and ending tokens
                tokens_embeds = tokens_embeds[:, 1:-1]
                # if self.return_pooled and not self.legacy:
                #     last_embeds = last_embeds[:, 1:-1]

            if i == 0:
                text_embeddings = pooled
                max_ids = ops.max(text_input_chunk, axis=1, keepdims=True)[0].view(-1, 1)
            else:
                now_max_ids = ops.max(text_input_chunk, axis=1, keepdims=True)[0].view(-1, 1)
                text_embeddings = ops.where(ops.cast(max_ids > now_max_ids, ms.bool_), text_embeddings, pooled)

            # [[499, ...], [40970, ..., 0, 0, 0],]
            # [[0, 0,...], [0, 0, ..., 1, 1, 1],]
            # [0, 60]
            # indices = [75, 60]
            indices = ops.argmax(ops.eq(text_input_chunk, 0).astype(ms.int32), dim=1)
            indices = ops.where(ops.cast(indices == 0, ms.bool_), 75 * ops.ones_like(indices), indices)

            weight_all.append(indices.unsqueeze(1))
            # shape: (bs, 1, 77)
            text_embeddings_all.append(text_embeddings.unsqueeze(1))
            tokens_embeds_all.append(tokens_embeds)

        tokens_embeds = ops.concat(tokens_embeds_all, axis=1)

        # [[75, 75], [75, 60]]
        weight = ops.concat(weight_all, axis=1).astype(text_embeddings.dtype)
        # [[0.5, 0.5], [0.55, 0.44]]
        weight = weight / ops.sum(weight, dim=-1).unsqueeze(1)
        # shape: (bs, n_chunk, 77)
        text_embeddings_all = ops.concat(text_embeddings_all, axis=1)
        # shape: (bs, 77)
        pooled = ops.sum(weight.unsqueeze(-1) * text_embeddings_all, dim=1)

        return tokens_embeds, pooled


class FrozenOpenCLIPEmbedder2_lora(FrozenOpenCLIPEmbedder2):
    """Lora injection to openclip embedder.
    Currently only support injection to dense layers in attention modules."""

    def __init__(self, *, lora_dim=4, lora_alpha=None, lora_dropout=0.0, lora_merge_weights=True, **kwargs):
        super(FrozenOpenCLIPEmbedder2_lora, self).__init__(**kwargs)
        from gm.modules.embedders.open_clip.transformer import MultiheadAttention
        from gm.modules.lora import Dense as Dense_lora
        from gm.modules.lora import mark_only_lora_as_trainable

        for cell_name, cell in self.cells_and_names():
            if isinstance(cell, MultiheadAttention):
                assert hasattr(cell, "out_proj")
                inner_dim, query_dim = cell.out_proj.in_channels, cell.out_proj.out_channels
                cell.out_proj = Dense_lora(
                    inner_dim,
                    query_dim,
                    r=lora_dim,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    merge_weights=lora_merge_weights,
                )
                _ = [_ for _ in map(partial(self._prefix_param, cell_name), cell.out_proj.get_parameters())]

        mark_only_lora_as_trainable(self, bias="none")

        num_param = sum([p.size for _, p in self.parameters_and_names()])
        num_param_trainable = sum([p.size for p in self.trainable_params()])
        print(
            f"FrozenOpenCLIPEmbedder2_lora total params: {float(num_param) / 1e9}B, "
            f"trainable params: {float(num_param_trainable) / 1e6}M."
        )

    @staticmethod
    def _prefix_param(prefix, param):
        if not param.name.startswith(prefix):
            param.name = f"{prefix}.{param.name}"


class FrozenOpenCLIPImageEmbedder(AbstractEmbModel):
    """
    Uses the OpenCLIP vision transformer encoder for images
    """

    def __init__(
        self,
        arch="ViT-H-14",
        version: str = "",
        max_length=77,
        freeze=True,
        antialias=True,
        ucg_rate=0.0,
        unsqueeze_dim=False,
        repeat_to_max_len=False,
        num_image_crops=0,
        output_tokens=False,
    ):
        super().__init__()
        model = openclip_create_model(arch, pretrained=version)
        del model.transformer
        self.model = model
        self.max_crops = num_image_crops
        self.pad_to_max_len = self.max_crops > 0
        self.repeat_to_max_len = repeat_to_max_len and (not self.pad_to_max_len)
        self.max_length = max_length
        if freeze:
            self.freeze()

        self.antialias = antialias

        self.mean = Tensor(np.expand_dims([0.48145466, 0.4578275, 0.40821073], axis=(0, 2, 3)).astype(np.float32))
        self.std = Tensor(np.expand_dims([0.26862954, 0.26130258, 0.27577711], axis=(0, 2, 3)).astype(np.float32))

        self.ucg_rate = ucg_rate
        self.unsqueeze_dim = unsqueeze_dim
        self.stored_batch = None
        self.model.visual.output_tokens = output_tokens
        self.output_tokens = output_tokens

    def preprocess(self, x):
        # FIXME: antialias is not supported
        x = ops.interpolate(x, (224, 224), mode="bicubic", align_corners=True)
        # normalize to [0,1]
        x = (x + 1.0) / 2.0
        # renormalize according to clip
        x = (x - self.mean) / self.std
        return x

    def construct(self, image: Tensor, no_dropout: bool = False):
        z = self.encode_with_vision_transformer(image)
        tokens = None

        if self.output_tokens:
            z, tokens = z[0], z[1]
        z = z.to(image.dtype)

        if self.ucg_rate > 0.0 and not no_dropout and not (self.max_crops > 0):
            z = ops.bernoulli((1.0 - self.ucg_rate) * ops.ones(z.shape[0], dtype=z.dtype)).expand_dims(-1) * z
            if tokens is not None:
                tokens = (
                    expand_dims_like(
                        ops.bernoulli((1.0 - self.ucg_rate) * ops.ones(tokens.shape[0], dtype=tokens.dtype)), tokens
                    )
                    * tokens
                )

        if self.unsqueeze_dim:
            z = z.expand_dims(1)

        if self.output_tokens:
            assert not self.repeat_to_max_len
            assert not self.pad_to_max_len
            return tokens, z

        if self.repeat_to_max_len:
            z_ = z.expand_dims(1) if z.ndim == 2 else z
            return z_.repeat(self.max_length, axis=1), z

        elif self.pad_to_max_len:
            assert z.ndim == 3
            z_pad = ops.cat(
                (z, ops.zeros((z.shape[0], self.max_length - z.shape[1], z.shape[2]), dtype=z.dtype)), axis=1
            )
            return z_pad, z_pad[:, 0, ...]

        return z

    def encode_with_vision_transformer(self, img: Tensor) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if img.ndim == 5:
            assert self.max_crops == img.shape[1]
            img = img.reshape(-1, *img.shape[2:])  # b n c h w -> (b n) c h w
        img = self.preprocess(img)
        if not self.output_tokens:
            assert not self.model.visual.output_tokens
            x = self.model.visual(img)
            tokens = None
        else:
            assert self.model.visual.output_tokens
            x, tokens = self.model.visual(img)
        if self.max_crops > 0:
            x = x.reshape(-1, self.max_crops, x.shape[-1])  # (b n) d -> b n d
            # drop out between 0 and all along the sequence axis
            x = ops.bernoulli((1.0 - self.ucg_rate) * ops.ones((x.shape[0], x.shape[1], 1), dtype=x.dtype)) * x
            if tokens is not None:
                tokens = tokens.reshape(-1, self.max_crops, *tokens.shape[1:]).swapaxes(1, 2)  # (b n) t d -> b t n d
                tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1)  # b t n d -> b t (n d)
                ops.print_(
                    f"You are running very experimental token-concat in {self.__class__.__name__}. "
                    f"Check what you are doing, and then remove this message."
                )
        if self.output_tokens:
            return x, tokens
        return x

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


class FrozenOpenCLIPEmbedder2_CLIPTokenizer(FrozenOpenCLIPEmbedder2):
    """
    A wrapper over FrozenOpenCLIPEmbedder2 to use the CLIPTokenizer (from transformer library) instead of SimpleTokenizer
    clip tokenizer from 'openai/clip-vit-large-patch14' is the same as the tokenizer from 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        # change to pad with zeros, not eos
        self.tokenizer._pad_token = "!"
        assert (
            self.tokenizer.pad_token_id == 0
        ), f"Expect FrozenOpenCLIPEmbedder2 pads with zeros, not {self.tokenizer.pad_token_id}"

    # rewrite the tokenize function
    def tokenize(self, text, **kwargs):
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
