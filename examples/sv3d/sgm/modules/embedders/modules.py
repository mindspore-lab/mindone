# reference to https://github.com/Stability-AI/generative-models
# for modular debugging only
import sys
from typing import Dict, List, Optional, Tuple, Union

sys.path.append("/mnt/disk4/fredhong/mindone_master/examples/sv3d/")
sys.path.append("/mnt/disk4/fredhong/mindone_master/")
import numpy as np
from omegaconf import ListConfig, OmegaConf
from sgm.modules.diffusionmodules.openaimodel import Timestep

# OpenCLIP model
from sgm.modules.embedders.open_clip import create_model as openclip_create_model
from sgm.util import append_dims, count_params, expand_dims_like, instantiate_from_config

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
        # assert len(tokens) == len(self.embedders), (
        #     f"tokens and self.embedders length is not equal, " f"{len(tokens)}, {len(self.embedders)}"
        # )

        vector, crossattn, concat = (
            Tensor([0], dtype=ms.float64),
            Tensor([0], dtype=ms.float64),
            Tensor([0], dtype=ms.float64),
        )  # graph mode requires same type to join for if-clause

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
                    # if vector is None:
                    vector = ops.zeros_like(emb)
                    vector = emb
                # else:
                #     vector = ops.concat((vector, emb), 1)
                elif emb.dim() == 3:  # crossattn
                    # if crossattn is None:
                    crossattn = ops.zeros_like(emb)
                    crossattn = emb
                # else:
                #     if crossattn.shape[1] == emb.shape[1]:
                #         crossattn = ops.concat((crossattn, emb), 2)
                #     else:
                #         # for image/text emb fusion
                #         if emb.shape[0] == 1:
                #             emb = ops.tile(emb, (crossattn.shape[0], 1, 1))
                #         crossattn = ops.concat((crossattn, emb), 1)
                else:  # concat
                    # if concat is None:
                    concat = ops.zeros_like(emb)
                    concat = emb
                # else:
                #     concat = ops.concat((concat, emb), 1)

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
        c = self.tokenize_embedding(batch_c, lpw=lpw, max_embeddings_multiples=max_embeddings_multiples)
        uc = self.tokenize_embedding(
            batch_c if batch_uc is None else batch_uc,
            force_uc_zero_embeddings,
            lpw=lpw,
            max_embeddings_multiples=max_embeddings_multiples,
        )
        return c, uc


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


class FrozenOpenCLIPImagePredictionEmbedder(AbstractEmbModel):
    def __init__(
        self,
        open_clip_embedding_config: dict,
        n_cond_frames: int,
        n_copies: int,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.open_clip = instantiate_from_config(open_clip_embedding_config)

    def construct(self, vid: Tensor) -> Tensor:
        vid = self.open_clip(vid)
        vid = vid.reshape(-1, self.n_cond_frames, vid.shape[1])  # (b t) d -> b t d
        vid = vid.repeat(self.n_copies, axis=0)  # b t d -> (b s) t d

        return vid


class VideoPredictionEmbedderWithEncoder(AbstractEmbModel):
    def __init__(
        self,
        n_cond_frames: int,
        n_copies: int,
        encoder_config: dict,
        sigma_sampler_config: Optional[dict] = None,
        sigma_cond_config: Optional[dict] = None,
        is_ae: bool = False,
        scale_factor: float = 1.0,
        disable_encoder_autocast: bool = False,
        en_and_decode_n_samples_a_time: int = 0,
    ):
        super().__init__()

        self.n_cond_frames = n_cond_frames
        self.n_copies = n_copies
        self.encoder = instantiate_from_config(encoder_config)
        self.sigma_sampler = instantiate_from_config(sigma_sampler_config) if sigma_sampler_config is not None else None
        self.sigma_cond = instantiate_from_config(sigma_cond_config) if sigma_cond_config is not None else None
        self.is_ae = is_ae
        self.scale_factor = scale_factor
        self.disable_encoder_autocast = disable_encoder_autocast
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

    def construct(
        self, vid: Tensor
    ) -> Union[Tensor, Tuple[Tensor, Tensor], Tuple[Tensor, dict], Tuple[Tuple[Tensor, Tensor], dict]]:
        sigma_cond = None
        if self.sigma_sampler is not None:
            b = vid.shape[0] // self.n_cond_frames
            sigmas = self.sigma_sampler(b)
            if self.sigma_cond is not None:
                sigma_cond = self.sigma_cond(sigmas)
                sigma_cond = sigma_cond.repeat(self.n_copies, axis=0)  # b d -> (b t) d
            sigmas = sigmas.repeat(self.n_copies, axis=0)  # b -> (b t)
            noise = ops.randn_like(vid)
            vid = vid + noise * append_dims(sigmas, vid.ndim)

        n_samples = self.en_and_decode_n_samples_a_time or vid.shape[0]
        all_out = []
        for n in range(0, vid.shape[0], n_samples):
            if self.is_ae:
                out = self.encoder.encode(vid[n : n + n_samples])
            else:
                out = self.encoder(vid[n : n + n_samples])
            all_out.append(out)

        vid = ops.cat(all_out, axis=0)
        vid *= self.scale_factor

        vid = vid.reshape(-1, self.n_cond_frames, *vid.shape[1:])  # (b t) c h w -> b t c h w
        vid = vid.reshape(vid.shape[0], -1, *vid.shape[3:])  # b t c h w -> b (t c) h w
        vid = vid.repeat(self.n_copies, axis=0)  # b (t c) h w -> (b s) (t c) h w

        if self.sigma_cond is not None:
            return vid, sigma_cond
        return vid


if __name__ == "__main__":
    # 1. make sure all conditioners func used in sampling is ok for graph mode
    from mindone.utils.seed import set_random_seed

    ms.context.set_context(mode=0, device_target="Ascend", device_id=0)
    set_random_seed(42)

    cond_frames_without_noise = ops.randn(1, 3, 576, 576)
    u_cond_frames_without_noise = ops.randn(1, 3, 576, 576)
    cond_frames = ops.randn(1, 3, 576, 576)
    u_cond_frames = ops.randn(1, 3, 576, 576)
    cond_aug = ops.randn(21)
    u_cond_aug = ops.randn(21)

    batch = {"cond_frames": cond_frames, "cond_frames_without_noise": cond_frames_without_noise, "cond_aug": cond_aug}
    # batch_uc = {
    #     "cond_frames": u_cond_frames,
    #     "cond_frames_without_noise": u_cond_frames_without_noise,
    #     "cond_aug": u_cond_aug
    # }
    batch_uc = {}
    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], Tensor):
            batch_uc[key] = batch[key].copy()

    class TestPipeline(nn.Cell):
        def __init__(self, model_cfg):
            super().__init__()
            self.conditioner = instantiate_from_config(model_cfg)
            self.num_frames = 21

        def construct(self, batch, batch_uc):
            c, uc = self.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )
            return c["vector"], uc["vector"]

    model_config = OmegaConf.load("/mnt/disk4/fredhong/mindone_master/examples/sv3d/configs/sampling/sv3d_u.yaml")
    model_config.model.params.sampler_config.params.verbose = False
    model_config.model.params.sampler_config.params.num_steps = 50
    model_config.model.params.sampler_config.params.guider_config.params.num_frames = 21
    conditioner_cfg = model_config.model.params.conditioner_config
    tester = TestPipeline(conditioner_cfg)
    c, uc = tester(batch, batch_uc)
    print(c)
    print(uc)
