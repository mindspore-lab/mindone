import glob
import logging
import os
from typing import Optional

from opensora.models.diffusion.common import PatchEmbed2D
from opensora.models.diffusion.opensora.modules import Attention, BasicTransformerBlock, LayerNorm
from opensora.npu_config import npu_config

import mindspore as ms
from mindspore import nn, ops

from mindone.diffusers import __version__
from mindone.diffusers.configuration_utils import ConfigMixin, register_to_config
from mindone.diffusers.models.embeddings import PixArtAlphaTextProjection
from mindone.diffusers.models.modeling_utils import ModelMixin, load_state_dict
from mindone.diffusers.models.normalization import AdaLayerNormSingle
from mindone.diffusers.utils import SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, _add_variant, _get_model_file

logger = logging.getLogger(__name__)


class OpenSoraT2V_v1_3(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = True,
        sample_size_h: Optional[int] = None,
        sample_size_w: Optional[int] = None,
        sample_size_t: Optional[int] = None,
        patch_size: Optional[int] = None,
        patch_size_t: Optional[int] = None,
        activation_fn: str = "geglu",
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        caption_channels: int = None,
        interpolation_scale_h: float = 1.0,
        interpolation_scale_w: float = 1.0,
        interpolation_scale_t: float = 1.0,
        sparse1d: bool = False,
        sparse_n: int = 2,
        use_recompute=False,  # NEW
        FA_dtype=ms.bfloat16,  # NEW
        num_no_recompute: int = 0,  # NEW
    ):
        super().__init__()
        # Set some common variables used across the board.
        self.out_channels = in_channels if out_channels is None else out_channels
        self.config.hidden_size = self.config.num_attention_heads * self.config.attention_head_dim  # 24*96=2304
        self.gradient_checkpointing = use_recompute  # NEW
        self.use_recompute = use_recompute  # NEW
        self.FA_dtype = FA_dtype  # NEW
        self.sparse_n = sparse_n  # NEW
        self._init_patched_inputs()

        if self.use_recompute:
            num_no_recompute = self.config.num_no_recompute
            num_blocks = len(self.transformer_blocks)
            assert num_no_recompute >= 0, "Expect to have num_no_recompute as a positive integer."
            assert (
                num_no_recompute <= num_blocks
            ), "Expect to have num_no_recompute as an integer no greater than the number of blocks,"
            f"but got {num_no_recompute} and {num_blocks}."
            logger.info(f"Excluding {num_no_recompute} blocks from the recomputation list.")
            for bidx, block in enumerate(self.transformer_blocks):
                if bidx < num_blocks - num_no_recompute:
                    self.recompute(block)

    def _init_patched_inputs(self):
        self.config.sample_size = (self.config.sample_size_h, self.config.sample_size_w)
        interpolation_scale_thw = (
            self.config.interpolation_scale_t,
            self.config.interpolation_scale_h,
            self.config.interpolation_scale_w,
        )

        self.caption_projection = PixArtAlphaTextProjection(
            in_features=self.config.caption_channels, hidden_size=self.config.hidden_size
        )

        self.pos_embed = PatchEmbed2D(
            patch_size=self.config.patch_size,  # 2
            in_channels=self.config.in_channels,  # 8
            embed_dim=self.config.hidden_size,  # 2304
        )
        self.transformer_blocks = nn.CellList(
            [
                BasicTransformerBlock(
                    self.config.hidden_size,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    interpolation_scale_thw=interpolation_scale_thw,
                    sparse1d=self.config.sparse1d if i > 1 and i < 30 else False,
                    sparse_n=self.config.sparse_n,
                    sparse_group=i % 2 == 1,
                    FA_dtype=self.FA_dtype,
                )
                for i in range(self.config.num_layers)
            ]
        )
        self.norm_out = LayerNorm(self.config.hidden_size, elementwise_affine=False, eps=1e-6)
        self.scale_shift_table = ms.Parameter(ops.randn((2, self.config.hidden_size)) / self.config.hidden_size**0.5)
        self.proj_out = nn.Dense(
            self.config.hidden_size,
            self.config.patch_size_t * self.config.patch_size * self.config.patch_size * self.out_channels,
        )
        self.adaln_single = AdaLayerNormSingle(self.config.hidden_size)
        self.max_pool3d = nn.MaxPool3d(
            kernel_size=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size),
            stride=(self.config.patch_size_t, self.config.patch_size, self.config.patch_size),
            pad_mode="pad",
        )

        # save as attributes used in construct
        self.patch_size_t = self.config.patch_size_t
        self.patch_size = self.config.patch_size
        self.sparse1d = self.config.sparse1d
        self.sparse_n = self.config.sparse_n

    def recompute(self, b):
        if not b._has_config_recompute:
            b.recompute(parallel_optimizer_comm_recompute=True)
        if isinstance(b, nn.CellList):
            self.recompute(b[-1])
        elif ms.get_context("mode") == ms.GRAPH_MODE:
            b.add_flags(output_no_recompute=True)

    # rewrite class method to allow the state dict as input
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        state_dict = kwargs.pop("state_dict", None)  # additional key argument
        cache_dir = kwargs.pop("cache_dir", None)
        ignore_mismatched_sizes = kwargs.pop("ignore_mismatched_sizes", False)
        force_download = kwargs.pop("force_download", False)
        from_flax = kwargs.pop("from_flax", False)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        mindspore_dtype = kwargs.pop("mindspore_dtype", None)
        subfolder = kwargs.pop("subfolder", None)
        variant = kwargs.pop("variant", None)
        use_safetensors = kwargs.pop("use_safetensors", None)

        allow_pickle = False
        if use_safetensors is None:
            use_safetensors = True
            allow_pickle = True

        # Load config if we don't provide a configuration
        config_path = pretrained_model_name_or_path

        user_agent = {
            "diffusers": __version__,
            "file_type": "model",
            "framework": "pytorch",
        }

        # load config
        config, unused_kwargs, commit_hash = cls.load_config(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            return_commit_hash=True,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            user_agent=user_agent,
            **kwargs,
        )

        # load model
        model_file = None
        if from_flax:
            raise NotImplementedError("loading flax checkpoint in mindspore model is not yet supported.")
        else:
            if state_dict is None:  # edits: only search for model_file if state_dict is not provided
                if use_safetensors:
                    try:
                        model_file = _get_model_file(
                            pretrained_model_name_or_path,
                            weights_name=_add_variant(SAFETENSORS_WEIGHTS_NAME, variant),
                            cache_dir=cache_dir,
                            force_download=force_download,
                            proxies=proxies,
                            local_files_only=local_files_only,
                            token=token,
                            revision=revision,
                            subfolder=subfolder,
                            user_agent=user_agent,
                            commit_hash=commit_hash,
                        )
                    except IOError as e:
                        if not allow_pickle:
                            raise e
                        pass
                if model_file is None:
                    model_file = _get_model_file(
                        pretrained_model_name_or_path,
                        weights_name=_add_variant(WEIGHTS_NAME, variant),
                        cache_dir=cache_dir,
                        force_download=force_download,
                        proxies=proxies,
                        local_files_only=local_files_only,
                        token=token,
                        revision=revision,
                        subfolder=subfolder,
                        user_agent=user_agent,
                        commit_hash=commit_hash,
                    )

            model = cls.from_config(config, **unused_kwargs)
            if state_dict is None:  # edits: only load model_file if state_dict is None
                state_dict = load_state_dict(model_file, variant=variant)
            model._convert_deprecated_attention_blocks(state_dict)

            model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
                model,
                state_dict,
                model_file,
                pretrained_model_name_or_path,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
            )

            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": mismatched_keys,
                "error_msgs": error_msgs,
            }

        if mindspore_dtype is not None and not isinstance(mindspore_dtype, ms.Type):
            raise ValueError(
                f"{mindspore_dtype} needs to be of type `ms.Type`, e.g. `ms.float16`, but is {type(mindspore_dtype)}."
            )
        elif mindspore_dtype is not None:
            model = model.to(mindspore_dtype)

        model.register_to_config(_name_or_path=pretrained_model_name_or_path)

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.set_train(False)
        if output_loading_info:
            return model, loading_info

        return model

    @classmethod
    def load_from_checkpoint(cls, model, ckpt_path):
        if os.path.isdir(ckpt_path) or ckpt_path.endswith(".safetensors"):
            return cls.load_from_safetensors(model, ckpt_path)
        elif ckpt_path.endswith(".ckpt"):
            return cls.load_from_ms_checkpoint(ckpt_path)
        else:
            raise ValueError("Only support safetensors pretrained ckpt or MindSpore pretrained ckpt!")

    @classmethod
    def load_from_safetensors(cls, model, ckpt_path):
        if os.path.isdir(ckpt_path):
            ckpts = glob.glob(os.path.join(ckpt_path, "*.safetensors"))
            n_ckpt = len(ckpts)
            assert (
                n_ckpt == 1
            ), f"Expect to find only one safetenesors file under {ckpt_path}, but found {n_ckpt} .safetensors files."
            model_file = ckpts[0]
            pretrained_model_name_or_path = ckpt_path
        elif ckpt_path.endswith(".safetensors"):
            model_file = ckpt_path
            pretrained_model_name_or_path = os.path.dirname(ckpt_path)
        state_dict = load_state_dict(model_file, variant=None)
        state_dict = dict(
            [k.replace("_backbone.", "") if "_backbone." in k else k, v] for k, v in state_dict.items()
        )  # remove _backbone
        model._convert_deprecated_attention_blocks(state_dict)

        model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_pretrained_model(
            model,
            state_dict,
            model_file,
            pretrained_model_name_or_path,
            ignore_mismatched_sizes=False,
        )
        loading_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "mismatched_keys": mismatched_keys,
            "error_msgs": error_msgs,
        }
        logger.info(loading_info)
        return model

    @classmethod
    def load_from_ms_checkpoint(self, model, ckpt_path):
        sd = ms.load_checkpoint(ckpt_path)
        # filter 'network.' prefix
        rm_prefix = ["network."]
        all_pnames = list(sd.keys())
        for pname in all_pnames:
            for pre in rm_prefix:
                if pname.startswith(pre):
                    new_pname = pname.replace(pre, "")
                    sd[new_pname] = sd.pop(pname)

        sd = dict(
            [k.replace("_backbone.", "") if "_backbone." in k else k, v] for k, v in sd.items()
        )  # remove _backbone
        m, u = ms.load_param_into_net(model, sd)
        print("net param not load: ", m, len(m))
        print("ckpt param not load: ", u, len(u))
        return model

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def get_attention_mask(self, attention_mask):
        if attention_mask is not None:
            if not npu_config.enable_FA:
                attention_mask = attention_mask.to(ms.bool_)  # use bool for sdpa
        return attention_mask

    def construct(
        self,
        hidden_states: ms.Tensor,
        timestep: Optional[ms.Tensor] = None,
        encoder_hidden_states: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
        encoder_attention_mask: Optional[ms.Tensor] = None,
        **kwargs,
    ):
        batch_size, c, frame, h, w = hidden_states.shape
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 4:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # b, frame, h, w -> a video
            # b, 1, h, w -> only images
            attention_mask = attention_mask.to(self.dtype)

            attention_mask = attention_mask.unsqueeze(1)  # b 1 t h w
            attention_mask = self.max_pool3d(attention_mask)
            # b 1 t h w -> (b 1) 1 (t h w)
            attention_mask = attention_mask.reshape(batch_size, 1, -1)

            attention_mask = self.get_attention_mask(attention_mask)  # if use bool mask

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:
            # b, 1, l
            encoder_attention_mask = self.get_attention_mask(encoder_attention_mask)  # if use bool mask

        # 1. Input
        frame = ((frame - 1) // self.patch_size_t + 1) if frame % 2 == 1 else frame // self.patch_size_t  # patchfy
        height, width = (
            hidden_states.shape[-2] // self.patch_size,
            hidden_states.shape[-1] // self.patch_size,
        )

        hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
            hidden_states, encoder_hidden_states, timestep, batch_size, frame
        )

        # To
        # x            (t*h*w b d) or (t//sp*h*w b d)
        # cond_1       (l b d) or (l//sp b d)
        # b s h -> s b h
        hidden_states = hidden_states.swapaxes(0, 1).contiguous()
        # b s h -> s b h
        encoder_hidden_states = encoder_hidden_states.swapaxes(0, 1).contiguous()
        timestep = timestep.view(batch_size, 6, -1).swapaxes(0, 1).contiguous()

        sparse_mask = {}
        if self.sparse1d:
            for sparse_n in [1, self.sparse_n]:
                sparse_mask[sparse_n] = Attention.prepare_sparse_mask(
                    attention_mask, encoder_attention_mask, sparse_n, head_num=None
                )

        # 2. Blocks
        for i, block in enumerate(self.transformer_blocks):
            if self.sparse1d:
                if i > 1 and i < 30:
                    attention_mask, encoder_attention_mask = sparse_mask[block.attn1.processor.sparse_n][
                        block.attn1.processor.sparse_group
                    ]
                else:
                    attention_mask, encoder_attention_mask = sparse_mask[1][block.attn1.processor.sparse_group]

            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                frame=frame,
                height=height,
                width=width,
            )  # BSH

        # To (b, t*h*w, h) or (b, t//sp*h*w, h)
        # s b h -> b s h
        hidden_states = hidden_states.swapaxes(0, 1).contiguous()

        # 3. Output
        output = self._get_output_for_patched_inputs(
            hidden_states=hidden_states,
            timestep=timestep,
            embedded_timestep=embedded_timestep,
            num_frames=frame,
            height=height,
            width=width,
        )  # b c t h w

        return output

    def _operate_on_patched_inputs(self, hidden_states, encoder_hidden_states, timestep, batch_size, frame):
        hidden_states = self.pos_embed(hidden_states.to(self.dtype))  # (b, t*h*w, d)

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        timestep, embedded_timestep = self.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=self.dtype
        )  # b 6d, b d

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # b, 1, l, d
        assert encoder_hidden_states.shape[1] == 1, f"encoder_hidden_states.shape is {encoder_hidden_states}"
        # b 1 l d -> (b 1) l d
        encoder_hidden_states = encoder_hidden_states.reshape(
            -1, encoder_hidden_states.shape[-2], encoder_hidden_states.shape[-1]
        )

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep

    def _get_output_for_patched_inputs(self, hidden_states, timestep, embedded_timestep, num_frames, height, width):
        shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, axis=1)
        hidden_states = self.norm_out(hidden_states)  # BSH -> BSH
        hidden_states = hidden_states.squeeze(1) if hidden_states.shape[1] == 1 else hidden_states

        # Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1) if hidden_states.shape[1] == 1 else hidden_states

        # unpatchify
        hidden_states = hidden_states.reshape(
            -1,
            num_frames,
            height,
            width,
            self.config.patch_size_t,
            self.config.patch_size,
            self.config.patch_size,
            self.out_channels,
        )
        # nthwopqc -> nctohpwq
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.reshape(
            -1,
            self.out_channels,
            num_frames * self.config.patch_size_t,
            height * self.config.patch_size,
            width * self.config.patch_size,
        )
        return output


def OpenSoraT2V_v1_3_2B_122(**kwargs):
    kwargs.pop("skip_connection", None)
    return OpenSoraT2V_v1_3(
        num_layers=32,
        attention_head_dim=96,
        num_attention_heads=24,
        patch_size_t=1,
        patch_size=2,
        caption_channels=4096,
        cross_attention_dim=2304,
        activation_fn="gelu-approximate",
        **kwargs,
    )


OpenSora_v1_3_models = {
    "OpenSoraT2V_v1_3-2B/122": OpenSoraT2V_v1_3_2B_122,  # 2.7B
}

OpenSora_v1_3_models_class = {
    "OpenSoraT2V_v1_3-2B/122": OpenSoraT2V_v1_3,
}

if __name__ == "__main__":
    from opensora.models.causalvideovae import ae_stride_config

    args = type(
        "args",
        (),
        {
            "ae": "WFVAEModel_D8_4x8x8",
            "model_max_length": 300,
            "max_height": 256,
            "max_width": 512,
            "num_frames": 33,
            "compress_kv_factor": 1,
            "interpolation_scale_t": 1,
            "interpolation_scale_h": 1,
            "interpolation_scale_w": 1,
            "sparse1d": True,
            "sparse_n": 4,
            "rank": 64,
        },
    )
    b = 2
    c = 8
    cond_c = 4096
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    latent_size = (args.max_height // ae_stride_h, args.max_width // ae_stride_w)
    num_frames = (args.num_frames - 1) // ae_stride_t + 1

    model = OpenSoraT2V_v1_3_2B_122(
        in_channels=c,
        out_channels=c,
        sample_size_h=latent_size,
        sample_size_w=latent_size,
        sample_size_t=num_frames,
        # activation_fn="gelu-approximate",
        attention_bias=True,
        double_self_attention=False,
        norm_elementwise_affine=False,
        norm_eps=1e-06,
        only_cross_attention=False,
        upcast_attention=False,
        interpolation_scale_t=args.interpolation_scale_t,
        interpolation_scale_h=args.interpolation_scale_h,
        interpolation_scale_w=args.interpolation_scale_w,
        sparse1d=args.sparse1d,
        sparse_n=args.sparse_n,
    )

    try:
        path = "checkpoints/Open-Sora-Plan-v1.3.0/any93x640x640/diffusion_pytorch_model.safetensors"
        from safetensors.torch import load_file as safe_load

        ckpt = safe_load(path, device="cpu")
        msg = model.load_state_dict(ckpt, strict=True)
        print(msg)
        # some difference from sample.py
        # e.g. do not have mix precision
    except Exception as e:
        print(e)
    # print(model)
    # print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9} B")
    # import sys;sys.exit()
    x = ops.randn(
        b, c, 1 + (args.num_frames - 1) // ae_stride_t, args.max_height // ae_stride_h, args.max_width // ae_stride_w
    )
    cond = ops.randn(b, 1, args.model_max_length, cond_c)
    attn_mask = ops.randint(
        0,
        2,
        (b, 1 + (args.num_frames - 1) // ae_stride_t, args.max_height // ae_stride_h, args.max_width // ae_stride_w),
    )  # B L or B 1+num_images L
    cond_mask = ops.randint(0, 2, (b, 1, args.model_max_length))  # B L or B 1+num_images L
    timestep = ops.randint(0, 1000, (b,))
    model_kwargs = dict(
        hidden_states=x,
        encoder_hidden_states=cond,
        attention_mask=attn_mask,
        encoder_attention_mask=cond_mask,
        timestep=timestep,
    )
    model.set_train(False)
    output = model(**model_kwargs)
    print(output[0].shape)
