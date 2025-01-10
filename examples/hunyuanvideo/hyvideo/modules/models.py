from typing import Any, List, Tuple, Optional, Union, Dict
import math

import mindspore as ms
from mindspore import nn, ops
import mindspore.ops.functional as F

from .norm_layers import LayerNorm, get_norm_layer
from .activation_layers import get_activation_layer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .posemb_layers import apply_rotary_emb
from .attention import VanillaAttention #, parallel_attention, get_cu_seqlens
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .token_refiner import SingleTokenRefiner, rearrange_qkv



class MMDoubleStreamBlock(nn.Cell):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype = None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        self.head_dim = head_dim
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_attn_qkv = nn.Dense(
            hidden_size, hidden_size * 3, has_bias=qkv_bias)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Dense(
            hidden_size, hidden_size, has_bias=qkv_bias)

        self.img_norm2 = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Dense(
            hidden_size, hidden_size * 3, has_bias=qkv_bias
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Dense(
            hidden_size, hidden_size, has_bias=qkv_bias,
        )

        self.txt_norm2 = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        #
        self.compute_attention = VanillaAttention(head_dim)

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def construct(
        self,
        img: ms.Tensor,
        txt: ms.Tensor,
        vec: ms.Tensor,
        cu_seqlens_q: Optional[ms.Tensor] = None,
        cu_seqlens_kv: Optional[ms.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
    ):
        # img:
        # txt:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, axis=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, axis=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        # "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        img_q, img_k, img_v = rearrange_qkv(img_qkv, self.heads_num)

        # Apply QK-Norm if needed
        # TODO: check whether need to cast to dtype of img_v
        img_q = self.img_attn_q_norm(img_q) # .to(img_v)
        img_k = self.img_attn_k_norm(img_k) # .to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            # assert (
            #    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            # ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)

        # "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        txt_q, txt_k, txt_v = rearrange_qkv(txt_qkv, self.heads_num)

        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q) # .to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k) # .to(txt_v)

        # Run actual attention.
        q = ops.concat((img_q, txt_q), axis=1)
        k = ops.concat((img_k, txt_k), axis=1)
        v = ops.concat((img_v, txt_v), axis=1)
        # assert (
        #    cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        # ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"

        # attention computation start

        attn = self.compute_attention(q, k, v)
        # TODO: support FA and parallel attn

        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt

class MMSingleStreamBlock(nn.Cell):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype = None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Dense(
            hidden_size, hidden_size * 3 + mlp_hidden_dim,
        )
        # proj and mlp_out
        self.linear2 = nn.Dense(
            hidden_size + mlp_hidden_dim, hidden_size,
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

        self.compute_attention = VanillaAttention(head_dim)

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def construct(
        self,
        x: ms.Tensor,
        vec: ms.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[ms.Tensor] = None,
        cu_seqlens_kv: Optional[ms.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[ms.Tensor, ms.Tensor] = None,
    ) -> ms.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, axis=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = ops.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], axis=-1
        )

        # q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        q, k, v = rearrange_qkv(qkv, heads_num=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q) # .to(v)
        k = self.k_norm(k) # .to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            # assert (
            #    img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            # ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = ops.concat((img_q, txt_q), axis=1)
            k = ops.concat((img_k, txt_k), axis=1)

        # Compute attention.
        # assert (
        #     cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        # ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"

        # attention computation start
        attn = self.compute_attention(
            q,
            k,
            v,
        )
        # TODO: add FA
        # attention computation end

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(ops.concat((attn, self.mlp_act(mlp)), axis=2))
        return x + apply_gate(output, gate=mod_gate)

# TODO: inherit ModelMixin, ConfigMixin
class HYVideoDiffusionTransformer(nn.Cell):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model, i.e. model parameter dtype
    """

    # @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        use_conv2d_patchify: bool = False,
        dtype = None,
    ):
        factory_kwargs = {"dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.use_conv2d_patchify = use_conv2d_patchify

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        # TODO: no need to use args, just parse these two params
        self.text_states_dim = args.text_states_dim
        self.text_states_dim_2 = args.text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, use_conv2d=use_conv2d_patchify, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.CellList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.CellList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def construct(
        self,
        x: ms.Tensor,
        t: ms.Tensor,  # Should be in range(0, 1000).
        text_states: ms.Tensor = None,
        text_mask: ms.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[ms.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[ms.Tensor] = None,
        freqs_sin: Optional[ms.Tensor] = None,
        guidance: ms.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
    ) -> ms.Tensor:
        '''
        x: (B C T H W), video latent
        t: (B,)
        text_states: (B S_t D_t); S_t - seq len of padded text tokens, D_t: text feature dim, from LM text encoder, default: S_t=256, D_t = 4096
        text_mask: (B S_t), 1 - retain, 0 - drop
        text_states_2: (B D_t2), from CLIP text encoder, global text feature (fuse 77 tokens), D_t2=768
        freqs_cos: (S attn_head_dim), S - seq len of the patchified video latent (T * H //2 * W//2)
        freqs_sin: (S attn_head_dim)
        guidance: (B,)
        '''
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # TODO: support setting max_seqlen
        # Compute cu_squlens and max_seqlen for flash attention
        # cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        # cu_seqlens_kv = cu_seqlens_q
        # max_seqlen_q = img_seq_len + txt_seq_len
        # max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for _, block in enumerate(self.double_blocks):
            img, txt = block(
                img,
                txt,
                vec,
                # cu_seqlens_q,
                # cu_seqlens_kv,
                # max_seqlen_q,
                # max_seqlen_kv,
                freqs_cis,
                )

        # Merge txt and img to pass through single stream blocks.
        x = ops.concat((img, txt), axis=1)
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                x = block(
                    x,
                    vec,
                    txt_seq_len,
                    # cu_seqlens_q,
                    # cu_seqlens_kv,
                    # max_seqlen_q,
                    # max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                    )

        # TODO: slicing replaced with
        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)

        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]
        # import pdb; pdb.set_trace()
        x = x.reshape((x.shape[0], t, h, w, c, pt, ph, pw))

        # x = torch.einsum("nthwcopq->nctohpwq", x)
        x = ops.transpose(x, (0, 4, 1, 5, 2, 6, 3, 7))

        imgs = x.reshape((x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def load_from_checkpoint(self, ckpt_path):
        '''
        model param dtype
        '''
        if ckpt_path.endswith('.pth'):
            state_dict = torch.load(ckpt_path)
            load_key = 'module'
            sd = state_dict[load_key]

            # TODO: support bf16 net params
            parameter_dict = dict()
            for pname in sd:
                parameter_dict[pname] = ms.Parameter(
                        ms.Tensor(sd[pname].cpu().detach().numpy(), dtype=self.dtype)
                        )
            # reshape conv3d weight to conv2d if use conv2d in PatchEmbed
            if self.use_conv2d_patchify:
                key_3d = "img_in.proj.weight"
                assert len(sd[key_3d].shape) == 5 and sd[key_3d].shape[-3] == 1 # c_out, c_in, 1, 2, 2
                conv3d_weight = sd.pop(key_3d)
                sd[key_3d] = ms.Parameter(conv3d_weight.squeeze(axis=-3), name=key_3d, dtype=self.dtype)

            param_not_load, ckpt_not_load = ms.load_param_into_net(self, parameter_dict, strict_load=True)
            print('param not load: ', param_not_load)
            print('ckpt not load: ', ckpt_not_load)

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]

        return counts


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}

