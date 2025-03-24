import math
from functools import partial
from typing import List, Optional, Tuple, Union

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import Parameter, mint, nn, ops
from mindspore.common.initializer import TruncatedNormal, XavierNormal, initializer

from .basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from .helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from .vqvae import VQVAE, VectorQuantizer2


class SharedAdaLin(mint.nn.Linear):
    def construct(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().construct(cond_BD).view((-1, 1, 6, C))  # B16C


class VAR(nn.Cell):
    def __init__(
        self,
        vae_local: VQVAE,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads

        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1  # progressive training

        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum(pn**2 for pn in self.patch_nums)
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = ms.Generator()

        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = mint.nn.Linear(self.Cvae, self.C)

        # 2. class embedding
        self.init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = mint.full((1, num_classes), fill_value=1.0 / num_classes, dtype=ms.float32)
        self.class_emb = mint.nn.Embedding(self.num_classes + 1, self.C)

        self.pos_start = Parameter(mint.empty(1, self.first_l, self.C))

        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = mint.empty(1, pn * pn, self.C)
            ops.assign(pe, initializer(TruncatedNormal(sigma=self.init_std, mean=0.0), pe.shape))
            pos_1LC.append(pe)
        pos_1LC = mint.cat(pos_1LC, dim=1)  # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = mint.nn.Embedding(len(self.patch_nums), self.C)

        self.ms_embed1 = Embed1()
        self.ms_embed2 = Embed2()

        # 4. backbone blocks
        self.shared_ada_lin = (
            nn.SequentialCell(mint.nn.SiLU(), SharedAdaLin(self.D, 6 * self.C))
            if shared_aln
            else mint.nn.Identity()
        )

        norm_layer = partial(mint.nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [
            x.item() for x in mint.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.CellList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                )
                for block_idx in range(depth)
            ]
        )

        print(
            f"  \n[VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n"
            f"    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g}"
            f" ({mint.linspace(0, drop_path_rate, depth)})",
            end="\n\n",
            flush=True,
        )

        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d = mint.cat([mint.full((pn * pn,), i) for i, pn in enumerate(self.patch_nums)]).view((1, self.L, 1))
        dT = d.transpose(1, 2)  # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)
        attn_bias_for_masking = mint.where(d >= dT, 0.0, -ms.numpy.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer("attn_bias_for_masking", attn_bias_for_masking.contiguous())

        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = mint.nn.Linear(self.C, self.V)
        self.init_weight()

    def register_buffer(self, name, attr):
        setattr(self, name, Parameter(default_input=attr, requires_grad=False))

    def init_weight(self):
        weight = initializer(TruncatedNormal(sigma=self.init_std, mean=0.0), self.class_emb.weight.shape)
        self.class_emb.weight.set_data(weight)
        weight = initializer(TruncatedNormal(sigma=self.init_std, mean=0.0), self.pos_start.shape)
        self.pos_start.set_data(weight)
        weight = initializer(TruncatedNormal(sigma=self.init_std, mean=0.0), self.lvl_embed.weight.shape)
        self.lvl_embed.weight.set_data(weight)

    def get_logits(
        self, h_or_h_and_residual: Union[ms.Tensor, Tuple[ms.Tensor, ms.Tensor]], cond_BD: Optional[ms.Tensor]
    ):
        if not isinstance(h_or_h_and_residual, ms.Tensor):
            h, resi = h_or_h_and_residual  # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:  # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def replace_embedding(
        self, edit_mask: ms.Tensor, h_BChw: ms.Tensor, gt_BChw: ms.Tensor, ph: int, pw: int
    ) -> ms.Tensor:
        B = h_BChw.shape[0]
        h, w = edit_mask.shape[-2:]
        if edit_mask.ndim == 2:
            edit_mask = edit_mask.unsqueeze(0).expand((B, h, w))

        force_gt_B1hw = (
            F.interpolate(
                edit_mask.unsqueeze(1).to(dtype=ms.float32), size=(ph, pw), mode="bilinear", align_corners=False
            )
            .gt(0.5)
            .int()
        )
        if ph * pw <= 3:
            force_gt_B1hw.fill_(1)
        return gt_BChw * force_gt_B1hw + h_BChw * (1 - force_gt_B1hw)

    def autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[Union[int, ms.Tensor]],
        g_seed: Optional[int] = None,
        cfg=1.5,
        top_k=0,
        top_p=0.0,
        more_smooth=False,
        input_img_tokens: Optional[List[ms.Tensor]] = None,
        edit_mask: Optional[ms.Tensor] = None,
    ) -> ms.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed)
            rng = self.rng

        if label_B is None:
            label_B = mint.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = mint.full((B,), fill_value=self.num_classes if label_B < 0 else label_B)

        sos = cond_BD = self.class_emb(mint.cat((label_B, mint.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = (
            sos.unsqueeze(1).expand((2 * B, self.first_l, -1))
            + self.pos_start.expand((2 * B, self.first_l, -1))
            + lvl_pos[:, : self.first_l]
        )

        cur_L = 0
        f_hat = sos.new_zeros((B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1]))

        for b in self.blocks:
            b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):  # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn * pn

            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            # AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth:  # this is the default case
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)  # B, l, Cvae
            else:  # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)  # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose(1, 2).reshape(B, self.Cvae, pn, pn)
            if edit_mask is not None:
                gt_BChw = (
                    self.vae_quant_proxy[0]
                    .embedding(input_img_tokens[si])
                    .transpose(1, 2)
                    .reshape(B, self.Cvae, pn, pn)
                )
                h_BChw = self.replace_embedding(edit_mask, h_BChw, gt_BChw, pn, pn)

            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw
            )
            if si != self.num_stages_minus_1:  # prepare for next stage
                next_token_map = next_token_map.view((B, self.Cvae, -1)).transpose(1, 2)
                next_token_map = (
                    self.word_embed(next_token_map) + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
                )
                next_token_map = next_token_map.tile((2, 1, 1))  # double the batch sizes due to CFG

        for b in self.blocks:
            b.attn.kv_caching(False)
        return self.vae_proxy[0].fhat_to_img(f_hat).add(1).mul(0.5)  # de-normalize, from [-1, 1] to [0, 1]

    def construct(self, label_B: ms.Tensor, x_BLCv_wo_first_l: ms.Tensor) -> ms.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l.shape[0]
        cond_BD = self.ms_embed1(B, self.cond_drop_rate, self.num_classes, label_B, self.class_emb)
        x_BLC = self.ms_embed2(
            cond_BD,
            B,
            self.first_l,
            self.pos_start,
            self.prog_si,
            self.word_embed,
            x_BLCv_wo_first_l,
            self.lvl_embed,
            self.lvl_1L,
            self.pos_1LC,
            ed,
        )

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones((8, 8))
        main_type = mint.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)

        for i, b in enumerate(self.blocks):
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)

        if self.prog_si == 0:
            if isinstance(self.word_embed, mint.nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view((-1))[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC  # logits BLV, V is vocab_size

    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0:
            init_std = (1 / self.C / 3) ** 0.5  # init_std < 0: automated

        # for m in self.modules():
        for name, cell in self.cells_and_names():
            with_weight = hasattr(cell, "weight") and cell.weight is not None
            with_bias = hasattr(cell, "bias") and cell.bias is not None
            if isinstance(cell, mint.nn.Linear):
                cell.weight.set_data(initializer(TruncatedNormal(sigma=init_std), cell.weight.shape, cell.weight.dtype))
                if with_bias:
                    cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Embedding):
                cell.weight.set_data(initializer(TruncatedNormal(sigma=init_std), cell.weight.shape, cell.weight.dtype))
                if cell.padding_idx is not None:
                    cell.weight[cell.padding_idx] = 0
            elif isinstance(
                cell,
                (
                    mint.nn.LayerNorm,
                    mint.nn.BatchNorm1d,
                    mint.nn.BatchNorm2d,
                    mint.nn.BatchNorm3d,
                    mint.nn.SyncBatchNorm,
                    mint.nn.GroupNorm,
                    nn.InstanceNorm1d,
                    nn.InstanceNorm2d,
                    nn.InstanceNorm3d,
                ),
            ):
                if with_weight:
                    cell.weight.set_data(initializer("ones", cell.bias.shape, cell.bias.dtype))
                if with_bias:
                    cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(
                cell,
                (
                    nn.Conv1d,
                    mint.nn.Conv2d,
                    mint.nn.Conv3d,
                    nn.Conv1dTranspose,
                    mint.nn.ConvTranspose2d,
                    nn.Conv3dTranspose,
                ),
            ):
                if conv_std_or_gain > 0:
                    cell.weight.set_data(
                        initializer(TruncatedNormal(sigma=conv_std_or_gain), cell.weight.shape, cell.weight.dtype)
                    )
                else:
                    cell.weight.set_data(
                        initializer(XavierNormal(gain=-conv_std_or_gain), cell.weight.shape, cell.weight.dtype)
                    )

                if with_bias:
                    cell.bias.set_data(initializer("zeros", cell.bias.shape, cell.bias.dtype))

        if init_head >= 0:
            if isinstance(self.head, mint.nn.Linear):
                ops.assign(self.head.weight.data, self.head.weight.data.mul(init_head))
                ops.assign(self.head.bias.data, mint.zeros_like(self.head.bias.data))
                # self.head.bias.set_data(initializer("zeros", self.head.bias.shape, self.head.bias.dtype))

            elif isinstance(self.head, nn.SequentialCell):
                ops.assign(self.head[-1].weight.data, self.head[-1].weight.data.mul(init_head))
                ops.assign(self.head[-1].bias.data, mint.zeros_like(self.head[-1].bias.data))

        if isinstance(self.head_nm, AdaLNBeforeHead):
            ops.assign(self.head_nm.ada_lin[-1].weight.data, self.head_nm.ada_lin[-1].weight.data.mul(init_adaln))
            if hasattr(self.head_nm.ada_lin[-1], "bias") and self.head_nm.ada_lin[-1].bias is not None:
                ops.assign(self.head_nm.ada_lin[-1].bias.data, mint.zeros_like(self.head_nm.ada_lin[-1].bias.data))

        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            # sab: AdaLNSelfAttn
            ops.assign(sab.attn.proj.weight.data, sab.attn.proj.weight.data.div(math.sqrt(2 * depth)))
            ops.assign(sab.ffn.fc2.weight.data, sab.ffn.fc2.weight.data.div(math.sqrt(2 * depth)))
            if hasattr(sab.ffn, "fcg") and sab.ffn.fcg is not None:
                sab.ffn.fcg.bias.set_data(initializer("ones", sab.ffn.fcg.bias.shape, sab.ffn.fcg.bias.dtype))
                sab.ffn.fcg.weight.set_data(
                    initializer(TruncatedNormal(sigma=1e-5), sab.ffn.fcg.weight.shape, sab.ffn.fcg.weight.dtype)
                )
            if hasattr(sab, "ada_lin"):
                ops.assign(
                    sab.ada_lin[-1].weight.data[2 * self.C :], sab.ada_lin[-1].weight.data[2 * self.C :].mul(init_adaln)
                )
                ops.assign(
                    sab.ada_lin[-1].weight.data[: 2 * self.C],
                    sab.ada_lin[-1].weight.data[: 2 * self.C].mul(init_adaln_gamma),
                )
                if hasattr(sab.ada_lin[-1], "bias") and sab.ada_lin[-1].bias is not None:
                    ops.assign(sab.ada_lin[-1].bias.data, mint.zeros_like(sab.ada_lin[-1].bias.data))
            elif hasattr(sab, "ada_gss"):
                ops.assign(sab.ada_gss.data[:, :, 2:], sab.ada_gss.data[:, :, 2:].mul(init_adaln))
                ops.assign(sab.ada_gss.data[:, :, :2], sab.ada_gss.data[:, :, :2].mul(init_adaln_gamma))

    def extra_repr(self):
        return f"drop_path_rate={self.drop_path_rate:g}"


class Embed1(nn.Cell):
    def construct(self, B, cond_drop_rate, num_classes, label_B, class_emb):
        label_B = mint.where(mint.rand(B) < cond_drop_rate, num_classes, label_B)
        cond_BD = class_emb(label_B)
        return cond_BD


class Embed2(nn.Cell):
    def construct(
        self, cond_BD, B, first_l, pos_start, prog_si, word_embed, x_BLCv_wo_first_l, lvl_embed, lvl_1L, pos_1LC, ed
    ):
        sos = cond_BD.unsqueeze(1).expand((B, first_l, -1)) + pos_start.expand((B, first_l, -1))

        if prog_si == 0:
            x_BLC = sos
        else:
            x_BLC = mint.cat((sos, word_embed(x_BLCv_wo_first_l.float())), dim=1)
        x_BLC += lvl_embed(lvl_1L[:, :ed].expand((B, -1))) + pos_1LC[:, :ed]  # lvl: BLC;  pos: 1LC
        return x_BLC


class VARHF(VAR):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-6,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes,
            depth=depth,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_eps=norm_eps,
            shared_aln=shared_aln,
            cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
        )
