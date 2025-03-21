from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

import mindspore as ms
from mindspore import Parameter, mint, nn
from mindspore.mint.nn import functional as F

# this file only provides the VectorQuantizer2 used in VQVAE
__all__ = [
    "VectorQuantizer2",
]


class VectorQuantizer2(nn.Cell):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self,
        vocab_size,
        Cvae,
        using_znorm,
        beta: float = 0.25,
        default_qresi_counts=0,
        v_patch_nums=None,
        quant_resi=0.5,
        share_quant_resi=4,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums

        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:  # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared(
                [
                    (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else mint.nn.Identity())
                    for _ in range(default_qresi_counts or len(self.v_patch_nums))
                ]
            )
        elif share_quant_resi == 1:  # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else mint.nn.Identity())
        else:  # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(
                nn.CellList(
                    [
                        (Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else mint.nn.Identity())
                        for _ in range(share_quant_resi)
                    ]
                )
            )

        self.register_buffer("ema_vocab_hit_SV", mint.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0

        self.beta: float = beta
        self.embedding = mint.nn.Embedding(self.vocab_size, self.Cvae)

        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1  # progressive training: not supported yet, prog_si always -1

    def register_buffer(self, name, attr):
        setattr(self, name, Parameter(default_input=attr, requires_grad=False))

    def extra_repr(self) -> str:
        return f"{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}"

    # # ===================== `construct` is only used in VAE training =====================
    def construct(self, f_BChw: ms.Tensor, ret_usages=False) -> Tuple[ms.Tensor, List[float], ms.Tensor]:
        dtype = f_BChw.dtype
        if dtype != ms.float32:
            f_BChw = f_BChw.float()
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw

        f_rest = mint.clone(f_no_grad)
        f_hat = mint.zeros_like(f_rest)

        mean_vq_loss = ms.Tensor(0.0)
        vocab_hit_V = mint.zeros(self.vocab_size)
        SN = len(self.v_patch_nums)
        for si, pn in enumerate(self.v_patch_nums):  # from small to large
            # find the nearest embedding
            if self.using_znorm:
                rest_NC = (
                    F.adaptive_avg_pool2d(f_rest, (pn, pn)).permute(0, 2, 3, 1).reshape(-1, C)
                    if (si != SN - 1)
                    else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                )
                rest_NC = F.normalize(rest_NC, dim=-1)
                idx_N = mint.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                rest_NC = (
                    F.adaptive_avg_pool2d(f_rest, (pn, pn)).permute(0, 2, 3, 1).reshape(-1, C)
                    if (si != SN - 1)
                    else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
                )
                d_no_grad = mint.sum(rest_NC.square(), dim=1, keepdim=True) + mint.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False
                )
                d_no_grad = mint.addmm(
                    d_no_grad, rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                )  # (B*h*w, vocab_size)
                idx_N = mint.argmin(d_no_grad, dim=1)

            hit_V = idx_N.bincount(minlength=self.vocab_size).float()

            # calc loss
            idx_Bhw = idx_N.view((B, pn, pn))
            h_BChw = (
                F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode="bicubic").contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat = f_hat + h_BChw
            f_rest -= h_BChw

            # if self.training and dist.initialized():
            #     handler.wait()
            #     if self.record_hit == 0:
            #         self.ema_vocab_hit_SV[si].copy_(hit_V)
            #     elif self.record_hit < 100:
            #         self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
            #     else:
            #         self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
            #     self.record_hit += 1
            vocab_hit_V = mint.add(vocab_hit_V, hit_V)
            mean_vq_loss += F.mse_loss(f_hat.data, f_BChw).mul(self.beta) + F.mse_loss(f_hat, f_no_grad)

            mean_vq_loss *= 1.0 / SN
            f_hat = (f_hat.data - f_no_grad).add(f_BChw)

        margin = mint.distributed.get_world_size() * (f_BChw.numel() / f_BChw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages:
            usages = [
                (self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100
                for si, pn in enumerate(self.v_patch_nums)
            ]
        else:
            usages = None
        return f_hat, usages, mean_vq_loss

    #
    # # ===================== `construct` is only used in VAE training =====================

    def embed_to_fhat(
        self, ms_h_BChw: List[ms.Tensor], all_to_max_scale=True, last_one=False
    ) -> Union[List[ms.Tensor], ms.Tensor]:
        ls_f_hat_BChw = []
        B = ms_h_BChw[0].shape[0]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BChw[0].new_zeros((B, self.Cvae, H, W), dtype=ms.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                h_BChw = ms_h_BChw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BChw = F.interpolate(h_BChw, size=(H, W), mode="bicubic")
                h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
                f_hat = mint.add(f_hat, h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training or inference (we'll interpolate every token map to the max H W, like above)
            # WARNING: this should only be used for experimental purpose
            f_hat = ms_h_BChw[0].new_zeros((B, self.Cvae, self.v_patch_nums[0], self.v_patch_nums[0]), dtype=ms.float32)
            for si, pn in enumerate(self.v_patch_nums):  # from small to large
                f_hat = F.interpolate(f_hat, size=(pn, pn), mode="bicubic")
                h_BChw = self.quant_resi[si / (SN - 1)](ms_h_BChw[si])
                f_hat = mint.add(f_hat, h_BChw)
                if last_one:
                    ls_f_hat_BChw = f_hat
                else:
                    ls_f_hat_BChw.append(f_hat)

        return ls_f_hat_BChw

    def f_to_idxBl_or_fhat(
        self, f_BChw: ms.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None
    ) -> List[Union[ms.Tensor]]:  # z_BChw is the feature from inp_img_no_grad
        B, C, H, W = f_BChw.shape
        f_no_grad = f_BChw
        f_rest = f_no_grad.clone()
        f_hat = mint.zeros_like(f_rest)

        f_hat_or_idx_Bl: List[ms.Tensor] = []

        patch_hws = [
            (pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in (v_patch_nums or self.v_patch_nums)
        ]  # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W, f"{patch_hws[-1]} != ({H}, {W})"

        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws):  # from small to large
            if 0 <= self.prog_si < si:
                break  # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = (
                F.adaptive_avg_pool2d(f_rest, (ph, pw)).permute(0, 2, 3, 1).reshape(-1, C)
                if (si != SN - 1)
                else f_rest.permute(0, 2, 3, 1).reshape(-1, C)
            )
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_N = mint.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = mint.sum(z_NC.square(), dim=1, keepdim=True) + mint.sum(
                    self.embedding.weight.data.square(), dim=1, keepdim=False
                )
                d_no_grad = mint.addmm(
                    d_no_grad, z_NC, self.embedding.weight.data.T, alpha=-2, beta=1
                )  # (B*h*w, vocab_size)
                idx_N = mint.argmin(d_no_grad, dim=1)

            idx_Bhw = idx_N.view((B, ph, pw))
            h_BChw = (
                F.interpolate(self.embedding(idx_Bhw).permute(0, 3, 1, 2), size=(H, W), mode="bicubic").contiguous()
                if (si != SN - 1)
                else self.embedding(idx_Bhw).permute(0, 3, 1, 2).contiguous()
            )
            h_BChw = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h_BChw)
            f_rest.sub_(h_BChw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_N.reshape(B, ph * pw))

        return f_hat_or_idx_Bl

    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[ms.Tensor]) -> ms.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)

        f_hat = gt_ms_idx_Bl[0].new_zeros((B, C, H, W), dtype=ms.float32)
        pn_next: int = self.v_patch_nums[0]
        for si in range(SN - 1):
            if self.prog_si == 0 or (0 <= self.prog_si - 1 < si):
                break  # progressive training: not supported yet, prog_si always -1
            h_BChw = F.interpolate(
                self.embedding(gt_ms_idx_Bl[si]).transpose(1, 2).view((B, C, pn_next, pn_next)),
                size=(H, W),
                mode="bicubic",
            )
            f_hat = mint.add(f_hat, self.quant_resi[si / (SN - 1)](h_BChw))
            pn_next = self.v_patch_nums[si + 1]
            next_scales.append(F.adaptive_avg_pool2d(f_hat, (pn_next, pn_next)).view((B, C, -1)).transpose(1, 2))
        return mint.cat(next_scales, dim=1) if len(next_scales) else None  # cat BlCs to BLC, this should be float32

    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(
        self, si: int, SN: int, f_hat: ms.Tensor, h_BChw: ms.Tensor
    ) -> Tuple[Optional[ms.Tensor], ms.Tensor]:  # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )  # conv after upsample
            f_hat = mint.add(f_hat, h)
            return f_hat, F.adaptive_avg_pool2d(f_hat, (self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]))
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat = mint.add(f_hat, h)
            return f_hat, f_hat


class Phi(mint.nn.Conv2d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=ks, stride=1, padding=ks // 2)
        self.resi_ratio = abs(quant_resi)

    def construct(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().construct(h_BChw).mul(self.resi_ratio)


class PhiShared(nn.Cell):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Cell):
    def __init__(self, qresi_ls: nn.CellList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


class PhiNonShared(nn.CellList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K) if K == 4 else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"
