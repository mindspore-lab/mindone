from typing import Tuple

from models import VAR, VQVAE

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.diffusers.training_utils import pynative_no_grad as no_grad


class GeneratorWithLoss(nn.Cell):
    def __init__(self, patch_nums: Tuple[int, ...], vae_local: VQVAE, var: VAR, label_smooth: float):
        super().__init__()
        self.vae_local = vae_local
        self.quantize_local = vae_local.quantize
        self.var = var
        self.train_loss = mint.nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction="none")
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = mint.ones((1, self.L)) / self.L
        self.patch_nums = patch_nums
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn * pn

        self.prog_it = ms.Tensor(0)
        self.last_prog_si = ms.Tensor(-1)
        self.first_prog = True

    def construct(self, inp_B3HW, label_B, prog_si, prog_wp_it):
        self.prog_it = self.prog_it.add(1)
        prog_wp = mint.maximum(mint.minimum(self.prog_it / prog_wp_it, 1), 0.01)
        if self.first_prog:
            prog_wp = 1  # no prog warmup at first prog stage, as it's already solved in wp
        if prog_si == len(self.patch_nums) - 1:
            prog_si = -1  # max prog, as if no prog

        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size

        with no_grad():
            gt_idx_Bl = ops.stop_gradient(self.vae_local.img_to_idxBl(inp_B3HW))
            x_BLCv_wo_first_l = ops.stop_gradient(self.quantize_local.idxBl_to_var_input(gt_idx_Bl))
        gt_BL = mint.cat(gt_idx_Bl, dim=1)
        logits_BLV = self.var(label_B, x_BLCv_wo_first_l)
        loss = self.train_loss(logits_BLV.view((-1, V)), gt_BL.view((-1))).view((B, -1))
        if prog_si >= 0:  # in progressive training
            bg, ed = self.begin_ends[prog_si]
            assert logits_BLV.shape[1] == gt_BL.shape[1] == ed
            lw = self.loss_weight[:, :ed].clone()
            lw[:, bg:ed] *= min(max(prog_wp, 0), 1)
        else:  # not in progressive training
            lw = self.loss_weight
        loss = loss.mul(lw).sum(dim=-1).mean()
        return loss
