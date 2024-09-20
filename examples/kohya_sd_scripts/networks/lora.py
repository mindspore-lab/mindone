# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
import re
from typing import Dict, List, Optional, Tuple, Type, Union

from library.utils import setup_logging

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.common.initializer import HeUniform, Zero, initializer

from mindone.diffusers import AutoencoderKL
from mindone.transformers import CLIPTextModel

setup_logging()
import logging

logger = logging.getLogger(__name__)

RE_UPDOWN = re.compile(r"(up|down)_blocks_(\d+)_(resnets|upsamplers|downsamplers|attentions)_(\d+)_")


def zeros_(tensor: Tensor) -> None:
    tensor.set_data(initializer(Zero(), tensor.shape, tensor.dtype))


def kaiming_uniform_(tensor: Tensor, a=0.0, mode="fan_in", nonlinearity="leaky_relu") -> None:
    tensor.set_data(initializer(HeUniform(a, mode, nonlinearity), tensor.shape, tensor.dtype))


def parse_block_lr_kwargs(nw_kwargs):
    down_lr_weight = nw_kwargs.get("down_lr_weight", None)
    mid_lr_weight = nw_kwargs.get("mid_lr_weight", None)
    up_lr_weight = nw_kwargs.get("up_lr_weight", None)

    # 以上のいずれにも設定がない場合は無効としてNoneを返す
    if down_lr_weight is None and mid_lr_weight is None and up_lr_weight is None:
        return None, None, None

    # extract learning rate weight for each block
    if down_lr_weight is not None:
        # if some parameters are not set, use zero
        if "," in down_lr_weight:
            down_lr_weight = [(float(s) if s else 0.0) for s in down_lr_weight.split(",")]

    if mid_lr_weight is not None:
        mid_lr_weight = float(mid_lr_weight)

    if up_lr_weight is not None:
        if "," in up_lr_weight:
            up_lr_weight = [(float(s) if s else 0.0) for s in up_lr_weight.split(",")]

    down_lr_weight, mid_lr_weight, up_lr_weight = get_block_lr_weight(
        down_lr_weight, mid_lr_weight, up_lr_weight, float(nw_kwargs.get("block_lr_zero_threshold", 0.0))
    )

    return down_lr_weight, mid_lr_weight, up_lr_weight


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: AutoencoderKL,
    text_encoder: Union[CLIPTextModel, List[CLIPTextModel]],
    unet,
    neuron_dropout: Optional[float] = None,
    train_unet: Optional[bool] = True,
    train_text_encoder: Optional[bool] = False,
    original_dtype=ms.float32,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # block dim/alpha/lr
    block_dims = kwargs.get("block_dims", None)
    down_lr_weight, mid_lr_weight, up_lr_weight = parse_block_lr_kwargs(kwargs)

    # 以上のいずれかに指定があればblockごとのdim(rank)を有効にする
    if block_dims is not None or down_lr_weight is not None or mid_lr_weight is not None or up_lr_weight is not None:
        block_alphas = kwargs.get("block_alphas", None)
        conv_block_dims = kwargs.get("conv_block_dims", None)
        conv_block_alphas = kwargs.get("conv_block_alphas", None)

        block_dims, block_alphas, conv_block_dims, conv_block_alphas = get_block_dims_and_alphas(
            block_dims,
            block_alphas,
            network_dim,
            network_alpha,
            conv_block_dims,
            conv_block_alphas,
            conv_dim,
            conv_alpha,
        )

        # remove block dim/alpha without learning rate
        block_dims, block_alphas, conv_block_dims, conv_block_alphas = remove_block_dims_and_alphas(
            block_dims, block_alphas, conv_block_dims, conv_block_alphas, down_lr_weight, mid_lr_weight, up_lr_weight
        )

    else:
        block_alphas = None
        conv_block_dims = None
        conv_block_alphas = None

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # すごく引数が多いな ( ^ω^)･･･
    network = LoRANetwork(
        text_encoder,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        block_dims=block_dims,
        block_alphas=block_alphas,
        conv_block_dims=conv_block_dims,
        conv_block_alphas=conv_block_alphas,
        varbose=True,
        train_unet=train_unet,
        train_text_encoder=train_text_encoder,
        original_dtype=original_dtype,
    )

    if up_lr_weight is not None or mid_lr_weight is not None or down_lr_weight is not None:
        network.set_block_lr_weight(up_lr_weight, mid_lr_weight, down_lr_weight)

    return network


# このメソッドは外部から呼び出される可能性を考慮しておく
# network_dim, network_alpha にはデフォルト値が入っている。
# block_dims, block_alphas は両方ともNoneまたは両方とも値が入っている
# conv_dim, conv_alpha は両方ともNoneまたは両方とも値が入っている
def get_block_dims_and_alphas(
    block_dims, block_alphas, network_dim, network_alpha, conv_block_dims, conv_block_alphas, conv_dim, conv_alpha
):
    num_total_blocks = LoRANetwork.NUM_OF_BLOCKS * 2 + 1

    def parse_ints(s):
        return [int(i) for i in s.split(",")]

    def parse_floats(s):
        return [float(i) for i in s.split(",")]

    # block_dimsとblock_alphasをパースする。必ず値が入る
    if block_dims is not None:
        block_dims = parse_ints(block_dims)
        assert (
            len(block_dims) == num_total_blocks
        ), f"block_dims must have {num_total_blocks} elements / block_dimsは{num_total_blocks}個指定してください"
    else:
        logger.warning(
            f"block_dims is not specified. all dims are set to {network_dim} / block_dimsが指定されていません。すべてのdimは{network_dim}になります"
        )
        block_dims = [network_dim] * num_total_blocks

    if block_alphas is not None:
        block_alphas = parse_floats(block_alphas)
        assert (
            len(block_alphas) == num_total_blocks
        ), f"block_alphas must have {num_total_blocks} elements / block_alphasは{num_total_blocks}個指定してください"
    else:
        logger.warning(
            f"block_alphas is not specified. all alphas are set to {network_alpha} / block_alphasが指定されていません。すべてのalphaは{network_alpha}になります"
        )
        block_alphas = [network_alpha] * num_total_blocks

    # conv_block_dimsとconv_block_alphasを、指定がある場合のみパースする。指定がなければconv_dimとconv_alphaを使う
    if conv_block_dims is not None:
        conv_block_dims = parse_ints(conv_block_dims)
        assert (
            len(conv_block_dims) == num_total_blocks
        ), f"conv_block_dims must have {num_total_blocks} elements / conv_block_dimsは{num_total_blocks}個指定してください"

        if conv_block_alphas is not None:
            conv_block_alphas = parse_floats(conv_block_alphas)
            assert (
                len(conv_block_alphas) == num_total_blocks
            ), f"conv_block_alphas must have {num_total_blocks} elements / conv_block_alphasは{num_total_blocks}個指定してください"
        else:
            if conv_alpha is None:
                conv_alpha = 1.0
            logger.warning(
                f"conv_block_alphas is not specified. all alphas are set to {conv_alpha} / conv_block_alphasが指定されていません。すべてのalphaは{conv_alpha}になります"
            )
            conv_block_alphas = [conv_alpha] * num_total_blocks
    else:
        if conv_dim is not None:
            logger.warning(
                f"conv_dim/alpha for all blocks are set to {conv_dim} and {conv_alpha} / すべてのブロックのconv_dimとalphaは{conv_dim}および{conv_alpha}になります"
            )
            conv_block_dims = [conv_dim] * num_total_blocks
            conv_block_alphas = [conv_alpha] * num_total_blocks
        else:
            conv_block_dims = None
            conv_block_alphas = None

    return block_dims, block_alphas, conv_block_dims, conv_block_alphas


# 層別学習率用に層ごとの学習率に対する倍率を定義する、外部から呼び出される可能性を考慮しておく
def get_block_lr_weight(
    down_lr_weight, mid_lr_weight, up_lr_weight, zero_threshold
) -> Tuple[List[float], List[float], List[float]]:
    # パラメータ未指定時は何もせず、今までと同じ動作とする
    if up_lr_weight is None and mid_lr_weight is None and down_lr_weight is None:
        return None, None, None

    max_len = LoRANetwork.NUM_OF_BLOCKS  # フルモデル相当でのup,downの層の数

    def get_list(name_with_suffix) -> List[float]:
        import math

        tokens = name_with_suffix.split("+")
        name = tokens[0]
        base_lr = float(tokens[1]) if len(tokens) > 1 else 0.0

        if name == "cosine":
            return [math.sin(math.pi * (i / (max_len - 1)) / 2) + base_lr for i in reversed(range(max_len))]
        elif name == "sine":
            return [math.sin(math.pi * (i / (max_len - 1)) / 2) + base_lr for i in range(max_len)]
        elif name == "linear":
            return [i / (max_len - 1) + base_lr for i in range(max_len)]
        elif name == "reverse_linear":
            return [i / (max_len - 1) + base_lr for i in reversed(range(max_len))]
        elif name == "zeros":
            return [0.0 + base_lr] * max_len
        else:
            logger.error(
                "Unknown lr_weight argument %s is used. Valid arguments:  / 不明なlr_weightの引数 %s が使われました。有効な引数:\n\tcosine, sine, linear, reverse_linear, zeros"
                % (name)
            )
            return None

    if type(down_lr_weight) == str:
        down_lr_weight = get_list(down_lr_weight)
    if type(up_lr_weight) == str:
        up_lr_weight = get_list(up_lr_weight)

    if (up_lr_weight is not None and len(up_lr_weight) > max_len) or (
        down_lr_weight is not None and len(down_lr_weight) > max_len
    ):
        logger.warning("down_weight or up_weight is too long. Parameters after %d-th are ignored." % max_len)
        logger.warning("down_weightもしくはup_weightが長すぎます。%d個目以降のパラメータは無視されます。" % max_len)
        up_lr_weight = up_lr_weight[:max_len]
        down_lr_weight = down_lr_weight[:max_len]

    if (up_lr_weight is not None and len(up_lr_weight) < max_len) or (
        down_lr_weight is not None and len(down_lr_weight) < max_len
    ):
        logger.warning("down_weight or up_weight is too short. Parameters after %d-th are filled with 1." % max_len)
        logger.warning("down_weightもしくはup_weightが短すぎます。%d個目までの不足したパラメータは1で補われます。" % max_len)

        if down_lr_weight is not None and len(down_lr_weight) < max_len:
            down_lr_weight = down_lr_weight + [1.0] * (max_len - len(down_lr_weight))
        if up_lr_weight is not None and len(up_lr_weight) < max_len:
            up_lr_weight = up_lr_weight + [1.0] * (max_len - len(up_lr_weight))

    if (up_lr_weight is not None) or (mid_lr_weight is not None) or (down_lr_weight is not None):
        logger.info("apply block learning rate / 階層別学習率を適用します。")
        if down_lr_weight is not None:
            down_lr_weight = [w if w > zero_threshold else 0 for w in down_lr_weight]
            logger.info(f"down_lr_weight (shallower -> deeper, 浅い層->深い層): {down_lr_weight}")
        else:
            logger.info("down_lr_weight: all 1.0, すべて1.0")

        if mid_lr_weight is not None:
            mid_lr_weight = mid_lr_weight if mid_lr_weight > zero_threshold else 0
            logger.info(f"mid_lr_weight: {mid_lr_weight}")
        else:
            logger.info("mid_lr_weight: 1.0")

        if up_lr_weight is not None:
            up_lr_weight = [w if w > zero_threshold else 0 for w in up_lr_weight]
            logger.info(f"up_lr_weight (deeper -> shallower, 深い層->浅い層): {up_lr_weight}")
        else:
            logger.info("up_lr_weight: all 1.0, すべて1.0")

    return down_lr_weight, mid_lr_weight, up_lr_weight


# lr_weightが0のblockをblock_dimsから除外する、外部から呼び出す可能性を考慮しておく
def remove_block_dims_and_alphas(
    block_dims, block_alphas, conv_block_dims, conv_block_alphas, down_lr_weight, mid_lr_weight, up_lr_weight
):
    # set 0 to block dim without learning rate to remove the block
    if down_lr_weight is not None:
        for i, lr in enumerate(down_lr_weight):
            if lr == 0:
                block_dims[i] = 0
                if conv_block_dims is not None:
                    conv_block_dims[i] = 0
    if mid_lr_weight is not None:
        if mid_lr_weight == 0:
            block_dims[LoRANetwork.NUM_OF_BLOCKS] = 0
            if conv_block_dims is not None:
                conv_block_dims[LoRANetwork.NUM_OF_BLOCKS] = 0
    if up_lr_weight is not None:
        for i, lr in enumerate(up_lr_weight):
            if lr == 0:
                block_dims[LoRANetwork.NUM_OF_BLOCKS + 1 + i] = 0
                if conv_block_dims is not None:
                    conv_block_dims[LoRANetwork.NUM_OF_BLOCKS + 1 + i] = 0

    return block_dims, block_alphas, conv_block_dims, conv_block_alphas


# 外部から呼び出す可能性を考慮しておく
def get_block_index(lora_name: str) -> int:
    block_idx = -1  # invalid lora name

    m = RE_UPDOWN.search(lora_name)
    if m:
        g = m.groups()
        i = int(g[1])
        j = int(g[3])
        if g[2] == "resnets":
            idx = 3 * i + j
        elif g[2] == "attentions":
            idx = 3 * i + j
        elif g[2] == "upsamplers" or g[2] == "downsamplers":
            idx = 3 * i + 2

        if g[0] == "down":
            block_idx = 1 + idx  # 0に該当するLoRAは存在しない
        elif g[0] == "up":
            block_idx = LoRANetwork.NUM_OF_BLOCKS + 1 + idx

    elif "mid_block_" in lora_name:
        block_idx = LoRANetwork.NUM_OF_BLOCKS  # idx=12

    return block_idx


# modified from original repo `LoRAInfModule.merge_to`
# freezeしてマージする
def _merge_to(n, p, lora_weights_sd, multiplier):
    lora_name = n[: -len("weight")]
    dim = lora_weights_sd[lora_name + "lora_down.weight"].shape[0]

    # support old LoRA without alpha
    alpha = lora_weights_sd.get(lora_name + "alpha", dim)
    scale = alpha / dim

    # get up/down weight
    up_weight = lora_weights_sd[lora_name + "lora_up.weight"].float()
    down_weight = lora_weights_sd[lora_name + "lora_down.weight"].float()

    # extract weight from org_module
    weight = p.float()

    # merge weight
    if len(weight.shape) == 2:
        # linear
        weight = weight + multiplier * (up_weight @ down_weight) * scale
    elif down_weight.shape[2:4] == (1, 1):
        # conv2d 1x1
        weight = (
            weight
            + multiplier
            * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
            * scale
        )
    else:
        # conv2d 3x3
        conved = ops.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
        # logger.info(conved.size(), weight.size(), module.stride, module.padding)
        weight = weight + multiplier * conved * scale

    return weight


# org kohya: Create network from weights for inference, weights are not loaded here (because can be merged)
# Notes: lora weights are directly loaded and merged for inference, without create lora network
def create_network_from_weights(
    multiplier,
    file,
    vae,
    text_encoder,
    unet,
    weights_sd=None,
    dtype=ms.float32,
    **kwargs,
):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd_safetensors = load_file(file)
            weights_sd = {}
            # turn the torch key : lora_unet_xxx_xx_x_weight
            # to ms key : lora_unet.xxx.xx.x.weight
            for key, value in weights_sd_safetensors.items():
                if key.startswith("lora_unet_"):
                    key.replace("lora_unet_", "lora_unet.")
                elif key.startswith("lora_te1_"):
                    key.replace("lora_te1_", "lora_te1.")
                elif key.startswith("lora_te2_"):
                    key.replace("lora_te2_", "lora_te2.")
                elif key.startswith("lora_te_"):
                    key.replace("lora_te_", "lora_te.")
                else:
                    continue
                tmp_l = key.split(".")
                tmp_l[1] = tmp_l.replace("_", ".")
                key = ".".join(tmp_l)
                weights_sd[key] = ms.Tensor(value.numpy())
            del weights_sd_safetensors

        elif os.path.splitext(file)[1] == ".ckpt":
            weights_sd = ms.load_checkpoint(file)

        else:
            NotImplementedError

    # place the weights for unet and te
    unet_lora_weights_sd, unet_replaced_keys = {}, set()
    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    # for sd
    te_lora_weights_sd, te_replaced_keys = {}, set()
    # for sdxl
    te1_lora_weights_sd, te1_replaced_keys = {}, set()
    te2_lora_weights_sd, te2_replaced_keys = {}, set()

    for key, value in weights_sd.items():
        # unet
        if key.startswith("lora_unet"):
            key = key[len("lora_unet.") :]
            unet_lora_weights_sd[key] = value
            # find the key of original module and add to the set
            if key.endswith(".lora_down.weight"):
                unet_replaced_keys.add(key[: -len("lora_down.weight")] + "weight")

        # sdxl text encoders
        elif key.startswith("lora_te1"):
            key = key[len("lora_te1.") :]
            te1_lora_weights_sd[key] = value
            # find the key of original module and add to the set
            if key.endswith(".lora_down.weight"):
                te1_replaced_keys.add(key[: -len("lora_down.weight")] + "weight")
        elif key.startswith("lora_te2"):
            key = key[len("lora_te2.") :]
            te2_lora_weights_sd[key] = value
            # find the key of original module and add to the set
            if key.endswith(".lora_down.weight"):
                te2_replaced_keys.add(key[: -len("lora_down.weight")] + "weight")

        # sd text encoder
        elif key.startswith("lora_te"):
            key = key[len("lora_te.") :]
            te_lora_weights_sd[key] = value
            # find the key of original module and add to the set
            if key.endswith(".lora_down.weight"):
                te_replaced_keys.add(key[: -len("lora_down.weight")] + "weight")

    # merge unet
    for n, p in unet.parameters_and_names():
        if n in unet_replaced_keys:
            merged_weight = _merge_to(n, p, unet_lora_weights_sd, multiplier=multiplier)
            # set weights to org_module
            merged_weight = merged_weight.to(dtype=dtype)
            p.set_data(merged_weight)

    # merge textencoders
    for idx, te in enumerate(text_encoders):
        if len(text_encoders) == 2:
            replaced_keys = te1_replaced_keys if idx == 0 else te2_replaced_keys
            lora_weights_sd = te1_lora_weights_sd if idx == 0 else te2_lora_weights_sd
        else:
            replaced_keys = te_replaced_keys
            lora_weights_sd = te_lora_weights_sd

        for n, p in te.parameters_and_names():
            if n in replaced_keys:
                merged_weight = _merge_to(n, p, lora_weights_sd, multiplier=multiplier)
                # set weights to org_module
                merged_weight = merged_weight.to(dtype=dtype)
                p.set_data(merged_weight)

    logger.info("weights are merged")


class LoRAModule(nn.Cell):
    """
    original kohya: replaces forward method of the original Linear, instead of replacing the original Linear module.
    mindspore version: replace the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Cell,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        org_dtype=ms.float32,
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__(auto_prefix=False)
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        nn.Dense

        if isinstance(org_module, nn.Conv2d):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        elif isinstance(org_module, nn.Dense):
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            NotImplementedError

        # 1. init sub-module
        if isinstance(org_module, nn.Conv2d):
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding=padding, has_bias=False)
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), has_bias=False)
        elif isinstance(org_module, nn.Dense):
            self.lora_down = nn.Dense(in_dim, self.lora_dim, has_bias=False)
            self.lora_up = nn.Dense(self.lora_dim, out_dim, has_bias=False)
        else:
            NotImplementedError

        self.org_module = org_module
        # notes: make sure the dtype of params of org_modules remain the same
        for p in self.org_module.get_parameters():
            p.set_dtype(org_dtype)

        # 2. modify prefix-name
        prefix_name = lora_name + "."
        self.lora_down.weight.name = prefix_name + "lora_down.weight"
        self.lora_up.weight.name = prefix_name + "lora_up.weight"

        # 3. init lora down/up weights
        # same as microsoft's
        kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        zeros_(self.lora_up.weight)

        # 4. init hyper-param
        if isinstance(alpha, Tensor):
            alpha = alpha.float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        # self.alpha = Tensor(alpha)  # 定数として扱える
        # kohya's torch implement saves the alpha in the state_dict
        self.alpha = ms.Parameter(ms.Tensor(alpha), name=prefix_name + "alpha", requires_grad=False)

        self.multiplier = multiplier
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    # notes: we remove the apply_to method in mindspore implements

    def construct(self, x):
        org_forwarded = self.org_module(x)

        # module dropout
        if self.module_dropout is not None and self.training:
            if ops.rand(1) < self.module_dropout:
                return org_forwarded

        lx = self.lora_down(x)

        # normal dropout
        if self.dropout is not None and self.training:
            lx = ops.dropout(lx, p=self.dropout)

        # rank dropout
        if self.rank_dropout is not None and self.training:
            mask = ops.rand((lx.shape(0), self.lora_dim)) > self.rank_dropout
            if len(lx.shape()) == 3:
                mask = mask.unsqueeze(1)  # for Text Encoder
            elif len(lx.shape()) == 4:
                mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
            lx = lx * mask

            # scaling for rank dropout: treat as if the rank is changed
            # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
            scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
        else:
            scale = self.scale

        lx = self.lora_up(lx)

        return org_forwarded + lx * self.multiplier * scale


class LoRANetwork(nn.Cell):
    NUM_OF_BLOCKS = 12  # フルモデル相当でのup,downの層の数

    UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
    UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    # SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
    LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

    def __init__(
        self,
        text_encoder: Union[List[CLIPTextModel], CLIPTextModel],
        unet,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        block_dims: Optional[List[int]] = None,
        block_alphas: Optional[List[float]] = None,
        conv_block_dims: Optional[List[int]] = None,
        conv_block_alphas: Optional[List[float]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        module_class: Type[object] = LoRAModule,
        varbose: Optional[bool] = False,
        train_unet: Optional[bool] = True,
        train_text_encoder: Optional[bool] = False,
        original_dtype=ms.float32,
    ) -> None:
        """
        LoRA network: すごく引数が多いが、パターンは以下の通り
        1. lora_dimとalphaを指定
        2. lora_dim、alpha、conv_lora_dim、conv_alphaを指定
        3. block_dimsとblock_alphasを指定 :  Conv2d3x3には適用しない
        4. block_dims、block_alphas、conv_block_dims、conv_block_alphasを指定 : Conv2d3x3にも適用する
        5. modules_dimとmodules_alphaを指定 (推論用)
        """
        super().__init__(auto_prefix=False)
        self.unet = unet
        if isinstance(text_encoder, list):
            self.text_encoder1 = text_encoder[0]
            self.text_encoder2 = text_encoder[1]
            text_encoders = [self.text_encoder1, self.text_encoder2]
        else:
            self.text_encoder = text_encoder
            text_encoders = [self.text_encoder]
        self.text_encoders = text_encoders

        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        elif block_dims is not None:
            logger.info("create LoRA network from block_dims")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            logger.info(f"block_dims: {block_dims}")
            logger.info(f"block_alphas: {block_alphas}")
            if conv_block_dims is not None:
                logger.info(f"conv_block_dims: {conv_block_dims}")
                logger.info(f"conv_block_alphas: {conv_block_alphas}")
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            if self.conv_lora_dim is not None:
                logger.info(
                    f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}"
                )

        def _replace_modules(is_unet: bool, module: nn.Cell, lora_name):
            replaced = []
            skipped = []

            for child_name, child_module in module.name_cells().items():
                is_linear = child_module.__class__.__name__ == "Dense"
                is_conv2d = child_module.__class__.__name__ == "Conv2d"
                is_conv2d_1x1 = is_conv2d and child_module.kernel_size == 1

                lora_name = lora_name + "." + child_name
                lora_name = lora_name.replace(".", "_")

                if is_linear or is_conv2d:
                    dim = None
                    alpha = None

                    if modules_dim is not None:
                        # モジュール指定あり
                        if lora_name in modules_dim:
                            dim = modules_dim[lora_name]
                            alpha = modules_alpha[lora_name]
                    elif is_unet and block_dims is not None:
                        # U-Netでblock_dims指定あり
                        block_idx = get_block_index(lora_name)
                        if is_linear or is_conv2d_1x1:
                            dim = block_dims[block_idx]
                            alpha = block_alphas[block_idx]
                        elif conv_block_dims is not None:
                            dim = conv_block_dims[block_idx]
                            alpha = conv_block_alphas[block_idx]
                    else:
                        # 通常、すべて対象とする
                        if is_linear or is_conv2d_1x1:
                            dim = self.lora_dim
                            alpha = self.alpha
                        elif self.conv_lora_dim is not None:
                            dim = self.conv_lora_dim
                            alpha = self.conv_alpha

                    if dim is None or dim == 0:
                        # skipした情報を出力
                        if (
                            is_linear
                            or is_conv2d_1x1
                            or (self.conv_lora_dim is not None or conv_block_dims is not None)
                        ):
                            skipped.append(lora_name)
                        continue

                    lora = module_class(
                        lora_name,
                        child_module,
                        self.multiplier,
                        dim,
                        alpha,
                        dropout=dropout,
                        rank_dropout=rank_dropout,
                        module_dropout=module_dropout,
                        org_dtype=original_dtype,
                    )
                    replaced.append(lora_name)
                    module._cells[child_name] = lora

                else:
                    _replaced, _skipped = _replace_modules(is_unet, child_module, lora_name)
                    replaced += _replaced
                    skipped += _skipped

            return replaced, skipped

        # notes: org repo create loras from target modules,
        # here we create the loras and directly replace the layers.
        # modified from `create_modules` method
        def replace_modules(
            is_unet: bool,
            text_encoder_idx: Optional[int],  # None, 1, 2
            root_module: nn.Cell,
            target_replace_modules: List[nn.Cell],
        ) -> List[LoRAModule]:
            prefix = (
                self.LORA_PREFIX_UNET
                if is_unet
                else (
                    self.LORA_PREFIX_TEXT_ENCODER
                    if text_encoder_idx is None
                    else (self.LORA_PREFIX_TEXT_ENCODER1 if text_encoder_idx == 1 else self.LORA_PREFIX_TEXT_ENCODER2)
                )
            )
            replaced = []
            skipped = []
            to_replace_module = {
                name: module
                for name, module in root_module.cells_and_names()
                if module.__class__.__name__ in target_replace_modules
            }

            # for name, module in root_module.named_modules():
            for name, module in to_replace_module.items():
                # if module.__class__.__name__ in target_replace_modules:
                lora_name = prefix + "." + name
                lora_name = lora_name.replace(".", "_")

                _replaced, _skipped = _replace_modules(is_unet, module, lora_name)

                replaced += _replaced
                skipped += _skipped

            return replaced, skipped

        # create LoRA for text encoder
        # 毎回すべてのモジュールを作るのは無駄なので要検討
        replaced_te, skipped_te = [], []
        if train_text_encoder:
            text_encoders = text_encoder if type(text_encoder) == list else [text_encoder]
            for i, text_encoder in enumerate(text_encoders):
                if len(text_encoders) > 1:
                    index = i + 1
                    logger.info(f"create LoRA for Text Encoder {index}:")
                else:
                    index = None
                    logger.info("create LoRA for Text Encoder:")

                text_encoder_loras, skipped = replace_modules(
                    False, index, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE
                )
                replaced_te.extend(text_encoder_loras)
                skipped_te += skipped
            logger.info(f"create LoRA for Text Encoder: {len(replaced_te)} modules.")

        # extend U-Net target modules if conv2d 3x3 is enabled, or load from weights
        target_modules = LoRANetwork.UNET_TARGET_REPLACE_MODULE
        if modules_dim is not None or self.conv_lora_dim is not None or conv_block_dims is not None:
            target_modules += LoRANetwork.UNET_TARGET_REPLACE_MODULE_CONV2D_3X3

        replaced_un, skipped_un = replace_modules(True, None, unet, target_modules)
        logger.info(f"create LoRA for U-Net: {len(replaced_un)} modules.")

        self.set_required_grad()
        logger.info("set lora params trainable and freezes other (required grads false)")

        # 3. verbose
        skipped = skipped_te + skipped_un
        if varbose and len(skipped) > 0:
            logger.warning(
                f"because block_lr_weight is 0 or dim (rank) is 0, {len(skipped)} LoRA modules are skipped \
                    / block_lr_weightまたはdim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                logger.info(f"\t{name}")

        self.up_lr_weight: List[float] = None
        self.down_lr_weight: List[float] = None
        self.mid_lr_weight: float = None
        self.block_lr = False

    def to(self, dtype: Optional[ms.Type] = None):
        for p in self.get_parameters():
            p.set_dtype(dtype)
        return self

    def set_required_grad(self):
        for n, p in self.parameters_and_names():
            if n.endswith("lora_down.weight") or n.endswith("lora_up.weight"):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def set_multiplier(self, multiplier):
        # TODO, set multiplier to Parameters
        pass

    def load_weights(self, file):
        # TODO, load in train scripts, not support yet
        weights_sd = ms.load_checkpoint(file)
        param_not_load, _ = ms.load_param_into_net(self, weights_sd)
        return param_not_load

    # マージできるかどうかを返す
    def is_mergeable(self):
        return True

    # 層別学習率用に層ごとの学習率に対する倍率を定義する　引数の順番が逆だがとりあえず気にしない
    def set_block_lr_weight(
        self,
        up_lr_weight: List[float] = None,
        mid_lr_weight: float = None,
        down_lr_weight: List[float] = None,
    ):
        self.block_lr = True
        self.down_lr_weight = down_lr_weight
        self.mid_lr_weight = mid_lr_weight
        self.up_lr_weight = up_lr_weight

    def get_lr_weight(self, lora: LoRAModule) -> float:
        lr_weight = 1.0
        block_idx = get_block_index(lora.lora_name)
        if block_idx < 0:
            return lr_weight

        if block_idx < LoRANetwork.NUM_OF_BLOCKS:
            if self.down_lr_weight is not None:
                lr_weight = self.down_lr_weight[block_idx]
        elif block_idx == LoRANetwork.NUM_OF_BLOCKS:
            if self.mid_lr_weight is not None:
                lr_weight = self.mid_lr_weight
        elif block_idx > LoRANetwork.NUM_OF_BLOCKS:
            if self.up_lr_weight is not None:
                lr_weight = self.up_lr_weight[block_idx - LoRANetwork.NUM_OF_BLOCKS - 1]

        return lr_weight

    # 二つのText Encoderに別々の学習率を設定できるようにするといいかも
    def prepare_optimizer_params(self, text_encoder_lr, unet_lr, default_lr):
        # FIXME: not supported yet
        pass

    def enable_gradient_checkpointing(self):
        # org kohya: not supported
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        for p in self.get_parameters():
            p.requires_grad = True

    def on_epoch_start(self, text_encoder, unet):
        self.set_train()

    def get_trainable_params(self):
        # return self.parameters()
        return self.get_parameters()

    def save_weights(self, file, dtype, metadata=None):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        assert os.path.splitext(file)[1] == ".ckpt"

        ckpt_lora = []
        for n, p in self.unet.parameters_and_names():
            if ".lora_" in n or ".alpha" in n:
                ckpt_lora.append(
                    {"name": self.LORA_PREFIX_UNET + "." + n, "data": p if dtype is None else p.to(dtype=dtype)}
                )

        for idx, te in enumerate(self.text_encoders):
            # prefix
            if len(self.text_encoders) == 1:
                prefix = self.LORA_PREFIX_TEXT_ENCODER
            else:
                prefix = self.LORA_PREFIX_TEXT_ENCODER1 if idx == 0 else self.LORA_PREFIX_TEXT_ENCODER2
            # save
            for n, p in te.parameters_and_names():
                if ".lora_" in n or ".alpha" in n:
                    ckpt_lora.append({"name": prefix + "." + n, "data": p if dtype is None else p.to(dtype=dtype)})

        ms.save_checkpoint(ckpt_lora, file, append_dict=metadata)

    def apply_max_norm_regularization(self, max_norm_value):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = {k: v for k, v in self.parameters_and_names()}
        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]]
            up = state_dict[upkeys[i]]
            alpha = state_dict[alphakeys[i]]
            dim = down.shape[0]
            scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = ops.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = ops.clamp(norm, max=max_norm_value)
            ratio = desired / norm
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm() * ratio
            norms.append(scalednorm.item())

        return keys_scaled, sum(norms) / len(norms), max(norms)
