import importlib
import itertools
import logging

import numpy as np
import torch
from diffusers.utils import BaseOutput
from ml_dtypes import bfloat16

import mindspore as ms
from mindspore import nn, ops

logger = logging.getLogger("ModelingsUnitTest")


FULLY_CASES_ARGS = "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,dtype,mode"
FULLY_CASES_LENGTH = 9
TORCH_FP16_BLACKLIST = (
    "LayerNorm",
    "Timesteps",
    "AvgPool2d",
    "Upsample2D",
    "ResnetBlock2D",
    "FirUpsample2D",
    "FirDownsample2D",
    "KDownsample2D",
    "AutoencoderTiny",
)
MS_FP16_WHITELIST = (nn.Conv3d,)
MS_BF16_BLACKLIST = (nn.Loss,)
PT_DTYPE_MAPPING = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}
MS_DTYPE_MAPPING = {"fp16": ms.float16, "fp32": ms.float32, "bf16": ms.bfloat16}
NP_DTYPE_MAPPING = {"fp16": np.float16, "fp32": np.float32, "bf16": bfloat16}


class _OutputTo(nn.Cell):
    """Wrap cell for amp. Cast network output back to dtype."""

    def __init__(self, backbone, dtype=ms.float16):
        super(_OutputTo, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.dtype = dtype
        self._get_attr_from_cell(backbone)

    def construct(self, *args, **kwargs):
        return self.backbone(*args, **kwargs).to(self.dtype)


def _ms_mixed_precision(net, dtype):
    ms_dtype = MS_DTYPE_MAPPING[dtype]
    cells = net.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == net:
            continue
        if ms_dtype == ms.bfloat16 and isinstance(subcell, tuple(MS_BF16_BLACKLIST)):
            set_dtype(subcell, ms.float32)
            net._cells[name] = _OutputTo(subcell.to_float(ms.float32), ms.bfloat16)
            change = True
        if ms_dtype != ms.float16 and isinstance(subcell, tuple(MS_FP16_WHITELIST)):
            set_dtype(subcell, ms.float16)
            net._cells[name] = _OutputTo(subcell.to_float(ms.float16), ms_dtype)
            change = True
        else:
            _ms_mixed_precision(subcell, dtype)
    if isinstance(net, nn.SequentialCell) and change:
        net.cell_list = list(net.cells())
    return net


def set_default_dtype_mode_for_single_case(case):
    assert len(case) in (
        FULLY_CASES_LENGTH,
        FULLY_CASES_LENGTH - 2,
    ), (
        f"Case should contain all arguments({FULLY_CASES_ARGS}), or all arguments except for the last two(dtype and mode). "
        f"In the latter situation, dtype will be set to `(fp16, fp32)` by default, and mode will be set to `(0, 1)` by default."
    )

    if len(case) == FULLY_CASES_LENGTH - 2:
        case += [("fp16", "fp32"), (0, 1)]

    return case


def expand_dtype_mode_for_all_case(all_cases):
    all_cases = map(set_default_dtype_mode_for_single_case, all_cases)
    expanded_cases = []

    for case in all_cases:
        case_wo_dtype_mode, dtype, mode = case[:-2], case[-2], case[-1]
        expanded_cases.extend([case_wo_dtype_mode + list(context) for context in itertools.product(dtype, mode)])

    return expanded_cases


# copied from mindone.diffusers.models.modeling_utils
def get_pt2ms_mappings(m):
    mappings = {}  # pt_param_name: (ms_param_name, pt_param_to_ms_param_func)
    for name, cell in m.cells_and_names():
        if isinstance(cell, (nn.Conv1d, nn.Conv1dTranspose)):
            mappings[f"{name}.weight"] = f"{name}.weight", lambda x: ms.Parameter(
                ops.expand_dims(x, axis=-2), name=f"{name}.weight"
            )
        elif isinstance(cell, nn.Embedding):
            mappings[f"{name}.weight"] = f"{name}.embedding_table", lambda x: x
        elif isinstance(cell, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            mappings[f"{name}.weight"] = f"{name}.gamma", lambda x: x
            mappings[f"{name}.bias"] = f"{name}.beta", lambda x: x
            if isinstance(cell, (nn.BatchNorm2d,)):
                mappings[f"{name}.running_mean"] = f"{name}.moving_mean", lambda x: x
                mappings[f"{name}.running_var"] = f"{name}.moving_variance", lambda x: x
                mappings[f"{name}.num_batches_tracked"] = None, lambda x: x
    return mappings


# adapted from mindone.diffusers.models.modeling_utils
def convert_state_dict(m, state_dict_pt):
    dtype_mappings = {
        torch.float16: ms.float16,
        torch.float32: ms.float32,
        torch.bfloat16: ms.bfloat16,
    }

    mappings = get_pt2ms_mappings(m)
    state_dict_ms = {}
    for name_pt, data_pt in state_dict_pt.items():
        name_ms, data_mapping = mappings.get(name_pt, (name_pt, lambda x: x))
        data_ms = ms.Parameter(
            data_mapping(ms.Tensor.from_numpy(data_pt.float().numpy()).to(dtype_mappings[data_pt.dtype])), name=name_ms
        )
        if name_ms is not None:
            state_dict_ms[name_ms] = data_ms
    return state_dict_ms


def get_modules(pt_module, ms_module, dtype, *args, **kwargs):
    ms_dtype = pt_dtype = dtype

    pt_path, pt_cls_name = pt_module.rsplit(".", 1)
    ms_path, ms_cls_name = ms_module.rsplit(".", 1)
    pt_module_cls = getattr(importlib.import_module(pt_path), pt_cls_name)
    ms_module_cls = getattr(importlib.import_module(ms_path), ms_cls_name)

    pt_modules_instance = pt_module_cls(*args, **kwargs)
    ms_modules_instance = ms_module_cls(*args, **kwargs)

    missing_keys, unexpected_keys = ms.load_param_into_net(
        ms_modules_instance, convert_state_dict(ms_modules_instance, pt_modules_instance.state_dict()), strict_load=True
    )
    if missing_keys or unexpected_keys:
        logger.warning(
            f"When load state_dict of '{pt_module}' to encounterpart mindspore model:\n"
            f"Missing keys: {missing_keys}\n"
            f"Unexpected keys: {unexpected_keys}\n"
        )

    if dtype == "fp16":
        pt_modules_instance = pt_modules_instance.to(torch.float16)
        ms_modules_instance = set_dtype(ms_modules_instance, ms.float16)
    elif dtype == "bf16":
        pt_modules_instance = pt_modules_instance.to(torch.float32)
        ms_modules_instance = set_dtype(ms_modules_instance, ms.bfloat16)
        pt_dtype = "fp32"
    elif dtype == "fp32":
        pt_modules_instance = pt_modules_instance.to(torch.float32)
        ms_modules_instance = set_dtype(ms_modules_instance, ms.float32)
    else:
        raise NotImplementedError(f"Dtype {dtype} for model is not implemented")

    pt_modules_instance.eval()
    ms_modules_instance.set_train(False)

    # Dealing some issues are not supported by certain types on MindSpore
    _ms_mixed_precision(ms_modules_instance, ms_dtype)

    if pt_dtype == "fp32":
        return pt_modules_instance, ms_modules_instance, pt_dtype, ms_dtype

    # Some torch modules do not support fp16 in CPU, converted to fp32 instead.
    for _, submodule in pt_modules_instance.named_modules():
        if submodule.__class__.__name__ in TORCH_FP16_BLACKLIST:
            logger.warning(
                f"Model '{pt_module}' has submodule {submodule.__class__.__name__} which doens't support fp16, converted to fp32 instead."
            )
            pt_modules_instance = pt_modules_instance.to(torch.float32)
            pt_dtype = "fp32"
            break

    return pt_modules_instance, ms_modules_instance, pt_dtype, ms_dtype


def set_dtype(model, dtype):
    for p in model.get_parameters():
        p = p.set_dtype(dtype)
    return model


def generalized_parse_args(pt_dtype, ms_dtype, *args, **kwargs):
    # parse args
    pt_inputs_args = tuple()
    ms_inputs_args = tuple()
    for x in args:
        if isinstance(x, np.ndarray):
            if x.dtype in (np.float16, np.float32, np.float64, bfloat16):
                px = x.astype(NP_DTYPE_MAPPING[pt_dtype])
                mx = x.astype(NP_DTYPE_MAPPING[ms_dtype])
            else:
                px = mx = x

            pt_inputs_args += (
                (torch.from_numpy(px.astype(np.float32)).to(torch.bfloat16),)
                if pt_dtype == "bf16"
                else (torch.from_numpy(px),)
            )
            ms_inputs_args += (ms.Tensor.from_numpy(mx),)
        else:
            pt_inputs_args += (x,)
            ms_inputs_args += (x,)

    # parse kwargs
    pt_inputs_kwargs = dict()
    ms_inputs_kwargs = dict()
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            if v.dtype in (np.float16, np.float32, np.float64, bfloat16):
                px = v.astype(NP_DTYPE_MAPPING[pt_dtype])
                mx = v.astype(NP_DTYPE_MAPPING[ms_dtype])
            else:
                px = mx = v

            pt_inputs_kwargs[k] = (
                torch.from_numpy(px.astype(np.float32)).to(torch.bfloat16)
                if pt_dtype == "bf16"
                else torch.from_numpy(px)
            )
            ms_inputs_kwargs[k] = ms.Tensor.from_numpy(mx)
        else:
            pt_inputs_kwargs[k] = v
            ms_inputs_kwargs[k] = v

    return pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs


def compute_diffs(pt_outputs: torch.Tensor, ms_outputs: ms.Tensor):
    if isinstance(pt_outputs, BaseOutput):
        pt_outputs = tuple(pt_outputs.values())
    elif not isinstance(pt_outputs, (tuple, list)):
        pt_outputs = (pt_outputs,)
    if not isinstance(ms_outputs, (tuple, list)):
        ms_outputs = (ms_outputs,)

    diffs = []
    for p, m in zip(pt_outputs, ms_outputs):
        if isinstance(p, BaseOutput):
            p = tuple(p.values())[0]

        p = p.detach().cpu().numpy()
        m = m.asnumpy()

        # relative error defined by Frobenius norm
        # dist(x, y) := ||x - y|| / ||y||, where ||·|| means Frobenius norm
        d = np.linalg.norm(p - m) / np.linalg.norm(p)

        diffs.append(d)

    return diffs
