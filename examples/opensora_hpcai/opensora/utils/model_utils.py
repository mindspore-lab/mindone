import argparse
import glob
import logging
import os
from typing import Dict, Optional, Tuple

from huggingface_hub import snapshot_download
from safetensors import safe_open

from mindspore import Model as MSModel
from mindspore import Parameter, context, load_checkpoint
from mindspore.nn import GELU, GraphCell, GroupNorm, SiLU
from mindspore.train.callback import _CallbackManager

from ..models.layers.blocks import Attention, LayerNorm, LlamaRMSNorm, PositionEmbedding2D, SinusoidalEmbedding
from ..models.text_encoder.flan_t5_large.t5 import T5LayerNorm

# SORA's whitelist (FP32) operators
WHITELIST_OPS = [
    LayerNorm,
    Attention,
    LlamaRMSNorm,
    SiLU,
    GELU,
    GroupNorm,
    PositionEmbedding2D,
    SinusoidalEmbedding,
    T5LayerNorm,
]

logger = logging.getLogger(__name__)


def remove_pname_prefix(param_dict, prefix="network."):
    # replace the prefix of param dict
    new_param_dict = {}
    for pname in param_dict:
        if pname.startswith(prefix):
            new_pname = pname[len(prefix) :]
        else:
            new_pname = pname
        new_param_dict[new_pname] = param_dict[pname]
    return new_param_dict


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def count_params(model, verbose=False):
    total_params = sum([param.size for param in model.get_parameters()])
    trainable_params = sum([param.size for param in model.get_parameters() if param.requires_grad])

    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params, trainable_params


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def _load_hf_state_dict(
    ckpt_path: str, name_map: Optional[Dict[str, str]] = None, param_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
) -> Dict[str, Parameter]:
    state_dict = {}
    name_map = name_map or {}
    param_shapes = param_shapes or {}
    with safe_open(ckpt_path, framework="numpy") as f:
        for k in f.keys():
            name = name_map.get(k, k)
            tensor = f.get_tensor(k)
            if param_shapes:
                tensor = tensor.reshape(param_shapes[name])
            state_dict[name] = Parameter(tensor, name=name)
    return state_dict


def load_state_dict(
    ckpt_path: str, name_map: Optional[Dict[str, str]] = None, param_shapes: Optional[Dict[str, Tuple[int, ...]]] = None
) -> Tuple[Dict[str, Parameter], str]:
    ext = os.path.splitext(ckpt_path)[-1]
    if ext == ".ckpt":  # MindSpore
        sd = load_checkpoint(ckpt_path)
    elif ext == ".safetensors":  # safetensors
        sd = _load_hf_state_dict(ckpt_path, name_map, param_shapes)
    elif not os.path.exists(ckpt_path):  # HuggingFace hub
        # FIXME: scripts convert all paths to absolute paths for modelarts. Extract the original repo_id:
        ckpt_path = "/".join(ckpt_path.split(os.sep)[-2:])
        ckpt_path = snapshot_download(ckpt_path, allow_patterns=["*.safetensors"], endpoint="https://hf-mirror.com")
        ckpt_path = glob.glob(ckpt_path + "/*.safetensors")[0]
        sd = _load_hf_state_dict(ckpt_path, name_map, param_shapes)
    else:
        raise ValueError(f"Invalid checkpoint format: {ext}. Please convert to `.ckpt` or `.safetensors` first.")
    return sd, ckpt_path


class Model(MSModel):
    def _eval_in_fit(self, valid_dataset, callbacks=None, dataset_sink_mode=True, cb_params=None):
        # BUG: `_eval_process` has a bug that results in accessing `eval_indexes` even when it is None.
        # This method fixes it by setting `add_eval_loss` to `False`.
        if isinstance(self._eval_network, GraphCell) and dataset_sink_mode:
            raise ValueError("Sink mode is currently not supported when evaluating with a GraphCell.")

        cb_params.eval_network = self._eval_network
        cb_params.valid_dataset = valid_dataset
        cb_params.batch_num = valid_dataset.get_dataset_size()
        cb_params.mode = "eval"
        cb_params.cur_step_num = 0

        self._clear_metrics()

        if context.get_context("device_target") == "CPU" and dataset_sink_mode:
            dataset_sink_mode = False
            logger.info(
                "CPU cannot support dataset sink mode currently."
                "So the evaluating process will be performed with dataset non-sink mode."
            )

        with _CallbackManager(callbacks) as list_callback:
            if dataset_sink_mode:
                return self._eval_dataset_sink_process(valid_dataset, list_callback, cb_params, add_eval_loss=False)
            return self._eval_process(valid_dataset, list_callback, cb_params, add_eval_loss=False)
