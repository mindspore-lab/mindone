import argparse
import logging

from mindspore import Model as MSModel
from mindspore import context, nn
from mindspore.nn import GroupNorm, SiLU  # GELU
from mindspore.train.callback import _CallbackManager

from ..models.layers.blocks import Attention, LayerNorm, LlamaRMSNorm, PositionEmbedding2D, SinusoidalEmbedding
from ..models.text_encoder.flan_t5_large.t5 import T5LayerNorm

# SORA's whitelist (FP32) operators
WHITELIST_OPS = [
    LayerNorm,
    Attention,
    LlamaRMSNorm,
    SiLU,
    # GELU,
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


class Model(MSModel):
    def _eval_in_fit(self, valid_dataset, callbacks=None, dataset_sink_mode=True, cb_params=None):
        # BUG: `_eval_process` has a bug that results in accessing `eval_indexes` even when it is None.
        # This method fixes it by setting `add_eval_loss` to `False`.
        if isinstance(self._eval_network, nn.GraphCell) and dataset_sink_mode:
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
