import logging
import os
import re
from typing import Union

from mindcv.utils.download import DownLoad

import mindspore as ms
from mindspore import nn

from mindone.utils.params import load_param_into_net_with_filter

logger = logging.getLogger()


def is_url(string):
    # Regex to check for URL patterns
    url_pattern = re.compile(r"^(http|https|ftp)://")
    return bool(url_pattern.match(string))


def load_from_pretrained(
    net: nn.Cell,
    checkpoint: Union[str, dict],
    ignore_net_params_not_loaded=False,
    ensure_all_ckpt_params_loaded=False,
    cache_dir: str = None,
):
    """load checkpoint into network.

    Args:
        net: network
        checkpoint: local file path to checkpoint, or url to download checkpoint, or a dict for network parameters
        ignore_net_params_not_loaded: set True for inference if only a part of network needs to be loaded, the flushing net-not-loaded warnings will disappear.
        ensure_all_ckpt_params_loaded : set True for inference if you want to ensure no checkpoint param is missed in loading
        cache_dir: directory to cache the downloaded checkpoint, only effective when `checkpoint` is a url.
    """
    if isinstance(checkpoint, str):
        if is_url(checkpoint):
            url = checkpoint
            cache_dir = os.path.join(os.path.expanduser("~"), ".mindspore/models") if cache_dir is None else cache_dir
            os.makedirs(cache_dir, exist_ok=True)
            DownLoad().download_url(url, path=cache_dir)
            checkpoint = os.path.join(cache_dir, os.path.basename(url))
        if os.path.exists(checkpoint):
            param_dict = ms.load_checkpoint(checkpoint)
        else:
            raise FileNotFoundError(f"{checkpoint} doesn't exist")
    elif isinstance(checkpoint, dict):
        param_dict = checkpoint
    else:
        raise TypeError(f"unknown checkpoint type: {checkpoint}")

    if param_dict:
        if ignore_net_params_not_loaded:
            filter = param_dict.keys()
        else:
            filter = None
        param_not_load, ckpt_not_load = load_param_into_net_with_filter(net, param_dict, filter=filter)

        if ensure_all_ckpt_params_loaded:
            assert (
                len(ckpt_not_load) == 0
            ), f"All params in checkpoint must be loaded. but got these not loaded {ckpt_not_load}"

        if not ignore_net_params_not_loaded:
            if len(param_not_load) > 0:
                logger.info("Net params not loaded: {}".format([p for p in param_not_load if not p.startswith("adam")]))
        logger.info("Checkpoint params not loaded: {}".format([p for p in ckpt_not_load if not p.startswith("adam")]))
