import glob
import logging
import os
from pathlib import Path
from typing import Union

import mindspore as ms

from mindone.utils.params import load_param_into_net_with_filter

_logger = logging.getLogger(__name__)


def get_resume_ckpt(resume: Union[bool, str], work_dir_with_date: str = None):
    """
    Args:
        resume: path to resume ckpt, or folder containing resume ckpt, or bool variable indicating to search train_resume.ckpt from work_dir_with_date
        work_dir_with_date: work dir for saving model ckpts and logs, assuming it is appended with a date folder,
            e.g. outputs/opensora1.2_stage2/2024-08-01T01-02-03
    return:
        path of the found resume checkpoint. if None, no resume ckpt is found.
    """
    resume_ckpt = None
    ori_output_path = str(Path(work_dir_with_date).parents[0])
    if isinstance(resume, str):
        if os.path.isfile(resume):
            resume_ckpt = resume
        else:
            _logger.warning(f"{resume} does not exist. Skip loading previous training states.")
    else:
        pattern = os.path.join(ori_output_path, "**/train_resume.ckpt")
        found = glob.glob(pattern, recursive=True)  # find the latest train_resume.ckpt
        if len(found) > 0:
            resume_ckpt = sorted(found)[-1]
        else:
            _logger.warning(f"No train_resume.ckpt found in {ori_output_path}. Skip loading previous training states.")

    return resume_ckpt


def save_train_net(train_net, ckpt_dir, epoch, global_step):
    # train_net: i.e. net_with_grads, contains optimizer, ema, sense_scale, etc.
    ms.save_checkpoint(
        train_net,
        os.path.join(ckpt_dir, "train_resume.ckpt"),
        choice_func=lambda x: not (x.startswith("vae.") or x.startswith("swap.")),
        append_dict={
            "epoch_num": epoch,
            "cur_step": global_step,
            "loss_scale": train_net.scale_sense.asnumpy().item(),
        },
    )


def get_resume_states(resume_ckpt):
    state_dict = ms.load_checkpoint(resume_ckpt)
    start_epoch = int(state_dict.get("epoch_num", ms.Tensor(0, ms.int32)).asnumpy().item())

    # self-recorded cur_step and internal global_step should be the same
    global_step = int(state_dict.get("cur_step", ms.Tensor(0, ms.int32)).asnumpy().item())
    # global_step_internal = int(state_dict.get("global_step", ms.Tensor(0, ms.int32)).asnumpy().item())

    loss_scale = float(state_dict.get("loss_scale", ms.Tensor(0, ms.float32)).asnumpy().item())

    return start_epoch, global_step, loss_scale


def resume_train_net(train_net, resume_ckpt):
    state_dict = ms.load_checkpoint(resume_ckpt)
    global_step = int(state_dict.get("cur_step", ms.Tensor(0, ms.int32)).asnumpy().item())
    _logger.info(f"Resuming train network from {resume_ckpt} of global step {global_step}")

    # network, optimizer, ema will be loaded
    param_not_load, ckpt_not_load = load_param_into_net_with_filter(train_net, state_dict, filter=state_dict.keys())

    _logger.info(
        "Finish resuming. If no parameter fail-load warning displayed, all checkpoint params have been successfully loaded."
    )

    return global_step


def flush_from_cache(train_net):
    params = train_net.get_parameters()
    for param in params:
        if param.cache_enable:
            ms.Tensor(param).flush_from_cache()
