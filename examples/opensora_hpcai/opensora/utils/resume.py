import logging
import os
import mindspore as ms
from mindone.utils.params import load_param_into_net_with_filter

_logger = logging.getLogger(__name__)

def save_train_net(train_net, ckpt_dir, epoch, global_step):
    # train_net: i.e. net_with_grads, contains optimizer, ema, sense_scale, etc.
    ms.save_checkpoint(
        train_net,
        os.path.join(ckpt_dir, "train_resume.ckpt"),
        choice_func=lambda x: not (x.startswith('vae.') or x.startswith('swap.')),
        append_dict={
            "epoch_num": epoch,
            "cur_step": global_step,
            "loss_scale": train_net.scale_sense.asnumpy().item(),
        },
    )


def get_resume_states(resume_ckpt):
    state_dict = ms.load_checkpoint(resume_ckpt)
    start_epoch = int(state_dict.get("epoch_num", ms.Tensor(0, ms.int32)).asnumpy().item())
    
    # self-recorded cur_step and internal global_step seem to be same
    global_step = int(state_dict.get("cur_step", ms.Tensor(0, ms.int32)).asnumpy().item())
    # global_step_internal = int(state_dict.get("global_step", ms.Tensor(0, ms.int32)).asnumpy().item())
    # print('D--: recorded gs: ', global_step, 'internal gs: ', global_step_internal)

    loss_scale = float(state_dict.get("loss_scale", ms.Tensor(0, ms.float32)).asnumpy().item())

    return start_epoch, global_step, loss_scale


def resume_train_net(train_net, resume_ckpt):
    state_dict = ms.load_checkpoint(resume_ckpt)
    global_step = state_dict.get("global_step", ms.Tensor(0, ms.int32))
    _logger.info(f"Resuming train network from {resume_ckpt} of global step {global_step}")

    # network, optimizer, ema will be loaded
    param_not_load, ckpt_not_load = load_param_into_net_with_filter(train_net, state_dict, filter=state_dict.keys())
    # ms.load_param_into_net(train_net, state_dict)

    _logger.info(
        f"Finish resuming. If no parameter fail-load warning displayed, all checkpoint params have been successfully loaded. \n"
    )

    return global_step 


def flush_from_cache(train_net):
    params = train_net.get_parameters()
    for param in params:
        if param.cache_enable:
            ms.Tensor(param).flush_from_cache()
