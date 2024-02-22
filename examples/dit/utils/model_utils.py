import logging

import mindspore as ms

logger = logging.getLogger()


def load_dit_ckpt_params(model, ckpt_fp):
    logger.info(f"Loading {ckpt_fp} params into DiT model...")
    param_dict = ms.load_checkpoint(ckpt_fp)
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict)
    assert (
        len(param_not_load) == len(ckpt_not_load) == 0
    ), "Exist ckpt params not loaded: {} (total: {})\nor net params not loaded: {} (total: {})".format(
        ckpt_not_load, len(ckpt_not_load), param_not_load, len(param_not_load)
    )
    return model
