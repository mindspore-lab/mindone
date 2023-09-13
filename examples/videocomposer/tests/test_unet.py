# import difflib
import sys
import time

import numpy as np
from vc.models import UNetSD_temporal

import mindspore as ms

# from mindspore import ops

# from copy import deepcopy

# from vc.utils.pt2ms import auto_map


sys.path.append("../stable_diffusion_v2/")
import logging

from ldm.modules.train.tools import set_random_seed
from ldm.util import count_params

logger = logging.getLogger(__name__)
from ldm.modules.logger import set_logger

set_logger(name="", output_dir="./tests")


def test_unet():
    from configs.train_base import cfg

    set_random_seed(42)
    ms.set_context(mode=0)

    cfg.video_compositions = ["text"]  # , "depthmap"]
    cfg.temporal_attention = True  # Set to False can reduce half waiting time
    use_fp16 = True
    load = True
    use_droppath_masking = True

    model = UNetSD_temporal(
        cfg=cfg,
        in_dim=cfg.unet_in_dim,
        concat_dim=cfg.unet_concat_dim,
        dim=cfg.unet_dim,
        context_dim=cfg.unet_context_dim,
        out_dim=cfg.unet_out_dim,
        dim_mult=cfg.unet_dim_mult,
        num_heads=cfg.unet_num_heads,
        head_dim=cfg.unet_head_dim,
        num_res_blocks=cfg.unet_res_blocks,
        attn_scales=cfg.unet_attn_scales,
        dropout=cfg.unet_dropout,
        temporal_attention=cfg.temporal_attention,
        temporal_attn_times=cfg.temporal_attn_times,
        use_fps_condition=cfg.use_fps_condition,
        use_sim_mask=cfg.use_sim_mask,
        video_compositions=cfg.video_compositions,
        misc_dropout=cfg.misc_dropout,
        p_all_zero=cfg.p_all_zero,
        p_all_keep=cfg.p_all_zero,
        use_fp16=use_fp16,
        use_adaptive_pool=False,
        use_droppath_masking=use_droppath_masking,
    )

    model = model.set_train(False)  # .to_float(ms.float32)
    for name, param in model.parameters_and_names():
        param.requires_grad = False

    # ms.save_checkpoint(model, "outputs/tmp_unet.ckpt")
    # exit(1)

    if load:
        ckpt_path = "./model_weights/non_ema_228000-3bb2ee9a.ckpt"
        # ckpt_path = "./outputs/tmp_unet.ckpt"
        model.load_state_dict(ckpt_path)
        # param_dict = ms.load_checkpoint(ckpt_path)
        # param_dict = auto_map(model, param_dict)
        # param_not_load, ckpt_not_load = ms.load_param_into_net(model, param_dict, strict_load=True)
        # print("Net params not load: ", param_not_load)
        # print("Ckpt params not used: ", ckpt_not_load)

    num_params = count_params(model)[0]
    print("UNet params: {:,}".format(num_params))

    # prepare inputs
    dtype = ms.float16 if use_fp16 else ms.float32
    batch, c, f, h, w = 1, 4, 2, 128 // 8, 128 // 8
    txt_emb_dim = cfg.unet_context_dim
    seq_len = 77

    time_cost = []
    trials = 3
    for i in range(trials):
        latent_frames = np.ones([batch, c, f, h, w]) / 2.0
        x_t = latent_frames = ms.Tensor(latent_frames, dtype=dtype)
        txt_emb = np.ones([batch, seq_len, txt_emb_dim]) / 2.0
        y = txt_emb = ms.Tensor(txt_emb, dtype=dtype)
        # motion = ms.Tensor(np.random.normal(size=(1, 2, 16, 256, 256)), dtype=ms.float32)
        step = 50
        t = ms.Tensor([step] * batch, dtype=ms.int64)

        s = time.time()
        noise = model(x_t, t, y)
        dur = time.time() - s
        print("infer res: ", noise.max(), noise.min())
        print("time cost: ", dur)
        time_cost.append(dur)

    print("Time cost: ", time_cost)


if __name__ == "__main__":
    test_unet()
