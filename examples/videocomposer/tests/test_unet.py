from vc.config import cfg

cfg.resume_checkpoint = "non_ema_228000.pth"
import numpy as np
from vc.models import UNetSD_temporal
from vc.utils import get_abspath_of_weights

import mindspore as ms

ms.set_seed(100)
ms.set_context(mode=1, pynative_synchronize=True)
black_image_feature = ms.ops.zeros([1, 1024])
zero_y = ms.ops.zeros([1, 77, 1024])
unet_fp16 = UNetSD_temporal(
    cfg=cfg,
    in_dim=cfg.unet_in_dim,
    concat_dim=cfg.unet_concat_dim,
    dim=cfg.unet_dim,
    y_dim=cfg.unet_y_dim,
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
    use_checkpoint=cfg.use_checkpoint,
    use_fps_condition=cfg.use_fps_condition,
    use_sim_mask=cfg.use_sim_mask,
    video_compositions=cfg.video_compositions,
    misc_dropout=cfg.misc_dropout,
    p_all_zero=cfg.p_all_zero,
    p_all_keep=cfg.p_all_zero,
    zero_y=zero_y,
    black_image_feature=black_image_feature,
    use_fp16=True,
)

unet_fp16.set_train(False)
for name, param in unet_fp16.parameters_and_names():
    param.requires_grad = False
unet_fp16.load_state_dict(get_abspath_of_weights(cfg.resume_checkpoint), text_to_video_pretrain=False)
unet_fp32 = UNetSD_temporal(
    cfg=cfg,
    in_dim=cfg.unet_in_dim,
    concat_dim=cfg.unet_concat_dim,
    dim=cfg.unet_dim,
    y_dim=cfg.unet_y_dim,
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
    use_checkpoint=cfg.use_checkpoint,
    use_fps_condition=cfg.use_fps_condition,
    use_sim_mask=cfg.use_sim_mask,
    video_compositions=cfg.video_compositions,
    misc_dropout=cfg.misc_dropout,
    p_all_zero=cfg.p_all_zero,
    p_all_keep=cfg.p_all_zero,
    zero_y=zero_y,
    black_image_feature=black_image_feature,
    use_fp16=False,
)

unet_fp32.set_train(False)
for name, param in unet_fp32.parameters_and_names():
    param.requires_grad = False

unet_fp32.load_state_dict(get_abspath_of_weights(cfg.resume_checkpoint), text_to_video_pretrain=False)
xt = ms.ops.randn((1, 4, 16, 32, 32))
t = ms.ops.ones((1,), dtype=ms.int64) * 900
y = ms.ops.randn((1, 77, 1024), dtype=ms.float16)
single_sketch = ms.ops.randn((1, 1, 16, 384, 384), dtype=ms.float32)

out_fp16 = unet_fp16(xt, t, y=y, single_sketch=single_sketch).asnumpy()
out_fp32 = unet_fp32(xt, t, y=y, single_sketch=single_sketch).asnumpy()
if not np.allclose(out_fp16, out_fp32, atol=1e-2):
    print(f"max abs diff: {np.abs(out_fp16 - out_fp32).max()}")
    print("Outputs are not equal!")
else:
    print(f"max abs diff: {np.abs(out_fp16 - out_fp32).max()}")
    print("Outputs are equal!")
