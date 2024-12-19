import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.append("../instantmesh/utils/")

import numpy as np
from loss_util import LPIPS
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as calc_psnr
from skimage.metrics import structural_similarity as calc_ssim

from mindspore import Tensor, mint

from mindone.utils.logger import set_logger


def read_img(img, gt_size=256):
    im = Image.open(img)
    width, _ = im.size
    arr = im.resize((gt_size, gt_size)) if width != gt_size else im
    arr = np.array(arr)
    white_bg_arr = arr / 255.0
    white_bg_arr = white_bg_arr[..., :3] * white_bg_arr[..., -1:] + (1 - white_bg_arr[..., -1:])
    out = (white_bg_arr * 255).astype(np.uint8)
    return out


logger = set_logger(name="", output_dir=".")
vanilla_p_meta = "PATH/TO/YOUR/OUTPUT_VANILLA/"
overfitted_p_meta = "PATH/TO/YOUR/OUTPUT_OVERFITTED/"
gt_p_meta = "PATH/TO/YOUR/GT/"

l_pred_overfitted = [overfitted_p_meta + f"sv3d-1000-{i+1}.png" for i in range(5)]
l_pred_overfitted.insert(0, l_pred_overfitted.pop(-1))
l_pred_vanilla = [vanilla_p_meta + f"{i:02d}.png" for i in range(5)]
l_pred_vanilla.pop(-1)
l_pred_vanilla.insert(0, vanilla_p_meta + "20.png")  # sv3d vanilla takes the last one as condition
l_gt = [gt_p_meta + f"{i+8:03d}.png" for i in range(5)]

print(l_pred_overfitted)
print(l_pred_vanilla)
print(l_gt)


def cal_metrics(list_of_pred_path, list_of_gt_path, pred_name):
    assert len(list_of_pred_path) == len(list_of_gt_path)
    mean_psnr = 0
    mean_ssim = 0
    mean_lpips = 0
    num_samples = len(list_of_gt_path)

    arr_gt = np.array([read_img(list_of_gt_path[i]) for i in range(num_samples)])
    arr_pred = np.array([read_img(list_of_pred_path[i], norm=True) for i in range(num_samples)])

    psnr_cur = [calc_psnr(arr_gt[i], arr_pred[i]) for i in range(num_samples)]
    ssim_cur = [
        calc_ssim(arr_gt[i], arr_pred[i], data_range=255, channel_axis=-1, multichannel=True)
        for i in range(num_samples)
    ]
    mean_psnr += sum(psnr_cur)
    mean_ssim += sum(ssim_cur)
    mean_psnr /= num_samples
    mean_ssim /= num_samples

    cal_lpips = LPIPS(pretrained_vgg=False)
    arr_gt = (arr_gt / 255.0).astype(np.float32) * 2.0 - 1.0
    arr_pred = (arr_pred / 255.0).astype(np.float32) * 2.0 - 1.0

    lpips_loss = (
        2 * mint.mean(cal_lpips(Tensor(arr_gt).permute(0, 3, 1, 2), Tensor(arr_pred).permute(0, 3, 1, 2))).asnumpy()
    )
    mean_lpips += lpips_loss.mean()
    mean_lpips /= num_samples
    logger.info(f"{pred_name} mean psnr:{mean_psnr:.4f}")
    logger.info(f"{pred_name} mean ssim:{mean_ssim:.4f}")
    logger.info(f"{pred_name} mean lpips loss: {mean_lpips:.4f}")


cal_metrics(l_pred_vanilla, l_gt, pred_name="vanilla ckpt")
cal_metrics(l_pred_overfitted, l_gt, pred_name="overfitted ckpt")
