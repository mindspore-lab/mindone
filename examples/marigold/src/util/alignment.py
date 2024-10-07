import numpy as np
from PIL import Image


def align_depth_least_square(
    gt_arr: np.ndarray,
    pred_arr: np.ndarray,
    valid_mask_arr: np.ndarray,
    return_scale_shift=True,
    max_resolution=None,
):
    ori_shape = pred_arr.shape  # input shape

    gt = gt_arr.squeeze()  # [H, W]
    pred = pred_arr.squeeze()
    valid_mask = valid_mask_arr.squeeze()

    # Downsample
    if max_resolution is not None:
        scale_factor = np.min(max_resolution / np.array(ori_shape[-2:]))
        if scale_factor < 1:
            gt = np.array(
                Image.fromarray(gt).resize(
                    (int(gt.shape[1] * scale_factor), int(gt.shape[0] * scale_factor)), Image.NEAREST
                )
            )
            pred = np.array(
                Image.fromarray(pred).resize(
                    (int(pred.shape[1] * scale_factor), int(pred.shape[0] * scale_factor)), Image.NEAREST
                )
            )
            valid_mask = np.array(
                Image.fromarray(valid_mask.astype(np.uint8)).resize(
                    (int(valid_mask.shape[1] * scale_factor), int(valid_mask.shape[0] * scale_factor)), Image.NEAREST
                )
            ).astype(bool)

    assert gt.shape == pred.shape == valid_mask.shape, f"{gt.shape}, {pred.shape}, {valid_mask.shape}"

    gt_masked = gt[valid_mask].reshape((-1, 1))
    pred_masked = pred[valid_mask].reshape((-1, 1))

    gt_masked = gt_masked.astype(np.float32)
    pred_masked = pred_masked.astype(np.float32)

    # numpy solver
    _ones = np.ones_like(pred_masked)
    A = np.concatenate([pred_masked, _ones], axis=-1)
    X = np.linalg.lstsq(A, gt_masked, rcond=None)[0]
    scale, shift = X

    aligned_pred = pred_arr * scale + shift

    # restore dimensions
    aligned_pred = aligned_pred.reshape(ori_shape)

    if return_scale_shift:
        return aligned_pred, scale, shift
    else:
        return aligned_pred


def depth2disparity(depth, return_mask=False):
    disparity = np.zeros_like(depth)
    non_negative_mask = depth > 0
    disparity[non_negative_mask] = 1.0 / depth[non_negative_mask]
    if return_mask:
        return disparity, non_negative_mask
    else:
        return disparity


def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)
