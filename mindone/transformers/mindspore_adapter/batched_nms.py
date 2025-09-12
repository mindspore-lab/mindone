# This is a custom implementation of NMS (Non-Maximum Suppression) for MindSpore.
# It is adapted from torchvision's NMS implementation to work with MindSpore tensors.
# Ref: https://docs.pytorch.org/vision/main/generated/torchvision.ops.batched_nms.html

import numpy as np

import mindspore as ms
from mindspore import mint, ops


def batched_nms(
    boxes: ms.Tensor,
    scores: ms.Tensor,
    idxs: ms.Tensor,
    iou_threshold: float,
) -> ms.Tensor:
    """
    Performs non-maximum suppression in a batched fashion.

    Each index value correspond to a category, and NMS
    will not be applied between elements of different categories.

    Args:
        boxes (Tensor[N, 4]): boxes where NMS will be performed. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        idxs (Tensor[N]): indices of the categories for each one of the boxes.
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted
        in decreasing order of scores
    """
    # Benchmarks that drove the following thresholds are at
    # https://github.com/pytorch/vision/issues/1311#issuecomment-781329339
    # and https://github.com/pytorch/vision/pull/8925
    # TODO: to determine a reasonable threshold in MindSpore
    if boxes.numel() > 4000:
        return _batched_nms_vanilla(boxes, scores, idxs, iou_threshold)
    else:
        return _batched_nms_coordinate_trick(boxes, scores, idxs, iou_threshold)


def nms(boxes: ms.Tensor, scores: ms.Tensor, iou_threshold: float) -> ms.Tensor:
    """
    Performs non-maximum suppression (NMS) on the boxes according
    to their intersection-over-union (IoU).

    NMS iteratively removes lower scoring boxes which have an
    IoU greater than ``iou_threshold`` with another (higher scoring)
    box.

    If multiple boxes have the exact same score and satisfy the IoU
    criterion with respect to a reference box, the selected box is
    not guaranteed to be the same between CPU and GPU. This is similar
    to the behavior of argsort in PyTorch when repeated values are present.

    Args:
        boxes (Tensor[N, 4])): boxes to perform NMS on. They
            are expected to be in ``(x1, y1, x2, y2)`` format with ``0 <= x1 < x2`` and
            ``0 <= y1 < y2``.
        scores (Tensor[N]): scores for each one of the boxes
        iou_threshold (float): discards all overlapping boxes with IoU > iou_threshold

    Returns:
        Tensor: int32 tensor with the indices of the elements that have been kept by NMS
    """
    box_with_score = np.column_stack((boxes, scores))
    box_with_score_m = ms.Tensor(box_with_score)
    _, output_idx, selected_mask = ops.NMSWithMask(iou_threshold)(box_with_score_m)
    return output_idx[selected_mask]
    # equivalent to torch.ops.torchvision.nms(boxes, scores, iou_threshold)


def _batched_nms_vanilla(
    boxes: ms.Tensor,
    scores: ms.Tensor,
    idxs: ms.Tensor,
    iou_threshold: float,
) -> ms.Tensor:
    # Based on Detectron2 implementation, just manually call nms() on each class independently
    keep_mask = mint.zeros_like(scores, dtype=ms.bool_)
    for class_id in mint.unique(idxs):
        curr_indices = mint.where(idxs == class_id)[0]
        curr_keep_indices = nms(boxes[curr_indices], scores[curr_indices], iou_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = mint.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def _batched_nms_coordinate_trick(
    boxes: ms.Tensor,
    scores: ms.Tensor,
    idxs: ms.Tensor,
    iou_threshold: float,
) -> ms.Tensor:
    # strategy: in order to perform NMS independently per class,
    # we add an offset to all the boxes. The offset is dependent
    # only on the class idx, and is large enough so that boxes
    # from different classes do not overlap
    if boxes.numel() == 0:
        return mint.empty((0,), dtype=ms.int64)
    max_coordinate = boxes.max()
    offsets = idxs.to(boxes.dtype) * (max_coordinate + ms.tensor(1).to(boxes.dtype))
    boxes_for_nms = boxes + offsets[:, None]
    keep = nms(boxes_for_nms, scores, iou_threshold)
    return keep
