import cv2
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .deeplab_v3plus import DeepLabV3Plus

__all__ = ["SegmentDetector"]


class SegmentDetector:
    def __init__(self, ckpt_path, num_classes=21) -> None:
        self.num_classes = num_classes
        self.network = DeepLabV3Plus("eval", num_classes=self.num_classes, output_stride=16, freeze_bn=True)
        self.eval_net = BuildEvalNetwork(self.network)
        param_dict = load_checkpoint(ckpt_path)
        load_param_into_net(self.eval_net, param_dict)
        self.eval_net.set_train(False)

        self.image_mean = [103.53, 116.28, 123.675]
        self.image_std = [57.375, 57.120, 58.395]

    def __call__(self, img):
        batch_img_lst = []
        batch_img_lst.append(img)
        batch_res = self.eval_batch_scales(self.eval_net, batch_img_lst, scales=[1.0], base_crop_size=513, flip=False)
        result = batch_res[0]
        result = self.show_result(img, result, get_palette("voc"), opacity=1)
        return bgr2rgb(result)

    def eval_batch_scales(self, eval_net, img_lst, scales, base_crop_size=513, flip=True):
        """eval_batch_scales"""
        sizes_ = [int((base_crop_size - 1) * sc) + 1 for sc in scales]
        probs_lst = self.eval_batch(eval_net, img_lst, crop_size=sizes_[0], flip=flip)
        print(sizes_)
        for crop_size_ in sizes_[1:]:
            probs_lst_tmp = self.eval_batch(eval_net, img_lst, crop_size=crop_size_, flip=flip)
            for pl, _ in enumerate(probs_lst):
                probs_lst[pl] += probs_lst_tmp[pl]

        result_msk = []
        for i in probs_lst:
            result_msk.append(i.argmax(axis=2))
        return result_msk

    def eval_batch(self, eval_net, img_lst, crop_size=513, flip=True):
        """eval_batch"""
        result_lst = []
        batch_size = len(img_lst)
        batch_img = np.zeros((32, 3, crop_size, crop_size), dtype=np.float32)
        resize_hw = []
        for bs in range(batch_size):
            img_ = img_lst[bs]
            img_, resize_h, resize_w = self.pre_process(img_, crop_size)
            batch_img[bs] = img_
            resize_hw.append([resize_h, resize_w])

        batch_img = np.ascontiguousarray(batch_img)
        net_out = eval_net(Tensor(batch_img, mstype.float32))
        net_out = net_out.asnumpy()

        if flip:
            batch_img = batch_img[:, :, :, ::-1]
            net_out_flip = eval_net(Tensor(batch_img, mstype.float32))
            net_out += net_out_flip.asnumpy()[:, :, :, ::-1]

        for bs in range(batch_size):
            probs_ = net_out[bs][:, : resize_hw[bs][0], : resize_hw[bs][1]].transpose((1, 2, 0))
            ori_h, ori_w = img_lst[bs].shape[0], img_lst[bs].shape[1]
            probs_ = cv2.resize(probs_, (ori_w, ori_h))
            result_lst.append(probs_)

        return result_lst

    def pre_process(self, img_, crop_size=513):
        """pre_process"""
        # resize
        img_ = resize_long(img_, crop_size)
        resize_h, resize_w, _ = img_.shape

        # mean, std
        image_mean = np.array(self.image_mean)
        image_std = np.array(self.image_std)
        img_ = (img_ - image_mean) / image_std

        # pad to crop_size
        pad_h = crop_size - img_.shape[0]
        pad_w = crop_size - img_.shape[1]
        if pad_h > 0 or pad_w > 0:
            img_ = cv2.copyMakeBorder(img_, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

        # hwc to chw
        img_ = img_.transpose((2, 0, 1))
        return img_, resize_h, resize_w

    def show_result(self, img, result, palette=None, opacity=0.5):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor): The semantic segmentation results to draw over
                `img`.
            palette (list[list[int]]] | np.ndarray | None): The palette of
                segmentation map. If None is given, random palette will be
                generated. Default: None
            opacity(float): Opacity of painted segmentation map.
                Default 0.5.
                Must be in (0, 1] range.
        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        # img = mmcv.imread(img)
        img = img.copy()
        seg = result

        palette = np.array(palette)
        assert palette.shape[0] == self.num_classes
        assert palette.shape[1] == 3
        assert len(palette.shape) == 2
        assert 0 < opacity <= 1.0
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        # convert to BGR
        color_seg = color_seg[..., ::-1]

        img = img * (1 - opacity) + color_seg * opacity
        img = img.astype(np.uint8)
        # if out_file specified, do not show image in window

        return img


class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        output = self.network(input_data)
        output = self.softmax(output)
        return output


def resize_long(img, long_size=513):
    h, w, _ = img.shape
    if h > w:
        new_h = long_size
        new_w = int(1.0 * long_size * w / h)
    else:
        new_w = long_size
        new_h = int(1.0 * long_size * h / w)
    imo = cv2.resize(img, (new_w, new_h))
    return imo


def voc_palette():
    """Pascal VOC palette for external use."""
    return [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]


dataset_aliases = {
    "cityscapes": ["cityscapes"],
    "ade": ["ade", "ade20k"],
    "voc": ["voc", "pascal_voc", "voc12", "voc12aug"],
}


def get_palette(dataset):
    """Get class palette (RGB) of a dataset."""
    alias2name = {}
    for name, aliases in dataset_aliases.items():
        for alias in aliases:
            alias2name[alias] = name

    if isinstance(dataset, str):
        if dataset in alias2name:
            labels = eval(alias2name[dataset] + "_palette()")
        else:
            raise ValueError(f"Unrecognized dataset: {dataset}")
    else:
        raise TypeError(f"dataset must a str, but got {type(dataset)}")
    return labels


def convert_color_factory(src, dst):
    code = getattr(cv2, f"COLOR_{src.upper()}2{dst.upper()}")

    def convert_color(img):
        out_img = cv2.cvtColor(img, code)
        return out_img

    convert_color.__doc__ = f"""Convert a {src.upper()} image to {dst.upper()}
        image.

    Args:
        img (ndarray or str): The input image.

    Returns:
        ndarray: The converted {dst.upper()} image.
    """

    return convert_color


bgr2rgb = convert_color_factory("bgr", "rgb")
