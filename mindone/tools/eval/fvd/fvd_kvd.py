import os
import random

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import polynomial_kernel

import mindspore as ms
import mindspore.dataset as ds
from mindspore import ops
from mindspore.dataset import transforms, vision

from .inceptioni3d import inceptioni_3d_fvd

VIDEO_EXTENSIONS = {".mp4", ".gif"}


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        # fast resize
        while min(img.size) >= 2 * self.size:
            img = img.resize((img.width // 2, img.height // 2), resample=Image.BOX)
        scale = self.size / min(img.size)
        img = img.resize((round(scale * img.width), round(scale * img.height)), resample=Image.BICUBIC)

        # center crop
        x1 = (img.width - self.size) // 2
        y1 = (img.height - self.size) // 2
        img = img.crop((x1, y1, x1 + self.size, y1 + self.size))
        return img


class VideoPathDataset:
    """Video files dataload."""

    def __init__(self, files, video_length=16, resolution=224):
        self.files = files
        self.video_length = video_length
        self.resolution = resolution
        self.transform = transforms.Compose(
            [
                CenterCrop(size=self.resolution),
                vision.ToTensor(),
                vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        video_path = self.files[i]
        frames = self.get_video_data(video_path)
        return frames

    def get_video_data(self, video_path):
        cap = cv2.VideoCapture(video_path)
        video_frames = []
        index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                video_frames.append(frame)
                index += 1
            else:
                break
        cap.release()
        rand_idx = random.randint(0, len(video_frames) - self.video_length)
        video_frames = video_frames[rand_idx : rand_idx + self.video_length]
        # note video_frames are in BGR mode, need to trans to RGB mode
        video_frames = [Image.fromarray(video_frame[:, :, ::-1]) for video_frame in video_frames]
        video_frames = np.stack([self.transform(frame)[0] for frame in video_frames], axis=1)

        return video_frames


def get_video_paths(video_dir):
    datalist = []
    for root, _, files in os.walk(video_dir):
        for file in files:
            if os.path.splitext(file)[1] in VIDEO_EXTENSIONS:
                datalist.append(os.path.join(root, file))

    return datalist


def _symmetric_matrix_square_root(mat, eps=1e-10):
    if isinstance(mat, ms.Tensor):
        mat = mat.asnumpy()
    u, s, v = np.linalg.svd(mat, full_matrices=False)
    if not isinstance(s, ms.Tensor):
        s = ms.Tensor(s)
        u = ms.Tensor(u)
        v = ms.Tensor(v)
    v = ops.t(v)
    si = ops.where(s < eps, s, ops.sqrt(s))
    return ops.matmul(ops.matmul(u, ops.diag(si)), ops.t(v))


def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_a_sigmav_a = ops.matmul(sqrt_sigma, ops.matmul(sigma_v, sqrt_sigma))
    return ops.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = ops.mean(x1, axis=0), ops.mean(x2, axis=0)
    if x1.shape[0] != 1:
        x1 = x1.t()
    if x2.shape[0] != 1:
        x2 = x2.t()
    sigma, sigma_w = ops.cov(x1), ops.cov(x2)
    sqrt_trace_component = trace_sqrt_product(sigma, sigma_w)
    trace = ops.trace(sigma + sigma_w) - 2.0 * sqrt_trace_component
    mean = ops.sum((m - m_w) ** 2)
    fd = trace + mean
    return fd


def polynomial_mmd(X, Y):
    m = X.shape[0]
    n = Y.shape[0]
    # compute kernels
    K_XX = polynomial_kernel(X)
    K_YY = polynomial_kernel(Y)
    K_XY = polynomial_kernel(X, Y)
    # compute mmd distance
    K_XX_sum = (K_XX.sum() - np.diagonal(K_XX).sum()) / (m * (m - 1))
    K_YY_sum = (K_YY.sum() - np.diagonal(K_YY).sum()) / (n * (n - 1))
    K_XY_sum = K_XY.sum() / (m * n)
    mmd = K_XX_sum + K_YY_sum - 2 * K_XY_sum
    return mmd


def calucation(model, datalist):
    video = VideoPathDataset(datalist)
    dataloader = ds.GeneratorDataset(
        video,
        ["video_frames"],
        shuffle=False,
        num_parallel_workers=4,
        python_multiprocessing=True,
        max_rowsize=64,
    )
    dataloader = dataloader.batch(1, drop_remainder=False)
    pred_arr = []
    for batch in dataloader.create_dict_iterator():
        pred = model(batch["video_frames"])
        pred_arr.append(pred)

    pred_arr = ops.cat(pred_arr, 0)
    return pred_arr


class Distance:
    def __init__(self, ckpt_path=None):
        # TODO: set context
        if ckpt_path is not None:
            self.model = inceptioni_3d_fvd(pretrained=False, ckpt_path=ckpt_path)
        else:
            self.model = inceptioni_3d_fvd(pretrained=True)

        self.model.set_train(False)

    def comput_mode_feature(self, gen_video_list, gt_video_list):
        gen_feature = calucation(self.model, gen_video_list)
        gt_feature = calucation(self.model, gt_video_list)

        return gen_feature, gt_feature

    def comput_fvd(self, gen_feature, gt_feature):
        fvd = frechet_distance(gen_feature, gt_feature)
        return fvd

    def comput_kvd(self, gen_feature, gt_feature):
        kvd = polynomial_mmd(gen_feature, gt_feature)
        return kvd
