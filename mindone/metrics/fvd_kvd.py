import os

import cv2
import numpy as np
from PIL import Image
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
from tqdm import tqdm

import mindspore as ms
from mindspore import ops

from .models.inceptioni3d import inceptioni_3d_fvd


def _symmetric_matrix_square_root(mat, eps=1e-10):
    if isinstance(mat, ms.Tensor):
        mat = mat.asnumpy()
    u, s, v = linalg.svd(mat, full_matrices=False)
    if not isinstance(s, ms.Tensor):
        s = ms.Tensor(s)
        u = ms.Tensor(u)
        v = ms.Tensor(v)
    si = ops.where(s < eps, s, ops.sqrt(s))
    return ops.matmul(ops.matmul(u, ops.diag(si)), v)


def trace_sqrt_product(sigma, sigma_v):
    sqrt_sigma = _symmetric_matrix_square_root(sigma)
    sqrt_sigma = ops.cast(sqrt_sigma, sigma.dtype)
    sqrt_a_sigmav_a = ops.matmul(sqrt_sigma, ops.matmul(sigma_v, sqrt_sigma))
    return ops.trace(_symmetric_matrix_square_root(sqrt_a_sigmav_a))


def cov(m, rowvar=False):
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")
    if m.ndim < 2:
        m = m.view(1, -1)
    if not rowvar and m.shape[0] != 1:
        m = m.t()
    fact = 1.0 / (m.shape[1] - 1)
    m_center = m - ops.mean(m, axis=1, keep_dims=True)
    mt = m_center.t()
    return fact * m_center.matmul(mt).squeeze()


def frechet_distance(x1, x2):
    x1 = x1.flatten(start_dim=1)
    x2 = x2.flatten(start_dim=1)
    m, m_w = ops.mean(x1, axis=0), ops.mean(x2, axis=0)
    sigma, sigma_w = cov(x1, rowvar=False), cov(x2, rowvar=False)
    m = ops.atleast_1d(m)
    m_w = ops.atleast_1d(m_w)
    sigma = ops.atleast_2d(sigma)
    sigma_w = ops.atleast_2d(sigma_w)
    assert m.shape == m_w.shape, "mean vectors have different lengths"
    assert sigma.shape == sigma_w.shape, "covariances have different dimensions"
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


class Frechet_Kernel_Video_Distance:
    def __init__(self, ckpt_path=None, num_frames=32):
        # TODO: set context
        if ckpt_path is not None:
            self.model = inceptioni_3d_fvd(pretrained=False, ckpt_path=ckpt_path)
        else:
            self.model = inceptioni_3d_fvd(pretrained=True)
        self.num_frames = num_frames
        self.model.set_train(False)

    def comput_mode_feature(self, gen_video_list, gt_video_list):
        gen_feature = self.calucation(gen_video_list, self.num_frames)
        gt_feature = self.calucation(gt_video_list, self.num_frames)

        return gen_feature, gt_feature

    def calucation(self, video_list, num_frames):
        pred_arr = []
        for path in tqdm(video_list):
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            cap = cv2.VideoCapture(path)

            frames = []
            index = 0
            while cap.isOpened():
                ret, frame = cap.read()

                if ret:
                    frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frames.append(frame)
                    index += 1
                else:
                    break
            cap.release()
            frames = [i.resize((224, 224)) for i in frames]
            frames = np.stack(frames, axis=0)  # (b*t, h, w, c)
            bt, h, w, c = frames.shape
            if bt > num_frames:
                b = bt // num_frames
                frames = ms.Tensor(frames[: b * num_frames]).view(b, num_frames, h, w, c)  # (b, num_frames, h, w, c)
            else:
                frames = ops.unsqueeze(ms.Tensor(frames), dim=0)  # (1, bt, h, w, c)
            frames = ops.permute(frames, (0, 4, 1, 2, 3))
            frames = 2.0 * frames / 255.0 - 1  # [-1, 1]
            pred = self.model(frames)
            pred_arr.append(pred)
        pred_arr = ops.cat(pred_arr, 0)
        return pred_arr

    def comput_fvd(self, gen_feature, gt_feature):
        fvd = frechet_distance(gen_feature, gt_feature)
        return fvd

    def comput_kvd(self, gen_feature, gt_feature):
        kvd = polynomial_mmd(gen_feature, gt_feature)
        return kvd
