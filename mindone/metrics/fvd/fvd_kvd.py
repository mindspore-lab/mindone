import cv2
import numpy as np
from scipy import linalg
from sklearn.metrics.pairwise import polynomial_kernel
from tqdm import tqdm

import mindspore as ms
from mindspore import ops

from ..video_data import TextVideoDataset
from .inceptioni3d import inceptioni_3d_fvd


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
    def __init__(self, ckpt_path=None, sample_n_frames=64, sample_stride=1):
        # TODO: set context
        if ckpt_path is not None:
            self.model = inceptioni_3d_fvd(pretrained=False, ckpt_path=ckpt_path)
        else:
            self.model = inceptioni_3d_fvd(pretrained=True)
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride
        self.model.set_train(False)

    def comput_mode_feature(self, gen_video_folder, gt_video_folder, gen_csv_path=None, gt_csv_path=None):
        gen_feature = self.calucation(gen_video_folder, csv_path=gen_csv_path, sample_n_frames=self.sample_n_frames)
        gt_feature = self.calucation(
            gt_video_folder,
            csv_path=gt_csv_path,
            sample_stride=self.sample_stride,
            sample_n_frames=self.sample_n_frames,
        )

        return gen_feature, gt_feature

    def calucation(self, video_folder, csv_path=None, sample_stride=1, sample_n_frames=16):
        pred_arr = []
        dataset = TextVideoDataset(
            video_folder, csv_path=csv_path, sample_stride=sample_stride, sample_n_frames=sample_n_frames
        )
        for video_index in tqdm(range(len(dataset)), total=len(dataset)):
            pixel_values, _ = dataset.get_video_frame(video_index)  # (f c h w)
            pixel_values = ((pixel_values + 1) * 127.5).astype(np.uint8)  # (f c h w)
            pixel_values = np.transpose(pixel_values, (0, 2, 3, 1))  # (f h w c)
            pixel_values = [cv2.resize(pixel_values, (224, 224)) for pixel_value in pixel_values]
            pixel_values = np.stack(pixel_values, axis=0)  # (f h w c)
            pixel_values = (pixel_values / 127.5 - 1.0).astype(np.float32)  # (f h w c)
            pixel_values = ops.unsqueeze(ms.Tensor(pixel_values), dim=0)  # (1 f h w c)
            pixel_values = ops.permute(pixel_values, (0, 4, 1, 2, 3))  # (1 c f h w)
            pred = self.model(pixel_values)
            pred_arr.append(pred)

        pred_arr = ops.cat(pred_arr, 0).float()

        return pred_arr

    def comput_fvd(self, gen_feature, gt_feature):
        fvd = frechet_distance(gen_feature, gt_feature)
        fvd = fvd / gen_feature.shape[0]
        return fvd

    def comput_kvd(self, gen_feature, gt_feature):
        kvd = polynomial_mmd(gen_feature, gt_feature)
        kvd = kvd / gen_feature.shape[0]
        return kvd
