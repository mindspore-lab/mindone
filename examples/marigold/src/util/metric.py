import numpy as np
import pandas as pd


# A class to track all metrics
class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def abs_relative_difference(output, target, valid_mask=None):
    actual_output = np.array(output)
    actual_target = np.array(target)
    abs_relative_diff = np.abs(actual_output - actual_target) / actual_target
    if valid_mask is not None:
        abs_relative_diff = np.where(valid_mask, abs_relative_diff, 0)
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    abs_relative_diff = np.sum(abs_relative_diff, axis=(-1, -2)) / n
    return np.mean(abs_relative_diff)


def log10(output, target, valid_mask=None):
    actual_output = np.array(output)
    actual_target = np.array(target)
    log_output = np.log10(actual_output)
    log_target = np.log10(actual_target)
    diff = np.abs(log_output - log_target)
    if valid_mask is not None:
        diff = np.where(valid_mask, diff, 0)
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    return np.sum(diff, axis=(-1, -2)) / n


def squared_relative_difference(output, target, valid_mask=None):
    actual_output = np.array(output)
    actual_target = np.array(target)
    square_relative_diff = (np.abs(actual_output - actual_target) ** 2) / actual_target
    if valid_mask is not None:
        square_relative_diff = np.where(valid_mask, square_relative_diff, 0)
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    square_relative_diff = np.sum(square_relative_diff, axis=(-1, -2)) / n
    return np.mean(square_relative_diff)


def rmse_linear(output, target, valid_mask=None):
    actual_output = np.array(output)
    actual_target = np.array(target)
    diff = actual_output - actual_target
    if valid_mask is not None:
        diff = np.where(valid_mask, diff, 0)
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = diff**2
    mse = np.sum(diff2, axis=(-1, -2)) / n
    rmse = np.sqrt(mse)
    return np.mean(rmse)


def rmse_log(output, target, valid_mask=None):
    log_output = np.log(np.array(output))
    log_target = np.log(np.array(target))
    diff = log_output - log_target
    if valid_mask is not None:
        diff = np.where(valid_mask, diff, 0)
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = diff**2
    mse = np.sum(diff2, axis=(-1, -2)) / n
    rmse = np.sqrt(mse)
    return np.mean(rmse)


def threshold_percentage(output, target, threshold_val, valid_mask=None):
    d1 = np.array(output) / np.array(target)
    d2 = np.array(target) / np.array(output)
    max_d1_d2 = np.maximum(d1, d2)
    bit_mat = np.where(max_d1_d2 < threshold_val, 1, 0)
    if valid_mask is not None:
        bit_mat = np.where(valid_mask, bit_mat, 0)
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    count_mat = np.sum(bit_mat, axis=(-1, -2))
    threshold_mat = count_mat / n
    return np.mean(threshold_mat)


def delta1_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25, valid_mask)


def delta2_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**2, valid_mask)


def delta3_acc(pred, gt, valid_mask):
    return threshold_percentage(pred, gt, 1.25**3, valid_mask)


def i_rmse(output, target, valid_mask=None):
    output_inv = 1.0 / np.array(output)
    target_inv = 1.0 / np.array(target)
    diff = output_inv - target_inv
    if valid_mask is not None:
        diff = np.where(valid_mask, diff, 0)
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = output.shape[-1] * output.shape[-2]
    diff2 = diff**2
    mse = np.sum(diff2, axis=(-1, -2)) / n
    rmse = np.sqrt(mse)
    return np.mean(rmse)


def silog_rmse(depth_pred, depth_gt, valid_mask=None):
    log_depth_pred = np.log(np.array(depth_pred))
    log_depth_gt = np.log(np.array(depth_gt))
    diff = log_depth_pred - log_depth_gt
    if valid_mask is not None:
        diff = np.where(valid_mask, diff, 0)
        n = np.sum(valid_mask, axis=(-1, -2))
    else:
        n = depth_gt.shape[-2] * depth_gt.shape[-1]

    diff2 = diff**2

    first_term = np.sum(diff2, axis=(-1, -2)) / n
    second_term = (np.sum(diff, axis=(-1, -2)) ** 2) / (n**2)
    loss = np.sqrt(np.mean(first_term - second_term)) * 100
    return loss
