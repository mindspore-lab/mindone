from functools import partial
from typing import Optional, Tuple

import numpy as np
import scipy

from .image_util import get_tv_resample_method, resize_max_res


def inter_distances(tensors: np.ndarray):
    """
    To calculate the distance between each two depth maps.
    """
    distances = []
    for i in range(tensors.shape[0]):
        for j in range(i + 1, tensors.shape[0]):
            arr1 = tensors[i : i + 1]
            arr2 = tensors[j : j + 1]
            distances.append(arr1 - arr2)
    dist = np.concatenate(distances, axis=0)
    return dist


def ensemble_depth(
    depth: np.ndarray,
    scale_invariant: bool = True,
    shift_invariant: bool = True,
    output_uncertainty: bool = False,
    reduction: str = "median",
    regularizer_strength: float = 0.02,
    max_iter: int = 2,
    tol: float = 1e-3,
    max_res: int = 1024,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if depth.ndim != 4 or depth.shape[1] != 1:
        raise ValueError(f"Expecting 4D array of shape [B,1,H,W]; got {depth.shape}.")
    if reduction not in ("mean", "median"):
        raise ValueError(f"Unrecognized reduction method: {reduction}.")
    if not scale_invariant and shift_invariant:
        raise ValueError("Pure shift-invariant ensembling is not supported.")

    def init_param(depth: np.ndarray):
        ensemble_size = depth.shape[0]
        init_min = depth.reshape(ensemble_size, -1).min(axis=1)
        init_max = depth.reshape(ensemble_size, -1).max(axis=1)

        if scale_invariant and shift_invariant:
            init_s = 1.0 / np.clip(init_max - init_min, a_min=1e-6, a_max=None)
            init_t = -init_s * init_min
            param = np.concatenate((init_s, init_t))
        elif scale_invariant:
            init_s = 1.0 / np.clip(init_max, a_min=1e-6, a_max=None)
            param = init_s
        else:
            raise ValueError("Unrecognized alignment.")

        return param

    def align(
        depth: np.ndarray, param: np.ndarray, scale_invariant: bool, shift_invariant: bool, ensemble_size: int
    ) -> np.ndarray:
        if scale_invariant and shift_invariant:
            s, t = np.split(param, 2)
            s = s.reshape(ensemble_size, 1, 1, 1)
            t = t.reshape(ensemble_size, 1, 1, 1)
            out = depth * s + t
        elif scale_invariant:
            s = param.reshape(ensemble_size, 1, 1, 1)
            out = depth * s
        else:
            raise ValueError("Unrecognized alignment.")

        return out

    def ensemble(
        depth_aligned: np.ndarray, reduction: str, return_uncertainty: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        uncertainty = None
        if reduction == "mean":
            prediction = np.mean(depth_aligned, axis=0, keepdims=True)
            if return_uncertainty:
                uncertainty = np.std(depth_aligned, axis=0, keepdims=True)
        elif reduction == "median":
            prediction = np.median(depth_aligned, axis=0, keepdims=True)
            if return_uncertainty:
                uncertainty = np.median(np.abs(depth_aligned - prediction), axis=0, keepdims=True)
        else:
            raise ValueError(f"Unrecognized reduction method: {reduction}.")

        return prediction, uncertainty

    def cost_fn(
        param: np.ndarray,
        depth: np.ndarray,
        scale_invariant: bool,
        shift_invariant: bool,
        ensemble_size: int,
        regularizer_strength: float,
        reduction: str,
    ) -> float:
        cost = 0.0
        depth_aligned = align(depth, param, scale_invariant, shift_invariant, ensemble_size)

        for i in range(ensemble_size):
            for j in range(i + 1, ensemble_size):
                diff = depth_aligned[i] - depth_aligned[j]
                cost += np.sqrt(np.mean(diff**2))

        if regularizer_strength > 0:
            prediction, _ = ensemble(depth_aligned, reduction, return_uncertainty=False)
            err_near = np.abs(0.0 - prediction.min())
            err_far = np.abs(1.0 - prediction.max())
            cost += (err_near + err_far) * regularizer_strength

        return cost

    def compute_param(
        depth: np.ndarray,
        scale_invariant: bool,
        shift_invariant: bool,
        ensemble_size: int,
        regularizer_strength: float,
        reduction: str,
        max_res: int,
        tol: float,
        max_iter: int,
    ) -> np.ndarray:
        depth_to_align = depth.astype(np.float32)
        if max_res is not None and max(depth_to_align.shape[2:]) > max_res:
            depth_to_align = resize_max_res(
                depth_to_align.transpose(0, 2, 3, 1), max_res, get_tv_resample_method("nearest-exact")
            )
            depth_to_align = depth_to_align.transpose(0, 3, 1, 2)

        param = init_param(depth_to_align)

        res = scipy.optimize.minimize(
            partial(
                cost_fn,
                depth=depth_to_align,
                scale_invariant=scale_invariant,
                shift_invariant=shift_invariant,
                ensemble_size=ensemble_size,
                regularizer_strength=regularizer_strength,
                reduction=reduction,
            ),
            param,
            method="BFGS",
            tol=tol,
            options={"maxiter": max_iter, "disp": False},
        )

        return res.x

    requires_aligning = scale_invariant or shift_invariant
    ensemble_size = depth.shape[0]

    if requires_aligning:
        param = compute_param(
            depth,
            scale_invariant,
            shift_invariant,
            ensemble_size,
            regularizer_strength,
            reduction,
            max_res,
            tol,
            max_iter,
        )
        depth = align(depth, param, scale_invariant, shift_invariant, ensemble_size)

    depth, uncertainty = ensemble(depth, reduction, return_uncertainty=output_uncertainty)

    depth_max = depth.max()
    if scale_invariant and shift_invariant:
        depth_min = depth.min()
    elif scale_invariant:
        depth_min = 0
    else:
        raise ValueError("Unrecognized alignment.")
    depth_range = max(depth_max - depth_min, 1e-6)
    depth = (depth - depth_min) / depth_range
    if output_uncertainty:
        uncertainty /= depth_range

    return depth, uncertainty
