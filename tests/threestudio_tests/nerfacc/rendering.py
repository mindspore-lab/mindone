import numpy as np

import mindspore as ms
from mindspore import Tensor, mint


def test_render_visibility():
    from mindone.models.nerfacc.volrend import render_visibility_from_alpha

    ray_indices = Tensor([0, 2, 2, 2, 2], dtype=ms.int32)  # (all_samples,)
    alphas = Tensor([0.4, 0.3, 0.8, 0.8, 0.5], dtype=ms.float32)  # (all_samples,)

    # transmittance: [1.0, 1.0, 0.7, 0.14, 0.028]
    vis = render_visibility_from_alpha(alphas, ray_indices=ray_indices, early_stop_eps=0.03, alpha_thre=0.0)
    vis_tgt = Tensor([True, True, True, True, False], dtype=ms.bool)
    assert np.allclose(vis, vis_tgt)

    # transmittance: [1.0, 1.0, 1.0, 0.2, 0.04]
    vis = render_visibility_from_alpha(alphas, ray_indices=ray_indices, early_stop_eps=0.05, alpha_thre=0.35)
    vis_tgt = Tensor([True, False, True, True, False], dtype=ms.bool)
    assert np.allclose(vis, vis_tgt)


def test_render_weight_from_alpha():
    from mindone.models.nerfacc.volrend import render_weight_from_alpha

    ray_indices = Tensor([0, 2, 2, 2, 2], dtype=ms.int32)  # (all_samples,)
    alphas = Tensor([0.4, 0.3, 0.8, 0.8, 0.5], dtype=ms.float32)  # (all_samples,)

    # transmittance: [1.0, 1.0, 0.7, 0.14, 0.028]
    weights, _ = render_weight_from_alpha(alphas, ray_indices=ray_indices, n_rays=3)
    weights_tgt = Tensor([1.0 * 0.4, 1.0 * 0.3, 0.7 * 0.8, 0.14 * 0.8, 0.028 * 0.5], dtype=ms.float32)
    assert np.allclose(weights, weights_tgt)


def test_render_weight_from_density():
    from mindone.models.nerfacc.volrend import render_weight_from_alpha, render_weight_from_density

    ray_indices = Tensor([0, 2, 2, 2, 2], dtype=ms.int32)  # (all_samples,)
    sigmas = mint.rand((ray_indices.shape[0],))  # (all_samples,)
    t_starts = ms.rand_like(sigmas)
    t_ends = ms.rand_like(sigmas) + 1.0
    alphas = 1.0 - ms.exp(-sigmas * (t_ends - t_starts))

    weights, _, _ = render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=3)
    weights_tgt, _ = render_weight_from_alpha(alphas, ray_indices=ray_indices, n_rays=3)
    assert np.allclose(weights, weights_tgt)


def test_accumulate_along_rays():
    from mindone.models.nerfacc.volrend import accumulate_along_rays

    ray_indices = Tensor([0, 2, 2, 2, 2], dtype=ms.int32)  # (all_samples,)
    weights = Tensor([0.4, 0.3, 0.8, 0.8, 0.5], dtype=ms.float32)  # (all_samples,)
    values = mint.rand((5, 2))  # (all_samples, 2)

    ray_values = accumulate_along_rays(weights, values=values, ray_indices=ray_indices, n_rays=3)
    assert ray_values.shape == (3, 2)
    assert np.allclose(ray_values[0, :], weights[0, None] * values[0, :])
    assert (ray_values[1, :] == 0).all()
    assert np.allclose(ray_values[2, :], (weights[1:, None] * values[1:]).sum(dim=0))


def test_grads():
    from mindone.models.nerfacc.volrend import (
        render_transmittance_from_density,
        render_weight_from_alpha,
        render_weight_from_density,
    )

    ray_indices = Tensor([0, 2, 2, 2, 2], dtype=ms.int32)  # (all_samples,)
    packed_info = Tensor([[0, 1], [1, 0], [1, 4]], dtype=ms.long)
    sigmas = Tensor([0.4, 0.8, 0.1, 0.8, 0.1])
    sigmas.requires_grad = True
    t_starts = ms.rand_like(sigmas)
    t_ends = t_starts + 1.0

    weights_ref = Tensor([0.3297, 0.5507, 0.0428, 0.2239, 0.0174])
    sigmas_grad_ref = Tensor([0.6703, 0.1653, 0.1653, 0.1653, 0.1653])

    # naive impl. trans from sigma
    trans, _ = render_transmittance_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=3)
    weights = trans * (1.0 - ms.exp(-sigmas * (t_ends - t_starts)))
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert np.allclose(weights_ref, weights, atol=1e-4)
    assert np.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    # naive impl. trans from alpha
    trans, _ = render_transmittance_from_density(t_starts, t_ends, sigmas, packed_info=packed_info, n_rays=3)
    weights = trans * (1.0 - ms.exp(-sigmas * (t_ends - t_starts)))
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert np.allclose(weights_ref, weights, atol=1e-4)
    assert np.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    weights, _, _ = render_weight_from_density(t_starts, t_ends, sigmas, ray_indices=ray_indices, n_rays=3)
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert np.allclose(weights_ref, weights, atol=1e-4)
    assert np.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    weights, _, _ = render_weight_from_density(t_starts, t_ends, sigmas, packed_info=packed_info, n_rays=3)
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert np.allclose(weights_ref, weights, atol=1e-4)
    assert np.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    alphas = 1.0 - ms.exp(-sigmas * (t_ends - t_starts))
    weights, _ = render_weight_from_alpha(alphas, ray_indices=ray_indices, n_rays=3)
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert np.allclose(weights_ref, weights, atol=1e-4)
    assert np.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)

    alphas = 1.0 - ms.exp(-sigmas * (t_ends - t_starts))
    weights, _ = render_weight_from_alpha(alphas, packed_info=packed_info, n_rays=3)
    weights.sum().backward()
    sigmas_grad = sigmas.grad.clone()
    sigmas.grad.zero_()
    assert np.allclose(weights_ref, weights, atol=1e-4)
    assert np.allclose(sigmas_grad_ref, sigmas_grad, atol=1e-4)


def test_rendering():
    from mindone.models.nerfacc.volrend import rendering

    def rgb_sigma_fn(t_starts, t_ends, ray_indices):
        return ms.stack([t_starts] * 3, dim=-1), t_starts

    ray_indices = Tensor([0, 2, 2, 2, 2], dtype=ms.int32)  # (all_samples,)
    sigmas = mint.rand((ray_indices.shape[0],))  # (all_samples,)
    t_starts = ms.rand_like(sigmas)
    t_ends = ms.rand_like(sigmas) + 1.0

    _, _, _, _ = rendering(
        t_starts,
        t_ends,
        ray_indices=ray_indices,
        n_rays=3,
        rgb_sigma_fn=rgb_sigma_fn,
    )


if __name__ == "__main__":
    test_render_visibility()
    test_render_weight_from_alpha()
    test_render_weight_from_density()
    test_accumulate_along_rays()
    test_grads()
    test_rendering()
