import numpy as np
import pytest
from spherical_harmonics import components_from_spherical_harmonics, num_sh_bases

import mindspore as ms
from mindspore import Tensor, mint, ops


@pytest.mark.parametrize("degree", list(range(0, 5)))
def test_spherical_harmonics(degree):
    ms.set_seed(0)
    N = 1000000

    dx = mint.normal(size=(N, 3))
    dx = dx / ops.norm(dx, dim=-1, keepdim=True)
    sh = components_from_spherical_harmonics(degree, dx)
    matrix = (sh.T @ sh) / N * 4 * Tensor(np.pi)
    assert np.allclose(matrix.asnumpy(), mint.eye(num_sh_bases(degree)).asnumpy(), rtol=0, atol=1.5e-2)
