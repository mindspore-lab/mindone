import os
import sys
import unittest

import numpy as np
import torch

from mindspore import Tensor

sys.path.insert(0, os.path.abspath("./"))
sys.path.insert(0, os.path.abspath("../../"))
from opensora.models.causalvideovae.model.modules.wavelet import (
    HaarWaveletTransform2D,
    HaarWaveletTransform3D,
    InverseHaarWaveletTransform2D,
    InverseHaarWaveletTransform3D,
)

from tests.torch_wavelet import HaarWaveletTransform2D as HaarWaveletTransform2D_torch
from tests.torch_wavelet import HaarWaveletTransform3D as HaarWaveletTransform3D_torch
from tests.torch_wavelet import InverseHaarWaveletTransform2D as InverseHaarWaveletTransform2D_torch
from tests.torch_wavelet import InverseHaarWaveletTransform3D as InverseHaarWaveletTransform3D_torch

sys.path.append(".")
import mindspore as ms

dtype = ms.float16


class TestWaveletTransforms(unittest.TestCase):
    def setUp(self):
        # Initialize all modules
        self.modules = {
            "HaarWaveletTransform2D": [HaarWaveletTransform2D(), HaarWaveletTransform2D_torch()],
            "HaarWaveletTransform3D": [HaarWaveletTransform3D(), HaarWaveletTransform3D_torch()],
            "InverseHaarWaveletTransform3D": [InverseHaarWaveletTransform3D(), InverseHaarWaveletTransform3D_torch()],
            "InverseHaarWaveletTransform2D": [InverseHaarWaveletTransform2D(), InverseHaarWaveletTransform2D_torch()],
        }

    def generate_input(self, module_name):
        # Define input shapes based on module name
        input_shapes = {
            "HaarWaveletTransform2D": (1, 1, 6, 6),  # Example shape for 2D
            "HaarWaveletTransform3D": (1, 1, 6, 6, 6),  # Example shape for 3D
            "InverseHaarWaveletTransform3D": (1, 8, 6, 6, 6),  # Example shape for 3D
            "InverseHaarWaveletTransform2D": (1, 1, 6, 6),  # Example shape for 2D
        }
        shape = input_shapes[module_name]
        return torch.randn(*shape)

    def test_output_similarity(self):
        for module_name, module in self.modules.items():
            with self.subTest(module=module_name):
                x_torch = self.generate_input(module_name)
                x_mindspore = Tensor(x_torch.numpy())
                module_ms, module_torch = module

                output_torch = module_torch(x_torch)
                output_mindspore = module_ms(x_mindspore.to(dtype))

                abs_diff = np.abs(output_torch.numpy() - output_mindspore.asnumpy())
                print(f"Mean Absolute Difference for {module_name}: {abs_diff.mean()}")
                print(f"Relative Abs Difference for {module_name}: {np.mean(abs_diff/(output_torch.numpy()+1e-6))}")

                self.assertTrue(
                    np.allclose(output_torch.numpy(), output_mindspore.asnumpy(), atol=1e-5),
                    f"Outputs of {module_name} are not close enough.",
                )


if __name__ == "__main__":
    unittest.main()
