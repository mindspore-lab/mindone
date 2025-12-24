import math

import mindspore as ms
from mindspore import mint


class CacheContext(ms.nn.Cell):
    """
    Cache context manager for storing and retrieving intermediate computation results in diffusion models.
    Implements Taylor series approximation for efficient cache reuse.

    Args:
        batch_size (int): Batch size of the input tensors
        seq_len (int): Sequence length of the transformer inputs
        inner_dim (int): Hidden dimension size of the transformer
        dtype (ms.dtype): Data type for cache storage, default is ms.bfloat16
        taylorseer_derivative (int): Order of Taylor series approximation, default is 0 (disabled)
    """

    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        inner_dim: int,
        dtype: ms.dtype = ms.bfloat16,
        taylorseer_derivative: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.inner_dim = inner_dim
        self.order = taylorseer_derivative + 1
        self.first_residual = ms.Parameter(
            mint.ones((batch_size, seq_len, inner_dim), dtype=dtype), name="first_residual"
        )
        self.dtype = dtype
        self.enable_taylor = taylorseer_derivative > 0
        if self.enable_taylor:
            self.current_step = ms.Parameter([-1], name="current_step")
            self.last_non_approximated_step = ms.Parameter([-1], name="last_non_approximated_step")
            self.null_indicator = 9999.72  # a large number to indicate null cache
            self.residual = ms.Parameter(
                mint.ones((self.order, batch_size, seq_len, inner_dim), dtype=dtype), name="residual"
            )
        else:
            self.residual = ms.Parameter(mint.ones((batch_size, seq_len, inner_dim), dtype=dtype), name="residual")

    def construct(self, new_residual: ms.Tensor):
        self.update_residual(new_residual)

    def update_residual(self, new_residual: ms.Tensor):
        if self.enable_taylor:
            self.residual = self.approximate_derivative(new_residual)
            self.last_non_approximated_step = self.current_step * 1
        else:
            self.residual = new_residual

    def get_residual(self):
        if self.enable_taylor:
            return self.approximate_value()
        return self.residual

    def update_first_residual(self, new_first_residual: ms.Tensor):
        self.first_residual = new_first_residual

    def approximate_derivative(self, new_residual: ms.Tensor):
        dY_current = mint.ones_like(self.residual, dtype=self.dtype) * self.null_indicator
        dY_current[0] = new_residual
        window = self.current_step - self.last_non_approximated_step

        for i in range(self.order - 1):
            if not self.is_residual_null(self.residual[i]):
                dY_current[i + 1] = (dY_current[i] - self.residual[i]) / window
            else:
                break
        return dY_current

    def approximate_value(self):
        elapsed = self.current_step - self.last_non_approximated_step
        output = mint.zeros((self.batch_size, self.seq_len, self.inner_dim), dtype=self.dtype)
        for i in range(self.order - 1):
            if not self.is_residual_null(self.residual[i]):
                output += (1 / math.factorial(i)) * self.residual[i] * (elapsed ** (i))
            else:
                break
        return output

    def is_residual_null(self, residual: ms.Tensor):
        return residual[0, 0, 0] == self.null_indicator

    def step(self):
        self.current_step += 1


def are_two_tensors_similar(t1, t2, *, threshold):
    if threshold <= 0.0:
        return False

    if t1.shape != t2.shape:
        return False

    mean_diff = (t1 - t2).abs().mean()
    mean_t1 = t1.abs().mean()
    # if parallelized:
    #     mean_diff = DP.all_reduce_sync(mean_diff, "avg")
    #     mean_t1 = DP.all_reduce_sync(mean_t1, "avg")
    diff = mean_diff / mean_t1
    return diff < threshold
