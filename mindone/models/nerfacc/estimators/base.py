from typing import Any

from mindspore import Parameter, Tensor, nn


class AbstractEstimator(nn.Cell):
    """An abstract Transmittance Estimator class for Sampling."""

    def __init__(self) -> None:
        super().__init__()

    def sampling(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def update_every_n_steps(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def register_buffer_ms(self, name: str, tensor: Tensor):
        return setattr(self, name, Parameter(default_input=tensor, requires_grad=False))
