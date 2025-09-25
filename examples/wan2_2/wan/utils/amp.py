from typing import Any

from mindspore._c_expression.amp import AmpLevel
from mindspore.train.amp import AmpDecorator


def autocast(dtype: Any) -> None:
    """A context manager that allow regions of your script to run in the given dtype.
    Args:
        dtype (mindspore.dtype): The target dtype of the operations to be casted.

    Example:
        >>> import mindspore as ms
        >>> import mindspore.mint as mint
        >>> net = mint.nn.Linear(256, 256, dtype=ms.float16)
        >>> x = mint.ones((1, 256), dtype=ms.float32)
        >>> with autocast(ms.float32):
        ...     output = net(x)
    """
    return AmpDecorator(AmpLevel.AmpO3, dtype, [], [])
