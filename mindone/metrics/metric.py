import functools
import logging
from abc import ABC
from copy import deepcopy
from typing import Any, Callable, Dict, List, Union

from mindspore import Tensor, nn

logger = logging.getLogger(__name__)


class Metric(nn.Cell, ABC):
    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__()
        self.update: Callable = self._wrap_update(self.update)
        self.compute: Callable = self._wrap_compute(self.compute)
        self._computed = None
        self._update_count = 0
        self._defaults: Dict[str, Union[List, Tensor]] = {}

    @property
    def update_called(self) -> bool:
        """Returns `True` if `update` or `forward` has been called initialization or last `reset`."""
        return self._update_count > 0

    @property
    def update_count(self) -> int:
        """Get the number of times `update` and/or `forward` has been called since initialization or last `reset`."""
        return self._update_count

    @property
    def metric_state(self) -> Dict[str, Union[List[Tensor], Tensor]]:
        """Get the current state of the metric."""
        return {attr: getattr(self, attr) for attr in self._defaults}

    def add_state(self, name: str, default: Union[list, Tensor]) -> None:
        """Add metric state variable. Only used by subclasses.

        Args:
            name: The name of the state variable. The variable will then be accessible at ``self.name``.
            default: Default value of the state; can either be a :class:`~torch.Tensor` or an empty list.
                The state will be reset to this value when ``self.reset()`` is called.

        Note:
            The values inserted into a list state are deleted whenever :meth:`~Metric.reset` is called. This allows
            device memory to be automatically reallocated, but may produce unexpected effects when referencing list
            states. To retain such values after :meth:`~Metric.reset` is called, you must first copy them to another
            object.

        Raises:
            ValueError:
                If ``default`` is not a ``tensor`` or an ``empty list``.

        """
        if not isinstance(default, (Tensor, list)) or (isinstance(default, list) and default):
            raise ValueError("state variable must be a tensor or any empty list (where you can append tensors)")

        if isinstance(default, Tensor):
            default = default.contiguous()

        setattr(self, name, default)
        self._defaults[name] = deepcopy(default)

    def _wrap_compute(self, compute: Callable) -> Callable:
        @functools.wraps(compute)
        def wrapped_func(*args: Any, **kwargs: Any) -> Any:
            if not self.update_called:
                logger.warning(
                    f"The ``compute`` method of metric {self.__class__.__name__}"
                    " was called before the ``update`` method which may lead to errors,"
                    " as metric states have not yet been updated."
                )

            if self._computed is not None:
                return self._computed

            value = compute(*args, **kwargs)
            self._computed = value

            return value

        return wrapped_func

    def _wrap_update(self, update: Callable) -> Callable:
        @functools.wraps(update)
        def wrapped_func(*args: Any, **kwargs: Any) -> None:
            self._computed = None
            self._update_count += 1
            update(*args, **kwargs)

        return wrapped_func

    def reset(self) -> None:
        """Reset metric state variables to their default value."""
        self._update_count = 0
        self._forward_cache = None
        self._computed = None

        for attr, default in self._defaults.items():
            if isinstance(default, Tensor):
                setattr(self, attr, deepcopy(default))
            else:
                getattr(self, attr).clear()  # delete/free list items
