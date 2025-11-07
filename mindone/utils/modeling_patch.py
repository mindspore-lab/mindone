import inspect
from functools import wraps

import mindspore as ms
from mindspore import mint, nn

SKIP_CLASSES = {nn.Dropout}
# Store original __init__ for manual restore
_ORIG_INITS = {}


def patch_nn_default_dtype(dtype=ms.float32, force=False):
    """
    Iterate over all Cells under nn and mint.nn,
    automatically set or force the default dtype in __init__ if supported.

    Args:
        dtype (mindspore.dtype): target dtype to enforce
        force (bool): if True, even when user passes dtype explicitly, override it
    """
    for module in [ms.nn, mint.nn]:
        for name in dir(module):
            attr = getattr(module, name)
            if inspect.isclass(attr) and issubclass(attr, nn.Cell):
                if attr in SKIP_CLASSES:
                    continue  # skip specified classes
                sig = inspect.signature(attr.__init__)
                if "dtype" in sig.parameters:
                    if attr not in _ORIG_INITS:
                        _ORIG_INITS[attr] = attr.__init__

                    _orig_init = attr.__init__

                    @wraps(_orig_init)
                    def _new_init(self, *args, _orig_init=_orig_init, **kwargs):
                        if force or "dtype" not in kwargs:
                            kwargs["dtype"] = dtype
                        return _orig_init(self, *args, **kwargs)

                    setattr(attr, "__init__", _new_init)


def unpatch_nn_default_dtype():
    """
    Manually restore the original __init__ of all patched nn / mint.nn Cells.
    """
    for cls, orig_init in _ORIG_INITS.items():
        cls.__init__ = orig_init
    _ORIG_INITS.clear()
