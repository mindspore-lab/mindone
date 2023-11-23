import collections
from collections import OrderedDict
from typing import Dict, Iterable, Iterator

from mindspore import nn


class CellDict(nn.Cell):
    def __init__(self, *args, **kwargs):
        auto_prefix = kwargs.get("auto_prefix") if "auto_prefix" in kwargs.keys() else True
        nn.Cell.__init__(self, auto_prefix)
        if len(args) == 1:
            self.update(args[0])

    def __getitem__(self, key: str):
        return self._cells[key]

    def __setitem__(self, key: str, module) -> None:
        self._cells[key] = module

    def __delitem__(self, key: str) -> None:
        del self._cells[key]

    def __len__(self) -> int:
        return len(self._cells)

    def __iter__(self) -> Iterator[str]:
        return iter(self._cells)

    def __contains__(self, key: str) -> bool:
        return key in self._cells

    def clear(self) -> None:
        self._cells.clear()

    def pop(self, key: str):
        value = self[key]
        del self[key]
        return value

    def keys(self) -> Iterable[str]:
        return self._cells.keys()

    def items(self):
        return self._cells.items()

    def values(self):
        return list(self._cells.values())

    def update(self, modules) -> None:
        if not isinstance(modules, collections.abc.Iterable):
            raise TypeError("CellDict.update should be called with an "
                            "iterable of key/value pairs, but got " +
                            type(modules).__name__)

        if isinstance(modules, (OrderedDict, CellDict, collections.abc.Mapping)):
            for key, module in modules.items():
                self[key] = module
        else:
            for index, module in enumerate(modules):
                if not isinstance(module, collections.abc.Iterable):
                    raise TypeError("CellDict update sequence element "
                                    "#" + str(index) + " should be Iterable; is" +
                                    type(module).__name__)
                if not len(module) == 2:
                    raise ValueError("CellDict update sequence element "
                                     "#" + str(index) + " has length " + str(len(module)) +
                                     "; 2 is required")
                self[module[0]] = module[1]

    def construct(self, *inputs):
        raise NotImplementedError


class IntermediateLayerGetter(CellDict):
    """ implementation of `torchvision.models._utils.IntermediateLayerGetter`
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Cell): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    """

    def __init__(self, model: nn.Cell, return_layers: Dict[str, str]) -> None:
        super().__init__()
        if not set(return_layers).issubset([name for name, _ in model.cells_and_names()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        for name, module in model.cells_and_names():
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        self.return_layers = orig_return_layers

    def construct(self, x):
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
