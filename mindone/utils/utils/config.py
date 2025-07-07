import importlib
from typing import Union

from omegaconf import DictConfig, ListConfig, OmegaConf


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def parse_bool_str(b: str):
    """Allow input args to be either str2bool or str (e.g. a filepath)."""
    if b.lower() not in ["false", "true"]:
        return b
    if b.lower() in ["false"]:
        return False
    return True


def instantiate_from_config(config: Union[DictConfig, ListConfig, str]) -> object:
    """
    Args:
        config: a config dict or a string path to config dict for instantiating a class
    Return:
        instantiated object
    """
    if isinstance(config, str):
        config = OmegaConf.load(config).model
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string: str, reload: bool = False) -> object:
    """TODO: debug
    if string.startswith('mindone'):
        string = '../../' + string
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
