import importlib

Module = str
MODULES_BASE = "opensora.models.causalvideovae.model.modules."


def resolve_str_to_obj(str_val, append=True):
    if append:
        str_val = MODULES_BASE + str_val
    module_name, class_name = str_val.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


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
