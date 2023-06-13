from typing import Optional, List
from fnmatch import fnmatch
import mindspore.nn as nn

def freeze_delta(model: nn.Cell, mode: str,
                 include: Optional[List[str]] = None,
                 exclude: Optional[List[str]] = None) -> None:
    """
    根据微调算法类型以及指定模块列表冻结网络。
    目前已实现lora和prefixtuning两种微调算法的冻结模式。

    :param model: 需要冻结的模型实例，必填。
    :param mode: 微调算法类型，必填。可选填'lora'或'prefixtuning'。
    :param include: 需要冻结的模块名列表， 选填。
                    模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为False。
                    列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
    :param exclude: 不冻结的模块名列表， 选填。
                    模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为True。
                    列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
                    当include和exclude列表项冲突时，对该项匹配到的模块不做任何处理。
    """
    print("Start to freeze model for delta, mode: %s, include list: %s, exclude list: %s"%
                (mode, include, exclude))
    if not isinstance(model, nn.Cell):
        raise TypeError("A Cell is required for argument 'model'.")

    if isinstance(mode, str) and mode.lower() == 'lora':
        _freeze_for_mode(model, mode)
    else:
        raise ValueError(f"An argument of string type from {DELTA_LIST} is required.")

    if include or exclude:
        try:
            freeze_modules(model, include, exclude)
        except Exception as ex:
            raise Exception(f"Exception occurred when freeze model for delta, error message: {str(ex)}") from ex

    print("End to freeze model for delta.")

def _freeze_for_mode(model: nn.Cell, mode: str) -> None:
    """
    根据微调算法冻结网络。

    :param model: 需要冻结的模型实例，必填。
    """
    delta_name = '*tk_delta_' + mode + '*'
    freeze_modules(model, include=['*'], exclude=[delta_name])

def freeze_modules(model: nn.Cell,
                   include: Optional[List[str]] = None,
                   exclude: Optional[List[str]] = None) -> None:
    """
    根据指定模块列表冻结网络。

    :param model: 需要冻结的模型实例， 必填。
    :param include: 需要冻结的模块名列表， 选填。
                    模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为False。
                    列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
    :param exclude: 不冻结的模块名列表， 选填。
                    模糊匹配列表中所有模块名，挨个将匹配到的模块的requires_grad设置为True。
                    列表项支持配置符号*，代表任意字符串，格式如 ['*', '*dense*'，'*.dense.*', '*.dense.*.bias']
                    当include和exclude列表项冲突时，对该项匹配到的模块不做任何处理。
    """
    print("Start to freeze model, include list: %s, exclude list: %s"%(include, exclude))
    if not isinstance(model, nn.Cell):
        raise TypeError("A Cell is required for argument 'model'.")

    if include is None and exclude is None:
        raise ValueError("Argument 'include' (pos 2) and 'exclude' (pos 3) can't be None together.")

    if include is not None and (not isinstance(include, List) or not include):
        raise TypeError("A non-empty list is required for argument 'include' (pos 2).")

    if exclude is not None and (not isinstance(exclude, List) or not exclude):
        raise TypeError("A non-empty list is required for argument 'exclude' (pos 3).")

    _freeze_from_list(model, include, exclude)
    print("End to freeze model.")

def _freeze_from_list(model, include, exclude):
    """
    根据include/exclude列表冻结网络。
    """
    for name, param in model.parameters_and_names():
        if _match_str_and_list(name, include) and not _match_str_and_list(name, exclude):
            param.requires_grad = False
        elif not _match_str_and_list(name, include) and _match_str_and_list(name, exclude):
            param.requires_grad = True

def _match_str_and_list(m_str, m_list: Optional[List[str]] = None) -> bool:
    """
    校验字符串是否与列表中某一项匹配。
    :param m_str: 完整字符串
    :param m_list: 有关键词的列表
    """
    if m_list is None:
        return False

    for key in m_list:
        if not isinstance(key, str):
            raise TypeError(f"List item '{key}' is not a string.")

        if fnmatch(m_str, key):
            return True

    return False