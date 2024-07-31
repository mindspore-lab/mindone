import argparse

__all__ = ["str2bool", "check_cfgs_in_parser"]


def str2bool(b: str) -> bool:
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


def check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser) -> None:
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")
