#!/usr/bin/env python
from vc.config import Config
from vc.export_modules import export_multi, export_single


def main():
    """
    Main function to spawn the train and test process.
    """
    cfg = Config(load=True)
    assert hasattr(cfg, "TASK_TYPE"), "cfg must have attribute 'TASK_TYPE'!"
    task_type = cfg.TASK_TYPE
    print(f"TASK TYPE: {task_type}")
    if task_type == "MULTI_TASK":
        export_multi(cfg.cfg_dict)
    elif task_type == "SINGLE_TASK":
        export_single(cfg.cfg_dict)
    else:
        raise NotImplementedError(f"TASK TYPE: {task_type} is not supported!")


if __name__ == "__main__":
    main()
