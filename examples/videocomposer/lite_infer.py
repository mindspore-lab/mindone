#!/usr/bin/env python
from deploy.config import Config
from deploy.engine import inference_multi_lite, inference_single_lite


def main():
    """
    Main function to spawn the train and test process.
    """
    cfg = Config(load=True)
    assert hasattr(cfg, "TASK_TYPE"), "cfg must have attribute 'TASK_TYPE'!"
    task_type = cfg.TASK_TYPE
    print(f"TASK TYPE: {task_type}")
    if task_type == "MULTI_TASK":
        inference_multi_lite(cfg.cfg_dict)
    elif task_type == "SINGLE_TASK":
        inference_single_lite(cfg.cfg_dict)
    else:
        raise NotImplementedError(f"TASK TYPE: {task_type} is not supported!")


if __name__ == "__main__":
    main()
