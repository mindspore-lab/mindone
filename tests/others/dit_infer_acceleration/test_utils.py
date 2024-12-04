import importlib

import mindspore as ms


def set_dtype(model, dtype):
    if not model or not isinstance(model, ms.nn.Cell):
        return model
    for p in model.get_parameters():
        p = p.set_dtype(dtype)
    return model


def get_module(module, dtype, args, kwargs):
    if module is None:
        return None
    ms_path, ms_cls_name = module.rsplit(".", 1)
    ms_module_cls = getattr(importlib.import_module(ms_path), ms_cls_name)

    ms_modules_instance = ms_module_cls(*args, **kwargs)
    if dtype == "fp16":
        ms_modules_instance = set_dtype(ms_modules_instance, ms.float16)
    elif dtype == "bf16":
        ms_modules_instance = set_dtype(ms_modules_instance, ms.bfloat16)
    elif dtype == "fp32":
        ms_modules_instance = set_dtype(ms_modules_instance, ms.float32)
    else:
        raise NotImplementedError(f"Dtype {dtype} for model is not implemented")
    return ms_modules_instance


def generate_pipeline(cls, case_config):
    modules_dict = {module_name: get_module(*module_config) for module_name, module_config in case_config.items()}
    return cls(**modules_dict)


def mock_fn(obj, mock_info, mocker):
    for func_name, return_val in mock_info.items():
        mock_fn = mocker.patch.object(obj, func_name)
        mock_fn.return_value = return_val
