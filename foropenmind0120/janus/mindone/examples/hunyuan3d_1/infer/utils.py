# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import time
from functools import wraps

from mindone.utils.params import count_params
from mindone.utils.seed import set_random_seed


def seed_everything(seed):
    """
    seed everthing
    """
    set_random_seed(seed)


def timing_decorator(category: str):
    """
    timing_decorator: record time
    """

    def decorator(func):
        func.call_count = 0

        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            func.call_count += 1
            print(f"[HunYuan3D]-[{category}], cost time: {elapsed_time:.4f}s")  # huiwen
            return result

        return wrapper

    return decorator


def get_parameter_number(model):
    total_num, trainable_num = count_params(model)
    return {"Total": total_num, "Trainable": trainable_num}


def set_parameter_grad_false(model):
    for p in model.get_parameters():
        p.requires_grad = False


def str_to_bool(s):
    if s.lower() in ["true", "t", "yes", "y", "1"]:
        return True
    elif s.lower() in ["false", "f", "no", "n", "0"]:
        return False
    else:
        raise "bool arg must one of ['true', 't', 'yes', 'y', '1', 'false', 'f', 'no', 'n', '0']"
