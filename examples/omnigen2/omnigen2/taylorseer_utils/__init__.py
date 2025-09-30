# Adapted from https://github.com/VectorSpaceLab/OmniGen2/blob/main/omnigen2/taylorseer_utils/__init__.py
# Copied from https://github.com/Shenyi-Z/TaylorSeer/blob/main/TaylorSeers-xDiT/taylorseer_flux/taylorseer_utils/__init__.py

import math

from mindspore import Tensor


def derivative_approximation(cache_dic: dict, current: dict, feature: Tensor):
    """
    Compute derivative approximation.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    difference_distance = current["activated_steps"][-1] - current["activated_steps"][-2]

    updated_taylor_factors = {}
    updated_taylor_factors[0] = feature

    for i in range(cache_dic["max_order"]):
        if (
            cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]].get(i, None) is not None
        ) and (current["step"] > cache_dic["first_enhance"] - 2):
            updated_taylor_factors[i + 1] = (
                updated_taylor_factors[i]
                - cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][i]
            ) / difference_distance
        else:
            break

    cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = updated_taylor_factors


def taylor_formula(cache_dic: dict, current: dict) -> Tensor:
    """
    Compute Taylor expansion error.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    x = current["step"] - current["activated_steps"][-1]
    # x = current['t'] - current['activated_times'][-1]
    output = 0

    for i in range(len(cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]])):
        output += (
            (1 / math.factorial(i))
            * cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]][i]
            * (x**i)
        )

    return output


def taylor_cache_init(cache_dic: dict, current: dict):
    """
    Initialize Taylor cache and allocate storage for different-order derivatives in the Taylor cache.

    :param cache_dic: Cache dictionary
    :param current: Information of the current step
    """
    if (current["step"] == 0) and (cache_dic["taylor_cache"]):
        cache_dic["cache"][-1][current["stream"]][current["layer"]][current["module"]] = {}
