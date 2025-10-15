import numpy as np

import mindspore as ms


def calc_diff(ms_val, pt_val, eps=1e-8, relax=False):
    if isinstance(ms_val, ms.Tensor):
        ms_val = ms_val.asnumpy()

    abs_diff = np.fabs(ms_val - pt_val)
    mae = abs_diff.mean()
    max_ae = abs_diff.max()

    rel_diff = abs_diff / (np.fabs(pt_val) + eps)

    # relax
    if relax:
        rel_diff = abs_diff / (np.fabs(pt_val))
        tot = np.prod(rel_diff.shape)
        n_nan = np.isnan(rel_diff).sum()
        n_inf = np.isinf(rel_diff).sum()
        print(
            "# values: {}, # nan values: {}, # inf values:{}, (nan+inf)/tot: {}".format(
                tot, n_nan, n_inf, (n_nan + n_inf) / tot
            )
        )
        rel_diff = rel_diff[~np.isnan(rel_diff)]
        rel_diff = rel_diff[~np.isinf(rel_diff)]

    mre = rel_diff.mean()
    max_re = rel_diff.max()

    return dict(mae=mae, max_ae=max_ae, mre=mre, max_re=max_re)


def print_diff(ms_val, pt_np):
    pta_val = np.load(pt_np)
    res = calc_diff(ms_val, pta_val)
    print(res)
    return res, pta_val
