from packaging.version import parse

import mindspore as ms
from mindspore import mint, ops


def get_multinomial_op():
    if parse(ms.__version__) >= parse("2.5"):
        return mint.multinomial
    else:
        if ms.get_context("mode") == 0:
            return ops.multinomial
        else:
            # before ms2.5, mint multinomial doesn't support graph mode
            return mint.multinomial


# multnomial = get_multinomial_op()
