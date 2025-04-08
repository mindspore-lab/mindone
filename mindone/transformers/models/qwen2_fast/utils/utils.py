from mindspore import context


def is_pynative():
    """get whether the mode is pynative"""
    mode = context.get_context("mode")
    return mode == context.PYNATIVE_MODE
