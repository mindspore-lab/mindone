from packaging import version

def flag_fix_optim_global_step():
    # when the version of mindspore bigger than 2.2.0, it should update global step explicitly.
    return version.parse(ms.__version__) >= version.parse("2.2.0")


