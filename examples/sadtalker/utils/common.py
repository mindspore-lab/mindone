import mindspore as ms


def is_ascend():
    return ms.get_context("device_target").lower() == "ascend"
