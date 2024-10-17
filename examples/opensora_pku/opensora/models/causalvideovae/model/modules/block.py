from mindspore import nn


class Block(nn.Cell):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
