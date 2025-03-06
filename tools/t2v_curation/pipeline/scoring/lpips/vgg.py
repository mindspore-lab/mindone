import math
import logging
import os
from typing import List, Union
from mindspore import nn, Tensor, load_checkpoint, load_param_into_net
import mindspore.common.initializer as init

_logger = logging.getLogger(__name__)

cfg_vgg16 = [64, 64, "M",
             128, 128, "M",
             256, 256, 256, "M",
             512, 512, 512, "M",
             512, 512, 512, "M"]

def _make_layers(cfg: List[Union[str, int]], batch_norm=False, in_channels=3):
    """Construct the sequence of convolution and pooling layers."""
    layers = []
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, pad_mode="pad", padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)

class VGG(nn.Cell):
    def __init__(self, cfg, num_classes=1000, in_channels=3, drop_rate=0.5, batch_norm=False):
        super(VGG, self).__init__()
        self.features = _make_layers(cfg, batch_norm=batch_norm, in_channels=in_channels)
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=drop_rate),
            nn.Dense(4096, num_classes)
        ])
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(math.sqrt(5), mode="fan_out", nonlinearity="relu"),
                                     cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype)
                    )
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Normal(0.01), cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer("zeros", cell.bias.shape, cell.bias.dtype)
                    )

    def construct(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def load_from_pretrained(self, ckpt_path: str):
        if not os.path.exists(ckpt_path):
            raise ValueError(f"Checkpoint not found: {ckpt_path}. You may download it at"
                             f"https://download.mindspore.cn/toolkits/mindcv/vgg/vgg16-95697531.ckpt")
        state_dict = load_checkpoint(ckpt_path)
        m, u = load_param_into_net(self, state_dict)
        if len(m) > 0:
            print("missing keys:")
            print(m)
        if len(u) > 0:
            print("unexpected keys:")
            print(u)

        _logger.info("loaded pretrained VGG16 weights from {}".format(ckpt_path))