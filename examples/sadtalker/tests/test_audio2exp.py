from modules.audio2exp.expnet import ExpNet
from mindspore import context
import mindspore as ms
import numpy as np
from mindspore import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
net = ExpNet()


def test_inference_net():
    bs = 16
    T = 5

    audio_x = ms.Tensor(np.random.randn(bs*T, 1, 80, 16),
                        dtype=ms.float32)  # bs*T, 1, 80, 16
    ref = ms.Tensor(np.random.randn(bs, T, 64), dtype=ms.float32)  # bs T 64
    ratio = ms.Tensor(np.random.randn(bs, T), dtype=ms.float32)  # bs T

    # import pdb; pdb.set_trace()

    out = net(audio_x, ref, ratio)

    print(out.shape)


def test_training_net():
    raise NotImplementedError()


if __name__ == "__main__":
    test_inference_net()
