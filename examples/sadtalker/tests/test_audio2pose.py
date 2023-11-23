from models.audio2pose.audio2pose import Audio2Pose
from models.audio2pose.audio_encoder import AudioEncoder
from mindspore import context
import mindspore as ms
import numpy as np
from mindspore import dtype as mstype
from mindspore.amp import auto_mixed_precision
from yacs.config import CfgNode as CN

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend", device_id=7)

# context.set_context(mode=context.PYNATIVE_MODE,
#                     device_target="Ascend", device_id=7)


def get_config(config_path):
    fcfg_pose = open(config_path)
    cfg_pose = CN.load_cfg(fcfg_pose)
    cfg_pose.freeze()
    return cfg_pose


def test_audioencoder():
    bs = 16
    T = 100
    input_x = ms.Tensor(np.random.randn(bs, T, 1, 80, 16), dtype=ms.float32)
    net = AudioEncoder()
    out = net(input_x)
    print(out.shape)


def test_inference_net(net):
    bs = 16
    T = 100
    frame_len = 33

    input_x = {}
    input_x["ref"] = ms.Tensor(np.random.randn(
        bs, 1, 70), dtype=ms.float32)  # bs 1 70
    input_x["class"] = ms.Tensor(
        np.random.randint(0, 10, bs), dtype=ms.int16)  # bs
    input_x["gt"] = ms.Tensor(np.random.randint(
        0, 10, (bs, frame_len+1, 73)), dtype=ms.int16)  # bs frame_len+1 73
    input_x['indiv_mels'] = ms.Tensor(np.random.randn(
        bs, T, 1, 80, 16), dtype=ms.float32)  # bs T 1 80 16]
    input_x['num_frames'] = ms.Tensor(bs, dtype=ms.int16)  # bs

    out = net.test(input_x)

    print(out.shape)


def test_training_net():
    raise NotImplementedError()


if __name__ == "__main__":
    config_path = "config/audio2pose.yaml"
    cfg_pose = get_config(config_path)
    net = Audio2Pose(cfg_pose)
    auto_mixed_precision(net, "O2")
    test_inference_net(net)
