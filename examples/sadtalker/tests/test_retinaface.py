from models.face3d.facexlib.retinaface import RetinaFace
from mindspore import context
import mindspore as ms
import numpy as np
from mindspore import dtype as mstype

from PIL import Image


context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
net = RetinaFace()


def test_inference_net():
    img_path = "examples/source_image/people_0.png"
    image = np.array(Image.open(img_path))
    bboxes = net.detect_faces(image, 0.97)
    return bboxes


def test_training_net():
    raise NotImplementedError()


if __name__ == "__main__":
    test_inference_net()
