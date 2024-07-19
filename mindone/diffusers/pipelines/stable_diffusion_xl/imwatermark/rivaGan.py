import numpy as np
import mindspore as ms
import cv2
import os
import time


class RivaWatermark(object):
    encoder = None
    decoder = None

    def __init__(self, watermarks=[], wmLen=32, threshold=0.52):
        self._watermarks = watermarks
        self._threshold = threshold
        if wmLen not in [32]:
            raise RuntimeError('rivaGan only supports 32 bits watermarks now.')
        self._data = ms.Tensor(np.array([self._watermarks], dtype=np.float32))

    @classmethod
    def loadModel(cls):
        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "The `RivaWatermark` class requires onnxruntime to be installed. "
                "You can install it with pip: `pip install onnxruntime`."
            )

        if RivaWatermark.encoder and RivaWatermark.decoder:
            return
        modelDir = os.path.dirname(os.path.abspath(__file__))
        RivaWatermark.encoder = onnxruntime.InferenceSession(
            os.path.join(modelDir, 'rivagan_encoder.onnx'))
        RivaWatermark.decoder = onnxruntime.InferenceSession(
            os.path.join(modelDir, 'rivagan_decoder.onnx'))

    def encode(self, frame):
        if not RivaWatermark.encoder:
            raise RuntimeError('call loadModel method first')

        frame = ms.Tensor(np.array([frame], dtype=np.float32)) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0)

        inputs = {
            'frame': frame.asnumpy(),
            'data': self._data.asnumpy()
        }

        outputs = RivaWatermark.encoder.run(None, inputs)
        wm_frame = outputs[0]
        wm_frame = ms.ops.clamp(ms.Tensor(wm_frame), min=-1.0, max=1.0)
        wm_frame = (
            (wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
        ).asnumpy().astype('uint8')

        return wm_frame

    def decode(self, frame):
        if not RivaWatermark.decoder:
            raise RuntimeError('you need load model first')

        frame = ms.Tensor(np.array([frame], dtype=np.float32)) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0)
        inputs = {
            'frame': frame.asnumpy(),
        }
        outputs = RivaWatermark.decoder.run(None, inputs)
        data = outputs[0][0]
        return np.array(data > self._threshold, dtype=np.uint8)
