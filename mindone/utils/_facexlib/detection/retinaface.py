from collections import OrderedDict
from typing import Dict

import cv2
import numpy as np
from PIL import Image

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint, nn

from ._resnet50 import resnet50
from .align_trans import get_reference_facial_points, warp_and_crop_face
from .retinaface_net import FPN, SSH, MobileNetV1, make_bbox_head, make_class_head, make_landmark_head
from .retinaface_utils import PriorBox, batched_decode, batched_decode_landm, decode, decode_landm, py_cpu_nms


def generate_config(network_name):
    cfg_mnet = {
        "name": "mobilenet0.25",
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
        "steps": [8, 16, 32],
        "variance": [0.1, 0.2],
        "clip": False,
        "loc_weight": 2.0,
        "gpu_train": True,
        "batch_size": 32,
        "ngpu": 1,
        "epoch": 250,
        "decay1": 190,
        "decay2": 220,
        "image_size": 640,
        "return_layers": {"stage1": 1, "stage2": 2, "stage3": 3},
        "in_channel": 32,
        "out_channel": 64,
    }

    cfg_re50 = {
        "name": "Resnet50",
        "min_sizes": [[16, 32], [64, 128], [256, 512]],
        "steps": [8, 16, 32],
        "variance": [0.1, 0.2],
        "clip": False,
        "loc_weight": 2.0,
        "gpu_train": True,
        "batch_size": 24,
        "ngpu": 4,
        "epoch": 100,
        "decay1": 70,
        "decay2": 90,
        "image_size": 840,
        "return_layers": {"layer2": 1, "layer3": 2, "layer4": 3},
        "in_channel": 256,
        "out_channel": 256,
    }

    if network_name == "mobile0.25":
        return cfg_mnet
    elif network_name == "resnet50":
        return cfg_re50
    else:
        raise NotImplementedError(f"network_name={network_name}")


class IntermediateLayerGetter(nn.Cell):
    """
    Module wrapper that returns intermediate layers from a model

    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Cell
    twice in the forward if you want this to work.

    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.

    Args:
        model (nn.Cell): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).

    Examples::

        >>> m = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        >>> # extract layer1 and layer3, giving as names `feat1` and feat2`
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    __annotations__ = {
        "return_layers": Dict[str, str],
    }

    def __init__(self, model: nn.Cell, return_layers: Dict[str, str]) -> None:
        super().__init__()

        if not set(return_layers).issubset([name for name, _ in model.name_cells().items()]):
            raise ValueError("return_layers are not present in model")
        orig_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        for name, module in model.name_cells().items():
            super().__setattr__(name, module)
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        self.return_layers = orig_return_layers

    def construct(self, x):
        out = OrderedDict()
        for name, module in self._cells.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class RetinaFace(nn.Cell):
    def __init__(self, network_name="resnet50", half=False, phase="test", **kwargs):
        super(RetinaFace, self).__init__()
        self.half_inference = half
        cfg = generate_config(network_name)
        self.backbone = cfg["name"]

        self.model_name = f"retinaface_{network_name}"
        self.cfg = cfg
        self.phase = phase
        self.target_size, self.max_size = 1600, 2150
        self.resize, self.scale, self.scale1 = 1.0, None, None
        self.mean_tensor = ms.tensor([[[[104.0]], [[117.0]], [[123.0]]]])
        self.reference = get_reference_facial_points(default_square=True)
        # Build network.
        backbone = None
        if cfg["name"] == "mobilenet0.25":
            backbone = MobileNetV1()
            self.body = IntermediateLayerGetter(backbone, cfg["return_layers"])
        elif cfg["name"] == "Resnet50":
            backbone = resnet50()
            self.body = IntermediateLayerGetter(backbone, cfg["return_layers"])

        in_channels_stage2 = cfg["in_channel"]
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]

        out_channels = cfg["out_channel"]
        self.fpn = FPN(in_channels_list, out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = make_class_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.BboxHead = make_bbox_head(fpn_num=3, inchannels=cfg["out_channel"])
        self.LandmarkHead = make_landmark_head(fpn_num=3, inchannels=cfg["out_channel"])

        self.set_train(False)
        if self.half_inference:
            for p in self.get_parameters():
                p.set_dtype(ms.float16)

    def construct(self, inputs):
        out = self.body(inputs)

        if self.backbone == "mobilenet0.25" or self.backbone == "Resnet50":
            out = list(out.values())
        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = mint.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = mint.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        tmp = [self.LandmarkHead[i](feature) for i, feature in enumerate(features)]
        ldm_regressions = mint.cat(tmp, dim=1)

        if self.phase == "train":
            output = (bbox_regressions, classifications, ldm_regressions)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), ldm_regressions)
        return output

    def __detect_faces(self, inputs):
        # get scale
        height, width = inputs.shape[2:]
        self.scale = ms.tensor([width, height, width, height], dtype=ms.float32)
        tmp = [width, height, width, height, width, height, width, height, width, height]
        self.scale1 = ms.tensor(tmp, dtype=ms.float32)

        # forawrd
        if self.half_inference:
            inputs = inputs.half()
        loc, conf, landmarks = self(inputs)

        # get priorbox
        priorbox = PriorBox(self.cfg, image_size=inputs.shape[2:])
        priors = priorbox()

        return loc, conf, landmarks, priors

    # single image detection
    def transform(self, image, use_origin_size):
        # convert to opencv format
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        image = image.astype(np.float32)

        # testing scale
        im_size_min = np.min(image.shape[0:2])
        im_size_max = np.max(image.shape[0:2])
        resize = float(self.target_size) / float(im_size_min)

        # prevent bigger axis from being more than max_size
        if np.round(resize * im_size_max) > self.max_size:
            resize = float(self.max_size) / float(im_size_max)
        resize = 1 if use_origin_size else resize

        # resize
        if resize != 1:
            image = cv2.resize(image, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        # convert to torch.tensor format
        # image -= (104, 117, 123)
        image = image.transpose(2, 0, 1)
        image = ms.Tensor.from_numpy(image).unsqueeze(0)

        return image, resize

    def detect_faces(
        self,
        image,
        conf_threshold=0.8,
        nms_threshold=0.4,
        use_origin_size=True,
    ):
        image, self.resize = self.transform(image, use_origin_size)
        if self.half_inference:
            image = image.half()
        image = image - self.mean_tensor

        loc, conf, landmarks, priors = self.__detect_faces(image)

        boxes = decode(loc.squeeze(0), priors, self.cfg["variance"])
        boxes = boxes * self.scale / self.resize
        boxes = boxes.numpy()

        scores = conf.squeeze(0).numpy()[:, 1]

        landmarks = decode_landm(landmarks.squeeze(0), priors, self.cfg["variance"])
        landmarks = landmarks * self.scale1 / self.resize
        landmarks = landmarks.numpy()

        # ignore low scores
        inds = np.where(scores > conf_threshold)[0]
        boxes, landmarks, scores = boxes[inds], landmarks[inds], scores[inds]

        # sort
        order = scores.argsort()[::-1]
        boxes, landmarks, scores = boxes[order], landmarks[order], scores[order]

        # do NMS
        bounding_boxes = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(bounding_boxes, nms_threshold)
        bounding_boxes, landmarks = bounding_boxes[keep, :], landmarks[keep]
        # self.t['forward_pass'].toc()
        # print(self.t['forward_pass'].average_time)
        # import sys
        # sys.stdout.flush()
        return np.concatenate((bounding_boxes, landmarks), axis=1)

    def __align_multi(self, image, boxes, landmarks, limit=None):
        if len(boxes) < 1:
            return [], []

        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]

        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[2 * j], landmark[2 * j + 1]] for j in range(5)]

            warped_face = warp_and_crop_face(np.array(image), facial5points, self.reference, crop_size=(112, 112))
            faces.append(warped_face)

        return np.concatenate((boxes, landmarks), axis=1), faces

    def align_multi(self, img, conf_threshold=0.8, limit=None):
        rlt = self.detect_faces(img, conf_threshold=conf_threshold)
        boxes, landmarks = rlt[:, 0:5], rlt[:, 5:]

        return self.__align_multi(img, boxes, landmarks, limit)

    # batched detection
    def batched_transform(self, frames, use_origin_size):
        """
        Arguments:
            frames: a list of PIL.Image, or torch.Tensor(shape=[n, h, w, c],
                type=np.float32, BGR format).
            use_origin_size: whether to use origin size.
        """
        from_PIL = True if isinstance(frames[0], Image.Image) else False

        # convert to opencv format
        if from_PIL:
            frames = [cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR) for frame in frames]
            frames = np.asarray(frames, dtype=np.float32)

        # testing scale
        im_size_min = np.min(frames[0].shape[0:2])
        im_size_max = np.max(frames[0].shape[0:2])
        resize = float(self.target_size) / float(im_size_min)

        # prevent bigger axis from being more than max_size
        if np.round(resize * im_size_max) > self.max_size:
            resize = float(self.max_size) / float(im_size_max)
        resize = 1 if use_origin_size else resize

        # resize
        if resize != 1:
            if not from_PIL:
                frames = F.interpolate(frames, scale_factor=resize)
            else:
                frames = [
                    cv2.resize(frame, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
                    for frame in frames
                ]

        # convert to torch.tensor format
        if not from_PIL:
            frames = frames.transpose(1, 2).transpose(1, 3).contiguous()
        else:
            frames = frames.transpose((0, 3, 1, 2))
            frames = ms.Tensor.from_numpy(frames)

        return frames, resize

    def batched_detect_faces(self, frames, conf_threshold=0.8, nms_threshold=0.4, use_origin_size=True):
        """
        Arguments:
            frames: a list of PIL.Image, or np.array(shape=[n, h, w, c],
                type=np.uint8, BGR format).
            conf_threshold: confidence threshold.
            nms_threshold: nms threshold.
            use_origin_size: whether to use origin size.
        Returns:
            final_bounding_boxes: list of np.array ([n_boxes, 5],
                type=np.float32).
            final_landmarks: list of np.array ([n_boxes, 10], type=np.float32).
        """
        # self.t['forward_pass'].tic()
        frames, self.resize = self.batched_transform(frames, use_origin_size)
        frames = frames - self.mean_tensor

        b_loc, b_conf, b_landmarks, priors = self.__detect_faces(frames)

        final_bounding_boxes, final_landmarks = [], []

        # decode
        priors = priors.unsqueeze(0)
        b_loc = batched_decode(b_loc, priors, self.cfg["variance"]) * self.scale / self.resize
        b_landmarks = batched_decode_landm(b_landmarks, priors, self.cfg["variance"]) * self.scale1 / self.resize
        b_conf = b_conf[:, :, 1]

        # index for selection
        b_indice = b_conf > conf_threshold

        # concat
        b_loc_and_conf = mint.cat((b_loc, b_conf.unsqueeze(-1)), dim=2).float()

        for pred, landm, inds in zip(b_loc_and_conf, b_landmarks, b_indice):
            # ignore low scores
            pred, landm = pred[inds, :], landm[inds, :]
            if pred.shape[0] == 0:
                final_bounding_boxes.append(np.array([], dtype=np.float32))
                final_landmarks.append(np.array([], dtype=np.float32))
                continue

            # sort
            # order = score.argsort(descending=True)
            # box, landm, score = box[order], landm[order], score[order]

            # to CPU
            bounding_boxes, landm = pred.numpy(), landm.numpy()

            # NMS
            keep = py_cpu_nms(bounding_boxes, nms_threshold)
            bounding_boxes, landmarks = bounding_boxes[keep, :], landm[keep]

            # append
            final_bounding_boxes.append(bounding_boxes)
            final_landmarks.append(landmarks)
        # self.t['forward_pass'].toc(average=True)
        # self.batch_time += self.t['forward_pass'].diff
        # self.total_frame += len(frames)
        # print(self.batch_time / self.total_frame)

        return final_bounding_boxes, final_landmarks
