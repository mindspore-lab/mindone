"""
source: https://github.com/xinntao/facexlib/blob/master/facexlib/alignment/convert_98_to_68_landmarks.py
"""
import os
from copy import deepcopy
import numpy as np
import mindspore as ms
from .retinaface import RetinaFace
from .awing_fan import FAN
from .parsenet import ParseNet


def landmark_98_to_68(landmark_98):
    """Transfer 98 landmark positions to 68 landmark positions.
    Args:
        landmark_98(numpy array): Polar coordinates of 98 landmarks, (98, 2)
    Returns:
        landmark_68(numpy array): Polar coordinates of 98 landmarks, (68, 2)
    """

    landmark_68 = np.zeros((68, 2), dtype='float32')
    # cheek
    for i in range(0, 33):
        if i % 2 == 0:
            landmark_68[int(i / 2), :] = landmark_98[i, :]
    # nose
    for i in range(51, 60):
        landmark_68[i - 24, :] = landmark_98[i, :]
    # mouth
    for i in range(76, 96):
        landmark_68[i - 28, :] = landmark_98[i, :]
    # left eyebrow
    landmark_68[17, :] = landmark_98[33, :]
    landmark_68[18, :] = (landmark_98[34, :] + landmark_98[41, :]) / 2
    landmark_68[19, :] = (landmark_98[35, :] + landmark_98[40, :]) / 2
    landmark_68[20, :] = (landmark_98[36, :] + landmark_98[39, :]) / 2
    landmark_68[21, :] = (landmark_98[37, :] + landmark_98[38, :]) / 2
    # right eyebrow
    landmark_68[22, :] = (landmark_98[42, :] + landmark_98[50, :]) / 2
    landmark_68[23, :] = (landmark_98[43, :] + landmark_98[49, :]) / 2
    landmark_68[24, :] = (landmark_98[44, :] + landmark_98[48, :]) / 2
    landmark_68[25, :] = (landmark_98[45, :] + landmark_98[47, :]) / 2
    landmark_68[26, :] = landmark_98[46, :]
    # left eye
    LUT_landmark_68_left_eye = [36, 37, 38, 39, 40, 41]
    LUT_landmark_98_left_eye = [60, 61, 63, 64, 65, 67]
    for idx, landmark_98_index in enumerate(LUT_landmark_98_left_eye):
        landmark_68[LUT_landmark_68_left_eye[idx],
                    :] = landmark_98[landmark_98_index, :]
    # right eye
    LUT_landmark_68_right_eye = [42, 43, 44, 45, 46, 47]
    LUT_landmark_98_right_eye = [68, 69, 71, 72, 73, 75]
    for idx, landmark_98_index in enumerate(LUT_landmark_98_right_eye):
        landmark_68[LUT_landmark_68_right_eye[idx],
                    :] = landmark_98[landmark_98_index, :]

    return landmark_68


def init_detection_model(model_name, half=False, model_rootpath=None):
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half)
        model_path = 'gfpgan/weights/detection_Resnet50_Final.ckpt'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    if os.path.exists(model_path):
        param_dict = ms.load_checkpoint(model_path)
        # remove unnecessary 'module.'
        for k, v in deepcopy(param_dict).items():
            if k.startswith('module.'):
                param_dict[k[7:]] = v
                param_dict.pop(k)
        ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    return model


def init_alignment_model(model_name, half=False, model_rootpath=None):
    if model_name == 'awing_fan':
        model = FAN(num_modules=4, num_landmarks=98)
        model_path = 'gfpgan/weights/alignment_WFLW_4HG.ckpt'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    if os.path.exists(model_path):
        param_dict = ms.load_checkpoint(model_path)
        ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    return model


def init_parsing_model(model_name='bisenet', half=False, model_rootpath=None):
    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)
        model_path = 'checkpoints/ms/parsing_bisenet.ckpt'
        # model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth'
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_path = 'gfpgan/weights/parsing_parsenet.ckpt'
        # model_url = 'https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    if os.path.exists(model_path):
        param_dict = ms.load_checkpoint(model_path)
        ms.load_param_into_net(model, param_dict)
        print(f"loaded the pretrained model from {model_path}.")
    model.set_train(False)

    return model
