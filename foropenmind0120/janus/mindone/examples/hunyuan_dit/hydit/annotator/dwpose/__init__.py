# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)
import numpy as np

from . import util


def draw_pose(pose, H, W, draw_body=True):
    bodies = pose["bodies"]
    faces = pose["faces"]
    hands = pose["hands"]
    candidate = bodies["candidate"]
    subset = bodies["subset"]
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    if draw_body:
        canvas = util.draw_bodypose(canvas, candidate, subset)

    canvas = util.draw_handpose(canvas, hands)

    canvas = util.draw_facepose(canvas, faces)

    return canvas


def keypoint2bbox(keypoints):
    valid_keypoints = keypoints[keypoints[:, 0] >= 0]  # Ignore keypoints with confidence 0
    if len(valid_keypoints) == 0:
        return np.zeros(4)
    x_min, y_min = np.min(valid_keypoints, axis=0)
    x_max, y_max = np.max(valid_keypoints, axis=0)

    return np.array([x_min, y_min, x_max, y_max])


def expand_bboxes(bboxes, expansion_rate=0.5, image_shape=(0, 0)):
    expanded_bboxes = []
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = map(int, bbox)
        width = x_max - x_min
        height = y_max - y_min

        # 扩展宽度和高度
        new_width = width * (1 + expansion_rate)
        new_height = height * (1 + expansion_rate)

        # 计算新的边界框坐标
        x_min_new = max(0, x_min - (new_width - width) / 2)
        x_max_new = min(image_shape[1], x_max + (new_width - width) / 2)
        y_min_new = max(0, y_min - (new_height - height) / 2)
        y_max_new = min(image_shape[0], y_max + (new_height - height) / 2)

        expanded_bboxes.append([x_min_new, y_min_new, x_max_new, y_max_new])

    return expanded_bboxes


def create_mask(image_width, image_height, bboxs):
    mask = np.zeros((image_height, image_width), dtype=np.float32)
    for bbox in bboxs:
        x1, y1, x2, y2 = map(int, bbox)
        mask[y1 : y2 + 1, x1 : x2 + 1] = 1.0
    return mask
