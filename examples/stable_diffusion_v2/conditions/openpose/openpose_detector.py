import logging
import os

import numpy as np

from . import util
from .body import Body
from .hand import Hand

_logger = logging.getLogger(__name__)

# from annotator.util import annotator_ckpts_path
_annotator_ckpts_path = os.path.join(os.path.abspath(os.path.join(__file__, "../../..")), "models")


class OpenposeDetector:
    def __init__(self, annotator_ckpts_path=_annotator_ckpts_path):
        body_modelpath = os.path.join(annotator_ckpts_path, "ms_body_pose_model.ckpt")
        hand_modelpath = os.path.join(annotator_ckpts_path, "ms_hand_pose_model.ckpt")

        if not os.path.exists(hand_modelpath):
            _logger.error(f"{hand_modelpath} not exist.")
        if not os.path.exists(body_modelpath):
            _logger.error(f"{body_modelpath} not exist.")

        self.body_estimation = Body(body_modelpath)
        self.hand_estimation = Hand(hand_modelpath)

    def __call__(self, oriImg, hand=False):
        oriImg = oriImg[:, :, ::-1].copy()

        # cong TODO: make sure below codes runs without gradients update
        self.body_estimation.model.set_train(False)
        self.hand_estimation.model.set_train(False)

        candidate, subset = self.body_estimation(oriImg)
        canvas = np.zeros_like(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)
        if hand:
            hands_list = util.handDetect(candidate, subset, oriImg)
            all_hand_peaks = []
            for x, y, w, is_left in hands_list:
                peaks = self.hand_estimation(oriImg[y : y + w, x : x + w, :])
                peaks[:, 0] = np.where(peaks[:, 0] == 0, peaks[:, 0], peaks[:, 0] + x)
                peaks[:, 1] = np.where(peaks[:, 1] == 0, peaks[:, 1], peaks[:, 1] + y)
                all_hand_peaks.append(peaks)
            canvas = util.draw_handpose(canvas, all_hand_peaks)
        return canvas, dict(candidate=candidate.tolist(), subset=subset.tolist())
