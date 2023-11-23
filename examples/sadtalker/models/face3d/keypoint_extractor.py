import mindspore as ms
from mindspore import nn
import numpy as np
from tqdm import tqdm
import time
import os

from models.face3d.facexlib import landmark_98_to_68


class KeypointExtractor():
    def __init__(self, detector, det_net):

        self.detector = detector
        self.det_net = det_net

    def extract_keypoint(self, images, name=None, info=True):
        keypoints = []
        if isinstance(images, list):
            keypoints = []
            if info:
                i_range = tqdm(images, desc='landmark Det:')
            else:
                i_range = images

            for image in i_range:
                current_kp = self.extract_keypoint(image)
                # current_kp = self.detector.get_landmarks(np.array(image))
                if np.mean(current_kp) == -1 and keypoints:
                    keypoints.append(keypoints[-1])
                else:
                    keypoints.append(current_kp[None])

            keypoints = np.concatenate(keypoints, 0)
            np.savetxt(os.path.splitext(name)[0]+'.txt', keypoints.reshape(-1))
            return keypoints
        else:
            while True:
                try:
                    # face detection -> face alignment.
                    img = np.array(images)
                    bboxes = self.det_net.detect_faces(images, 0.97)

                    bboxes = bboxes[0]
                    img = img[int(bboxes[1]):int(bboxes[3]),
                              int(bboxes[0]):int(bboxes[2]), :]

                    keypoints = landmark_98_to_68(
                        self.detector.get_landmarks(img))  # [0]

                    # keypoints to the original location
                    keypoints[:, 0] += int(bboxes[0])
                    keypoints[:, 1] += int(bboxes[1])

                    break

                except RuntimeError as e:
                    if str(e).startswith('CUDA'):
                        print("Warning: out of memory, sleep for 1s")
                        time.sleep(1)
                    else:
                        print(e)
                        break
                except TypeError:
                    print('No face detected in this image')
                    shape = [68, 2]
                    keypoints = -1. * np.ones(shape)
                    break
            if name is not None:
                np.savetxt(os.path.splitext(name)[
                           0]+'.txt', keypoints.reshape(-1))
            return keypoints


def read_video(filename):
    import cv2
    from PIL import Image
    frames = []
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frames.append(frame)
        else:
            break
    cap.release()
    return frames


def run(data):
    filename, opt, device = data
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    kp_extractor = KeypointExtractor()
    images = read_video(filename)
    name = filename.split('/')[-2:]
    os.makedirs(os.path.join(opt.output_dir, name[-2]), exist_ok=True)
    kp_extractor.extract_keypoint(
        images,
        name=os.path.join(opt.output_dir, name[-2], name[-1])
    )


if __name__ == "__main__":
    from models.face3d.facexlib import init_detection_model, init_alignment_model
    # gfpgan/weights
    root_path = 'gfpgan/weights'
    detector = init_alignment_model(
        'awing_fan', model_rootpath=root_path)
    det_net = init_detection_model(
        'retinaface_resnet50', half=False, model_rootpath=root_path)

    extractor = KeypointExtractor(detector, det_net)
