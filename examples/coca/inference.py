import argparse
import logging
import os
import sys

import cv2
import yaml
from coca import CoCa
from data.dataset import create_transforms
from modules.encoders.tokenizer import BpeTokenizer
from PIL import Image

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net, ops

# TODO: remove in future when mindone is ready for install
__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)
from mindone.utils.logger import set_logger

logger = logging.getLogger(__name__)


def extract_mid_frame(video_path, output_image=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.Error("Error opening video file")
        return

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the mid-frame index
    mid_frame_index = total_frames // 2

    # Set the current frame position to the mid-frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)

    # Read the mid-frame
    ret, frame = cap.read()
    if ret:
        middle_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        middle_frame = Image.fromarray(middle_frame)

        if output_image is not None:
            # Save the frame as an image
            image_name = f"name_{middle_frame}.jpg"
            cv2.imwrite(os.path.join(output_image, image_name), frame)
    else:
        logger.Error("Error reading the mid frame")

    # Release the video capture object
    cap.release()

    return middle_frame


def extract_frame_equal_interval(video_path, output_image=None, num_frames=16):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.Error("Error opening video file")
        return

    # Get the total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = int((total_frames - 1) / (num_frames - 1))
    current_frame = 0
    # Set the current frame position
    # cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.Warning(f"extract frame from video fail, current frame: {current_frame}")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        if output_image is not None:
            # Save the frame as an image
            image_name = f"name_{current_frame}.jpg"
            cv2.imwrite(os.path.join(output_image, image_name), frame)
        frames.append(frame)
        current_frame += frame_interval
        if current_frame > total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

    # Release the video capture object
    cap.release()

    return frames


def create_model(config, checkpoint_path=None):
    with open(
        config,
        "r",
    ) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = CoCa(**model_cfg)
    if checkpoint_path is not None:
        checkpoint_param = load_checkpoint(checkpoint_path)
        load_param_into_net(model, checkpoint_param)

    return model


class VideoCaptioner:
    def __init__(
        self,
        config: str = "./config/coca_vit-l-14.yaml",
        model_path: str = "./models/coca_model.ckpt",
    ):
        if not os.path.exists(model_path):
            raise ValueError("{} not exist".format(model_path))
        else:
            self.model = create_model(config, checkpoint_path=model_path)
            self.transforms = create_transforms()
            self.tokenizer = BpeTokenizer()

    def __call__(self, video_path: str, frame_type: str = "middle", repetition_penalty: float = 1.0, seq_len: int = 30):
        if frame_type == "middle":
            middle_frame = extract_mid_frame(video_path)
            img = ms.Tensor(self.transforms(middle_frame), dtype=ms.float32)
            self.model.set_train(False)
            generated = self.model.generate(img)
        elif frame_type == "mutil_frames":
            frames = extract_frame_equal_interval(video_path)
            img = ops.stack([ms.Tensor(self.transforms(frame)[0], dtype=ms.float32) for frame in frames], axis=0)
            self.model.set_train(False)
            generated = self.model.generate(img, is_video=True)

        caption = self.tokenizer.decode(generated[0].asnumpy()).split("<|endoftext|>")[0].replace("<|startoftext|>", "")

        return caption


if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", default=None, type=str, help="data path")
    parser.add_argument("-c", "--config", default="./config/coca_vit-l-14.yaml", type=str, help="config path")
    parser.add_argument("--model_path", default="./models/coca_model.ckpt", type=str, help="coca model checkpoint path")
    parser.add_argument(
        "--frame_type",
        default="middle",
        type=str,
        choices=["middle", "mutil_frames"],
        help="extract middle frame or mutil frames for video",
    )
    args = parser.parse_args()
    set_logger(name="")
    vc = VideoCaptioner(config=args.config, model_path=args.model_path)
    video_path = args.video_path
    caption = vc(video_path)

    logger.info(caption)
