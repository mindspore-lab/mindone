import argparse
import os

import cv2
import yaml
from coca import CoCa
from data.dataset import create_transforms
from modules.encoders.tokenizer import BpeTokenizer
from PIL import Image

import mindspore as ms
from mindspore import load_checkpoint, load_param_into_net


def extract_mid_frame(video_path, output_image=None):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video file")
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
            cv2.imwrite(output_image, frame)
    else:
        print("Error reading the mid frame")

    # Release the video capture object
    cap.release()

    return middle_frame


def create_model(config, checkpoint_path=None):
    with open(
        config,
        "r",
    ) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)
    # print(model_cfg)
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

    def __call__(self, video_path: str, repetition_penalty: float = 1.0, seq_len: int = 30):
        middle_frame = extract_mid_frame(video_path)

        img = ms.Tensor(self.transforms(middle_frame), dtype=ms.float32)
        self.model.set_train(False)
        generated = self.model.generate(img)

        caption = self.tokenizer.decode(generated[0].asnumpy()).split("<|endoftext|>")[0].replace("<|startoftext|>", "")

        return caption


if __name__ == "__main__":
    ms.set_context(mode=ms.PYNATIVE_MODE)

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video_path", default="moscow.mp4", type=str, help="data path")
    parser.add_argument("-c", "--config", default="./config/coca_vit-l-14.yaml", type=str, help="config path")
    parser.add_argument("--model_path", default="./models/coca_model.ckpt", type=str, help="coca model checkpoint path")
    args = parser.parse_args()

    vc = VideoCaptioner(config=args.config, model_path=args.model_path)
    video_path = args.video_path
    caption = vc(video_path)

    print(caption)

    # im = Image.open("/disk1/mindone/songyuanwei/mindone/examples/coca/cat.jpg").convert("RGB")
    # im = vision.ToTensor()(im)
    # im = ms.Tensor(im).expand_dims(0)
    # im = ops.ResizeBilinearV2()(im, (224, 224))
    # # im = im.squeeze()
    # print(im.shape)
    # model = create_model(config)
    #
    # model.set_train(False)
    # text = model.generate(im)
    # tokenizer = BpeTokenizer()
    # output = tokenizer.decode(text[0].asnumpy()).split("<|endoftext|>")[0].replace("<|startoftext|>", "")
    #
    # print(output)
