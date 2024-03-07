import json
import logging
import os

import cv2
import numpy as np

import mindspore as ms

_logger = logging.getLogger(__name__)


def resize_image(image: np.ndarray, resolution: int) -> np.ndarray:
    h, w = image.shape[:2]
    k = resolution / min(h, w)
    h = int(np.round(h * k / 64.0)) * 64
    w = int(np.round(w * k / 64.0)) * 64

    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)


# use for dataset with different resolution
def resize_image_to_sq(image: np.ndarray, resolution: int) -> np.ndarray:
    h, w = image.shape[:2]
    h_k = resolution / h
    w_k = resolution / w
    h = int(np.round(h * h_k / 64.0)) * 64
    w = int(np.round(w * w_k / 64.0)) * 64

    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4 if min(h_k, w_k) > 1 else cv2.INTER_AREA)


class ControlNetDataset:
    def __init__(
        self,
        tokenizer,
        root_dir="./datasets/fill5",
        image_size=512,
        shuffle=True,
        control_type="canny",
        image_filter_size=None,
        drop_rate=0.5,
    ):
        """
        root_dir: path to dataset folder which contains source/, target/, and prompt.json
        control_type: canny, openpose
        """
        super().__init__()
        # read json annotation files
        self.root_dir = root_dir
        self.control_paths = []  # control signals including canny, segmentation
        self.image_paths = []
        self.captions = []
        self.drop_rate = drop_rate
        if control_type == "openpose":
            # prompt.json may have multiple caption for the same image
            if os.path.exists(root_dir):
                annotations = json.loads(open(os.path.join(root_dir, "prompt.json")).read())
                img_list = os.listdir(os.path.join(root_dir, "source"))
                for img_name in img_list:
                    img_id = int(img_name.split(".")[0])
                    captions = list(filter(lambda x: x["image_id"] == img_id, annotations))
                    for caption in captions:
                        self.control_paths.append(os.path.join("source", img_name))
                        self.image_paths.append(os.path.join("target", img_name))
                        self.captions.append(caption["caption"])
        else:
            if os.path.exists(root_dir):
                with open(os.path.join(root_dir, "prompt.json"), "rt") as f:
                    for line in f:
                        item = json.loads(line)
                        self.control_paths.append(item["source"])
                        self.image_paths.append(item["target"])
                        self.captions.append(item["prompt"])

        _logger.info(f"Total number of training samples: {len(self.image_paths)}")

        self.tokenizer = tokenizer
        self.image_size = image_size
        self.image_filter_size = image_filter_size
        self.shuffle = shuffle
        self.control_type = control_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        returns:
            target: ground truth image, [h w 3] which will be transposed to CHW in LDM
            tokens: tokenized prompt
            control: input control signal, normalized canny image, in shape [h w c]
        """
        control_path = os.path.join(self.root_dir, self.control_paths[idx])
        img_path = os.path.join(self.root_dir, self.image_paths[idx])
        caption = self.captions[idx]

        # control and target image preprocess
        source = cv2.imread(control_path)
        target = cv2.imread(img_path)

        # resize
        if (target.shape[0] != self.image_size) or (target.shape[1] != self.image_size):
            if self.control_type == "canny":
                source = resize_image(source, self.image_size)
                target = resize_image(target, self.image_size)
            elif self.control_type == "openpose":
                source = resize_image_to_sq(source, self.image_size)
                target = resize_image_to_sq(target, self.image_size)

        # bgr to rgb
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # differ for different mask
        if self.control_type == "canny":
            # Normalize source-canny images to [0, 1].
            source = source.astype(np.float32) / 255.0

            # TODO: control hwc -> chw?? for canny?

        elif self.control_type == "openpose":
            # Normalize source-openpose images to [0, 1].
            source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1], for sd input
        target = (target.astype(np.float32) / 127.5) - 1.0

        # caption preprocess
        # drop captions by drop rate
        if np.random.uniform(0, 1) < self.drop_rate:
            caption = ""

        caption_input = self.tokenize(caption)

        # correspond to ControlNetLDM inputs: x, c, control
        return target, np.array(caption_input, dtype=np.int32), source

    def tokenize(self, text):
        # text = self.dropout_caption(text)
        SOT_TEXT = self.tokenizer.sot_text  # "[CLS]"
        EOT_TEXT = self.tokenizer.eot_text  # "[SEP]"
        CONTEXT_LEN = self.tokenizer.context_length

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        pad_token = eot_token
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        # TODO: use pad token instead of zero
        # result = np.zeros([CONTEXT_LEN])
        result = np.array([pad_token] * CONTEXT_LEN) + eot_token
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[: CONTEXT_LEN - 1] + [eot_token]
        result[: len(tokens)] = tokens

        return result


def build_dataset_controlnet(
    data_path,
    train_batch_size,
    tokenizer,
    image_size,
    filter_small_size=False,
    image_filter_size=256,
    device_num=1,
    rank_id=0,
    control_type="canny",
):
    dataset = ControlNetDataset(
        tokenizer,
        root_dir=data_path,  # './datasets/fill5',
        image_size=image_size,
        shuffle=True,
        control_type=control_type,
    )

    print("Total number of samples: ", len(dataset))

    dataloader = ms.dataset.GeneratorDataset(
        source=dataset,
        column_names=[
            "image",
            "caption_tokens",
            "control",
        ],
        num_shards=device_num,
        shard_id=rank_id,
        python_multiprocessing=True,
        shuffle=True,
        num_parallel_workers=4,  # 8 OOM
        max_rowsize=32,
    )

    dl = dataloader.batch(
        train_batch_size,
        drop_remainder=True,
    )

    return dl


if __name__ == "__main__":
    build_dataset_controlnet("./datasets/coco10sample", 4, None, 256, control_type="openpose")
