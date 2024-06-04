import random

import numpy as np
from gm.data.dataset import Text2ImageDataset
from PIL import Image
from transformers import CLIPImageProcessor


class IPAdapterImageDataset(Text2ImageDataset):
    def __init__(
        self,
        drop_text_prob=0.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.drop_text_prob = drop_text_prob
        self.clip_image_processor = CLIPImageProcessor()

    def __getitem__(self, idx):
        # images preprocess
        image_path = self.local_images[idx]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)

        # clip image preprocess
        clip_image = self.clip_image_processor(image).pixel_values[0]

        # caption preprocess
        if random.random() < self.drop_text_prob:
            caption = ""
        else:
            caption = self.local_captions[idx]

        sample = {
            "image": image,
            "txt": np.array(caption),
            "original_size_as_tuple": np.array([image.shape[0], image.shape[1]]),  # original h, original w
            "target_size_as_tuple": np.array([self.target_size[0], self.target_size[1]]),  # target h, target w
            "crop_coords_top_left": np.array([0, 0]),  # crop top, crop left
            "aesthetic_score": np.array(
                [
                    6.0,
                ]
            ),
        }

        for trans in self.transforms:
            sample = trans(sample)

        # append extra
        sample["clip_img"] = clip_image

        return sample
