# reference to https://github.com/Stability-AI/datapipelines

import warnings
from typing import Dict, List, Union

import numpy as np
from gm.util import instantiate_from_config
from omegaconf import DictConfig, ListConfig

import mindspore as ms
from mindspore.dataset.vision import Inter


class Rescaler:
    def __init__(
        self,
        key: Union[List[str], ListConfig, str] = "image",
        isfloat: bool = False,
        strict: bool = True,
    ):
        """
        :param key: the key indicating the sample
        :param isfloat: bool indicating whether input is float in [0,1]
        or uint in [0.255]
        """
        # keeping name of first argument to be 'key' for the sake of backwards compatibility
        if isinstance(key, str):
            key = [key]
        self.keys = set(key)
        self.isfloat = isfloat
        self.strict = strict
        self.has_warned = [False, False]

    def __call__(self, sample: Dict) -> Dict:
        """
        :param sample: Dict containing the speficied key, which should be a numpy array
        :return:
        """
        if not any(map(lambda x: x in sample, self.keys)):
            if self.strict:
                raise KeyError(f"None of {self.keys} in current sample with keys {list(sample.keys())}")
            else:
                if not self.has_warned[0]:
                    self.has_warned[0] = True
                    warnings.warn(
                        f"None of {self.keys} contained in sample"
                        f"(for sample with keys {list(sample.keys())}). "
                        f"Sample is returned unprocessed since strict mode not enabled"
                    )
                return sample

        matching_keys = set(self.keys.intersection(sample))
        if len(matching_keys) > 1:
            if self.strict:
                raise ValueError(
                    f"more than one matching key of {self.keys} in sample {list(sample.keys())}. "
                    f"This should not be the case"
                )
            else:
                if not self.has_warned[1]:
                    warnings.warn(
                        f"more than one matching key of {self.keys} in sample {list(sample.keys())}."
                        f" But strict mode disabled, so returning sample unchanged"
                    )
                    self.has_warned[1] = True
                return sample

        key = matching_keys.pop()

        if self.isfloat:
            sample[key] = sample[key] * 2 - 1.0
        else:
            sample[key] = sample[key] / 127.5 - 1.0

        return sample


class RescalerControlNet(Rescaler):
    def __call__(self, sample: Dict) -> Dict:
        matching_keys = set(self.keys.intersection(sample))

        for key in matching_keys:
            if key == "control":
                if self.isfloat:
                    # already in [0, 1]
                    pass
                else:
                    sample[key] = sample[key] / 255.0
            elif key == "image":
                if self.isfloat:
                    sample[key] = sample[key] * 2 - 1.0
                else:
                    sample[key] = sample[key] / 127.5 - 1.0
            else:
                raise ValueError("Unexpected key in `RescalerControlNet`")
        return sample


class Resize:
    def __init__(self, key: Union[str, List[str]] = "image", size: Union[int, List] = 256, interpolation: int = 2):
        inter_map = {
            0: Inter.NEAREST,
            1: Inter.ANTIALIAS,
            2: Inter.BILINEAR,
            3: Inter.BICUBIC,
            4: Inter.AREA,
            5: Inter.PILCUBIC,
        }
        if isinstance(interpolation, int):
            interpolation = inter_map.get(interpolation, Inter.BILINEAR)

        size = size if isinstance(size, int) else list(size)
        self.resize_op = ms.dataset.transforms.vision.Resize(size, interpolation)

        if isinstance(key, str):
            self.key = [key]
        else:
            self.key = key

    def __call__(self, sample: Dict):
        for k in self.key:
            sample[k] = self.resize_op(sample[k])
        return sample


class CenterCrop:
    def __init__(self, key: str = "image", size: Union[int, List] = 256):
        self.center_crop_op = ms.dataset.transforms.vision.CenterCrop(size)
        self.size = size
        self.key = key

    def __call__(self, sample: Dict):
        sample_key = sample[self.key]
        y = max(0, int((sample_key.shape[0] - self.size) / 2.0))  # crop in height
        x = max(0, int((sample_key.shape[1] - self.size) / 2.0))  # crop in weight

        sample[self.key] = self.center_crop_op(sample[self.key])
        sample["crop_coords_top_left"] = np.array([y, x], np.int32)
        return sample


class RandomHorizontalFlip:
    def __init__(self, key: str = "image", p: float = 0.5):
        self.random_flip_op = ms.dataset.transforms.vision.RandomHorizontalFlip(prob=p)
        self.key = key

    def __call__(self, sample: Dict):
        sample[self.key] = self.random_flip_op(sample[self.key])
        return sample


class Transpose:
    def __init__(self, key: Union[str, List[str]] = "image", type: str = "hwc2chw"):
        if isinstance(key, str):
            self.key = [key]
        else:
            self.key = key
        self.type = type

    def __call__(self, sample: Dict):
        for k in self.key:
            if self.type == "hwc2chw":
                sample[k] = np.transpose(sample[k], (2, 0, 1))
            elif self.type == "chw2hwc":
                sample[k] = np.transpose(sample[k], (1, 2, 0))
            else:
                raise NotImplementedError

        return sample


class MindDataImageTransforms:
    def __init__(
        self,
        transforms: Union[Union[Dict, DictConfig], ListConfig],
        key: str = "image",
        strict: bool = True,
    ):
        self.strict = strict
        self.key = key
        chained_transforms = []

        if isinstance(transforms, (DictConfig, Dict)):
            transforms = [transforms]

        for t in transforms:
            t = instantiate_from_config(t)
            chained_transforms.append(t)

        self.transform = ms.dataset.transforms.Compose(chained_transforms)

    def __call__(self, sample: Dict) -> Union[Dict, None]:
        if self.key not in sample:
            if self.strict:
                del sample
                return None
            else:
                return sample
        sample[self.key] = self.transform(sample[self.key])
        return sample


class AddOriginalImageSizeAsTupleAndCropToSquare:
    """
    Adds the original image size as params and crops to a square.
    Also adds cropping parameters. Requires that no RandomCrop/CenterCrop has been called before
    """

    def __init__(
        self,
        image_key: str = "image",
        use_data_key: bool = True,
        data_key: str = "json",
    ):
        self.image_key = image_key
        self.data_key = data_key
        self.use_data_key = use_data_key

    def __call__(self, x: Dict) -> Dict:
        jpg = x[self.image_key]  # (h, w, 3)
        if not isinstance(jpg, np.ndarray) or jpg.shape[2] != 3:
            raise ValueError(
                f"{self.__class__.__name__} requires input image to be a numpy.ndarray with channels-first"
            )
        # jpg should be chw tensor  in [-1, 1] at this point
        size = min(jpg.shape[0], jpg.shape[1])
        delta_h = jpg.shape[0] - size
        delta_w = jpg.shape[1] - size
        assert not all(
            [delta_h, delta_w]
        )  # we assume that the image is already resized such that the smallest size is at the desired size. Thus, eiter delta_h or delta_w must be zero
        top = np.random.randint(0, delta_h + 1)
        left = np.random.randint(0, delta_w + 1)
        crop_op = ms.dataset.transforms.vision.Crop((top, left), size)
        x[self.image_key] = crop_op(jpg)

        if "control" in x:
            x["control"] = crop_op(x["control"])

        x["crop_coords_top_left"] = np.array([top, left])
        return x
