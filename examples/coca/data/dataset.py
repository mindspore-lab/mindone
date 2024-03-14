from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import CenterCrop, Inter, Normalize, Resize, ToTensor

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)

def _convert_to_rgb(image):
    return image.convert('RGB')

def create_transforms(image_size=224, interpolation="bicubic"):
    mapping = {"bilinear": Inter.BILINEAR, "bicubic": Inter.BICUBIC}
    if not isinstance(image_size, (tuple, list)):
        image_size = (image_size, image_size)
        transforms = Compose(
            [
                Resize(image_size[0], interpolation=mapping[interpolation]),
                CenterCrop(image_size),
                _convert_to_rgb,
                ToTensor(),
                Normalize(mean=MEAN, std=STD, is_hwc=False)
            ]
        )
    return  transforms