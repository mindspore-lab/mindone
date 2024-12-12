import warnings
from dataclasses import asdict, dataclass
from random import randrange
from typing import Any, Dict, List, Optional, Tuple, Union

from lvdm.modules.encoders.clip import CLIPModel, parse, support_list
from lvdm.modules.encoders.openclip_tokenizer import tokenize
from utils.utils import freeze_params

import mindspore as ms
from mindspore import nn, ops
from mindspore.dataset import transforms, vision

# Image processing
CLIP_RESIZE = vision.Resize((224, 224), interpolation=vision.Inter.BICUBIC)
CENTER_CROP = vision.CenterCrop(224)

# Constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)

CLIP_NORMALIZE = vision.Normalize(mean=CLIP_MEAN, std=CLIP_STD)

ViCLIP_MEAN = [0.485, 0.456, 0.406]
ViCLIP_STD = [0.229, 0.224, 0.225]
ViCLIP_NORMALIZE = vision.Normalize(mean=ViCLIP_MEAN, std=ViCLIP_STD)


def load_clip_model(arch, pretrained_ckpt_path, dtype):
    """
    Load CLIP model.

    Args:
        arch (str): Model architecture.
        pretrained_ckpt_path (str): Path of the pretrained checkpoint.
    Returns:
        model (CLIPModel): CLIP model.
    """

    config_path = support_list[arch.lower()]
    config = parse(config_path, pretrained_ckpt_path)
    config.dtype = dtype
    model = CLIPModel(config)
    return model


def normalize(tensor, mean, std):
    tensor = ((tensor.T - mean) / std).T
    return tensor


def get_hpsv2_fn(precision="fp32", rm_ckpt_dir="HPS_v2_compressed.ckpt"):
    # Assert that precision is one of the allowed values
    assert precision in ["no", "bf16", "fp16", "fp32"], f"Invalid precision: {precision}"

    # Mapping of precision to data type
    dtype_mapping = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32", "no": "float32"}

    # Get the corresponding data type
    dtype = dtype_mapping[precision]

    model = load_clip_model(
        "open_clip_vit_h_14",
        pretrained_ckpt_path=rm_ckpt_dir,
        dtype=dtype,
    )

    preprocess_val = image_transform(
        model.config.vision_config.image_size,
        is_train=False,
        mean=None,
        std=None,
        resize_longest_max=True,
    )

    model.set_train(False)
    freeze_params(model)

    # gets vae decode as input
    def score_fn(image_inputs: ms.Tensor, text_inputs: ms.Tensor, return_logits=False):
        # Process pixels and multicrop
        for t in preprocess_val.transforms[2:]:
            image_inputs = ops.stack([t(img) for img in image_inputs])

        if isinstance(text_inputs[0], str):
            text_inputs = ms.Tensor(tokenize(text_inputs)[0], ms.int32)

        # embed
        image_features = model.encode_image(image_inputs, normalize=True)

        with ms._no_grad():
            text_features = model.encode_text(text_inputs, normalize=True)

        hps_score = (image_features * text_features).sum(-1)
        if return_logits:
            hps_score = hps_score * model.logit_scale.exp()
        return hps_score

    return score_fn


def get_img_reward_fn(precision="fp32"):
    # pip install image-reward
    import ImageReward as RM

    model = RM.load("ImageReward-v1.0")
    model.set_train(False)
    freeze_params(model)

    rm_preprocess = transforms.Compose(
        [
            vision.Resize(224, interpolation=vision.Inter.BICUBIC),
            vision.CenterCrop(224),
            CLIP_NORMALIZE,
        ]
    )

    # gets vae decode as input
    def score_fn(image_inputs: ms.Tensor, text_inputs: List[str], return_logits=False):
        del return_logits
        if precision == "fp16":
            model.to_float(ms.float16)

        image = rm_preprocess(image_inputs)
        text_input = model.blip.tokenizer(
            text_inputs,
            padding="max_length",
            truncation=True,
            max_length=35,
            return_tensors="pt",
        )
        rewards = model.score_gard(text_input.input_ids, text_input.attention_mask, image)
        return -ops.relu(-rewards + 2).squeeze(-1)

    return score_fn


class ResizeCropMinSize(nn.Cell):
    def __init__(self, min_size, interpolation=vision.Inter.BICUBIC, fill=0):
        super().__init__()
        if not isinstance(min_size, int):
            raise TypeError(f"Size should be int. Got {type(min_size)}")
        self.min_size = min_size
        self.interpolation = interpolation
        self.fill = fill
        # self.random_crop = vision.RandomCrop((min_size, min_size))

    def random_crop(self, img: ms.Tensor, crop_size):
        height, width = img.shape[-2:]

        h_max = height - crop_size[0]
        w_max = width - crop_size[1]

        random_h = randrange(0, h_max // 2 + 1) * 2
        random_w = randrange(0, w_max // 2 + 1) * 2

        new_img = img[:, :, random_h : random_h + crop_size[0], random_w : random_w + crop_size[1]]

        return new_img

    def construct(self, img):
        if isinstance(img, ms.Tensor):
            height, width = img.shape[-2:]
        else:
            width, height = img.size
        scale = self.min_size / float(min(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = ops.ResizeBicubic()(img, new_size)
            img = self.random_crop(img, (self.min_size, self.min_size))
        return img


@dataclass
class AugmentationCfg:
    scale: Tuple[float, float] = (0.9, 1.0)
    ratio: Optional[Tuple[float, float]] = None
    color_jitter: Optional[Union[float, Tuple[float, float, float]]] = None
    interpolation: Optional[str] = None
    re_prob: Optional[float] = None
    re_count: Optional[int] = None
    use_timm: bool = False


class ResizeMaxSize(nn.Cell):
    def __init__(self, max_size, interpolation=vision.Inter.BICUBIC, fn="max", fill=0):
        super().__init__()
        if not isinstance(max_size, int):
            raise TypeError(f"Size should be int. Got {type(max_size)}")
        self.max_size = max_size
        self.interpolation = interpolation
        self.fn = min if fn == "min" else min
        self.fill = fill

    def construct(self, img):
        if isinstance(img, ms.Tensor):
            height, width = img.shape[1:]
        else:
            width, height = img.size
        scale = self.max_size / float(max(height, width))
        if scale != 1.0:
            new_size = tuple(round(dim * scale) for dim in (height, width))
            img = ops.ResizeBicubic()(img.unsqueeze(0), new_size).squeeze(0)
            pad_h = self.max_size - new_size[0]
            pad_w = self.max_size - new_size[1]
            img = ops.pad(
                img, padding=[pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=self.fill
            )
        return img


class MaskAwareNormalize(nn.Cell):
    def __init__(self, mean, std):
        super().__init__()
        # self.normalize = vision.Normalize(mean=mean, std=std)
        self.mean = mean
        self.std = std

    def normalize(self, tensor):
        tensor = ((tensor.T - self.mean) / self.std).T
        return tensor

    def construct(self, tensor):
        if tensor.shape[0] == 4:
            return ops.cat([self.normalize(tensor[:3]), tensor[3:]], axis=0)
        else:
            return self.normalize(tensor)


def _convert_to_rgb_or_rgba(image):
    if image.mode == "RGBA":
        return image
    else:
        return image.convert("RGB")


def image_transform(
    image_size: int,
    is_train: bool,
    mean: Optional[Tuple[float, ...]] = None,
    std: Optional[Tuple[float, ...]] = None,
    resize_longest_max: bool = False,
    fill_color: int = 0,
    aug_cfg: Optional[Union[Dict[str, Any], AugmentationCfg]] = None,
):
    mean = mean or OPENAI_DATASET_MEAN
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3

    std = std or OPENAI_DATASET_STD
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3

    if isinstance(image_size, (list, tuple)) and image_size[0] == image_size[1]:
        # for square size, pass size as int so that Resize() uses aspect preserving shortest edge
        image_size = image_size[0]

    if isinstance(aug_cfg, dict):
        aug_cfg = AugmentationCfg(**aug_cfg)
    else:
        aug_cfg = aug_cfg or AugmentationCfg()
    normalize = MaskAwareNormalize(mean=mean, std=std)
    if is_train:
        aug_cfg_dict = {k: v for k, v in asdict(aug_cfg).items() if v is not None}
        use_timm = aug_cfg_dict.pop("use_timm", False)
        if use_timm:
            assert False, "not tested for augmentation with mask"
            from timm.data import create_transform  # timm can still be optional

            if isinstance(image_size, (tuple, list)):
                assert len(image_size) >= 2
                input_size = (3,) + image_size[-2:]
            else:
                input_size = (3, image_size, image_size)
            # by default, timm aug randomly alternates bicubic & bilinear for better robustness at inference time
            aug_cfg_dict.setdefault("interpolation", "random")
            aug_cfg_dict.setdefault("color_jitter", None)  # disable by default
            train_transform = create_transform(
                input_size=input_size,
                is_training=True,
                hflip=0.0,
                mean=mean,
                std=std,
                re_mode="pixel",
                **aug_cfg_dict,
            )
        else:
            train_transform = transforms.Compose(
                [
                    _convert_to_rgb_or_rgba,
                    vision.ToTensor(),
                    vision.RandomResizedCrop(
                        image_size,
                        scale=aug_cfg_dict.pop("scale"),
                        interpolation=vision.Inter.BICUBIC,
                    ),
                    normalize,
                ]
            )
            if aug_cfg_dict:
                warnings.warn(
                    f"Unused augmentation cfg items, specify `use_timm` to use ({list(aug_cfg_dict.keys())})."
                )
        return train_transform
    else:
        transformations = [
            _convert_to_rgb_or_rgba,
            vision.ToTensor(),
        ]
        if resize_longest_max:
            transformations.extend([ResizeMaxSize(image_size, fill=fill_color)])
        else:
            transformations.extend(
                [
                    vision.Resize(image_size, interpolation=vision.Inter.BICUBIC),
                    vision.CenterCrop(image_size),
                ]
            )
        transformations.extend(
            [
                normalize,
            ]
        )
        return transforms.Compose(transformations)


def get_vi_clip_score_fn(rm_ckpt_dir: str, precision="amp", n_frames=8):
    assert n_frames == 8
    from viclip import get_viclip

    model_dict = get_viclip("l", rm_ckpt_dir)
    vi_clip = model_dict["viclip"]
    vi_clip.set_train(False)
    freeze_params(vi_clip)
    if precision == "fp16":
        vi_clip.to(ms.float16)

    viclip_resize = ResizeCropMinSize(224)

    def score_fn(image_inputs: ms.Tensor, text_inputs: str):
        # Process pixels and multicrop
        b, t = image_inputs.shape[:2]
        image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])
        pixel_values = normalize(viclip_resize(image_inputs), mean=ViCLIP_MEAN, std=ViCLIP_STD)
        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])
        video_features = vi_clip.get_vid_feat_with_grad(pixel_values)

        with ms._no_grad():
            text_features = vi_clip.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        score = (video_features * text_features).sum(-1)
        return score

    return score_fn


def get_intern_vid2_score_fn(rm_ckpt_dir: str, precision="amp", n_frames=8):
    from intern_vid2.demo_config import Config, eval_dict_leaf
    from intern_vid2.demo_utils import setup_internvideo2

    config = Config.from_file("intern_vid2/configs/internvideo2_stage2_config.py")
    config = eval_dict_leaf(config)
    config["inputs"]["video_input"]["num_frames"] = n_frames
    config["inputs"]["video_input"]["num_frames_test"] = n_frames
    config["model"]["vision_encoder"]["num_frames"] = n_frames

    config["model"]["vision_encoder"]["pretrained"] = rm_ckpt_dir
    config["pretrained_path"] = rm_ckpt_dir

    dtype = {"no": ms.float32, "fp16": ms.float16, "fp32": ms.float32, "bf16": ms.bfloat16}[precision]
    vi_clip, tokenizer = setup_internvideo2(config, dtype)
    vi_clip.set_train(False)
    freeze_params(vi_clip)
    if precision == "fp16":
        vi_clip = vi_clip.to_float(ms.float16)

    viclip_resize = ResizeCropMinSize(224)

    def tokenize_fn(text_inputs: str):
        tokens = tokenizer(
            text_inputs,
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="np",
        )
        return tokens

    def score_fn(image_inputs: ms.Tensor, text_inputs):
        # Process pixels and multicrop
        b, t = image_inputs.shape[:2]
        image_inputs = image_inputs.view(b * t, *image_inputs.shape[2:])

        pixel_values = viclip_resize(image_inputs).transpose(0, 2, 3, 1)
        pixel_values = ((pixel_values - ViCLIP_MEAN) / ViCLIP_STD).transpose(0, 3, 1, 2)

        pixel_values = pixel_values.view(b, t, *pixel_values.shape[1:])
        video_features = vi_clip.get_vid_feat_with_grad(pixel_values)

        if not isinstance(text_inputs, str):
            text_inputs = text_inputs[0].astype(str)

        with ms._no_grad():
            text_inputs = tokenize_fn(text_inputs)
            text_inputs = {k: ms.Tensor(v) for k, v in text_inputs.items()}
            _, text_features = vi_clip.encode_text(text_inputs)
            text_features = vi_clip.text_proj(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        score = (video_features * text_features).sum(-1)
        return score

    return score_fn


def get_reward_fn(reward_fn_name: str, **kwargs):
    if reward_fn_name == "hpsv2":
        return get_hpsv2_fn(**kwargs)
    elif reward_fn_name == "img_reward":
        return get_img_reward_fn(**kwargs)
    elif reward_fn_name == "vi_clip":
        return get_vi_clip_score_fn(**kwargs)
    elif reward_fn_name == "vi_clip2":
        return get_intern_vid2_score_fn(**kwargs)
    else:
        raise ValueError("Invalid reward_fn_name")
