from functools import partial

import cv2
from albumentations import Compose, Lambda, Resize, ToFloat
from opensora.models.causalvideovae import ae_norm
from transformers import AutoTokenizer

from .t2v_datasets import T2V_dataset
from .transform import TemporalRandomCrop, center_crop_th_tw, maxhxw_resize, spatial_stride_crop_video


def getdataset(args, dataset_file):
    temporal_sample = TemporalRandomCrop(args.num_frames)  # 16 x
    norm_fun = ae_norm[args.ae]

    def norm_func_albumentation(image, **kwargs):
        return norm_fun(image)

    mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
    targets = {"image{}".format(i): "image" for i in range(args.num_frames)}

    if args.force_resolution:
        assert (args.max_height is not None) and (
            args.max_width is not None
        ), "set max_height and max_width for fixed resolution"
        resize = [
            Lambda(
                name="crop_centercrop",
                image=partial(center_crop_th_tw, th=args.max_height, tw=args.max_width, top_crop=False),
                p=1.0,
            ),
            Resize(args.max_height, args.max_width, interpolation=mapping["bilinear"]),
        ]
    else:  # dynamic resolution
        assert args.max_hxw is not None, "set max_hxw for dynamic resolution"
        resize = [
            Lambda(
                name="maxhxw_resize",
                image=partial(maxhxw_resize, max_hxw=args.max_hxw, interpolation_mode=mapping["bilinear"]),
                p=1.0,
            ),
            Lambda(
                name="spatial_stride_crop",
                image=partial(spatial_stride_crop_video, stride=args.hw_stride),  # default stride=32
                p=1.0,
            ),
        ]

    transform = Compose(
        [*resize, ToFloat(255.0), Lambda(name="ae_norm", image=norm_func_albumentation, p=1.0)],
        additional_targets=targets,
    )

    tokenizer_1 = AutoTokenizer.from_pretrained(args.text_encoder_name_1, cache_dir=args.cache_dir)
    tokenizer_2 = None
    if args.text_encoder_name_2 is not None:
        tokenizer_2 = AutoTokenizer.from_pretrained(args.text_encoder_name_2, cache_dir=args.cache_dir)

    if args.dataset == "t2v":
        return T2V_dataset(
            args,
            transform=transform,
            temporal_sample=temporal_sample,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            return_text_emb=args.text_embed_cache,
        )
    elif args.dataset == "inpaint" or args.dataset == "i2v":
        raise NotImplementedError

    raise NotImplementedError(args.dataset)
