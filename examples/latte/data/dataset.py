import albumentations
import cv2


def create_video_transforms(
    h, w, num_frames, interpolation="bicubic", backend="al", use_safer_augment=True, apply_same_transform=True
):
    """
    a common transform pipeline for video dataset: flip -> resize -> crop
    h, w : target resize height, weight
    NOTE: we change interpolation to bicubic for its better precision and used in SD. TODO: check impact on performance
    use_safer_augment: bool. If true and if backend is "al", it applies random horizontal flip augmentation.
    apply_same_transform: bool. If true and if backend is "al", it applies the same random augmentation to all frames.
    """
    if backend == "pt":
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        mapping = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC}

        pixel_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(h, interpolation=mapping[interpolation]),
                transforms.CenterCrop((h, w)),
            ]
        )
    elif backend == "al":
        # expect rgb image in range 0-255, shape (h w c)
        from albumentations import CenterCrop, HorizontalFlip, Resize, SmallestMaxSize

        # NOTE: to ensure augment all frames in a video in the same way.
        targets = {"image{}".format(i): "image" for i in range(num_frames)}
        if not apply_same_transform:
            targets = None
        mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        if use_safer_augment:
            pixel_transforms = albumentations.Compose(
                [
                    SmallestMaxSize(max_size=h, interpolation=mapping[interpolation]),
                    CenterCrop(h, w),
                ],
                additional_targets=targets,
            )
        else:
            # originally used in torch ad code, but not good for non-square video data
            # also conflict the learning of left-right camera movement
            pixel_transforms = albumentations.Compose(
                [
                    HorizontalFlip(p=0.5),
                    Resize(h, h, interpolation=mapping[interpolation]),
                    CenterCrop(h, w),
                ],
                additional_targets=targets,
            )

    elif backend == "ms":
        # TODO: MindData doesn't support batch transform. can NOT make sure all frames are flipped the same
        from mindspore.dataset import transforms, vision
        from mindspore.dataset.vision import Inter

        from .transforms import CenterCrop

        mapping = {"bilinear": Inter.BILINEAR, "bicubic": Inter.BICUBIC}
        pixel_transforms = transforms.Compose(
            [
                vision.RandomHorizontalFlip(),
                vision.Resize(h, interpolation=mapping[interpolation]),
                CenterCrop(h, w),
            ]
        )
    else:
        raise NotImplementedError

    return pixel_transforms


def get_dataset(dataset_name, config, device_num=1, rank_id=0, **kwargs):
    if dataset_name == "sky":
        from .sky_dataset import create_dataloader
    elif dataset_name == "csv":
        from .csv_dataset import create_dataloader
    dataset = create_dataloader(config, device_num=device_num, rank_id=rank_id, **kwargs)
    return dataset
