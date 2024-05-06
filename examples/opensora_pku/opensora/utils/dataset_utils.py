def create_video_transforms(
    size, crop_size, num_frames, interpolation="bicubic", backend="al", disable_flip=True, random_crop=False
):
    """
    pipeline: flip -> resize -> crop
    NOTE: we change interpolation to bicubic for its better precision and used in SD. TODO: check impact on performance
    Args:
        size: resize to this size
        crop_size: tuple or integer, crop to this size.
        num_frames: number of frames in the video.
        interpolation: interpolation method.
        backend: backend to use. Currently only support albumentations.
        disable_flip: disable flip.
        random_crop: crop randomly. If False, crop center.
    """
    if isinstance(crop_size, (tuple, list)):
        h, w = crop_size
    else:
        h, w = crop_size, crop_size

    if backend == "al":
        # expect rgb image in range 0-255, shape (h w c)
        import albumentations
        import cv2
        from albumentations import CenterCrop, HorizontalFlip, RandomCrop, SmallestMaxSize

        targets = {"image{}".format(i): "image" for i in range(num_frames)}
        mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        transforms_list = [
            SmallestMaxSize(max_size=size, interpolation=mapping[interpolation]),
            CenterCrop(h, w) if not random_crop else RandomCrop(h, w),
        ]
        if not disable_flip:
            transforms_list.insert(0, HorizontalFlip(p=0.5))
        pixel_transforms = albumentations.Compose(
            transforms_list,
            additional_targets=targets,
        )
    else:
        raise NotImplementedError

    return pixel_transforms


def create_image_transforms(
    size, crop_size, interpolation="bicubic", backend="al", random_crop=False, disable_flip=True
):
    if isinstance(crop_size, (tuple, list)):
        h, w = crop_size
    else:
        h, w = crop_size, crop_size

    if backend == "pt":
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        mapping = {"bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC}

        pixel_transforms = transforms.Compose(
            [
                transforms.Resize(size, interpolation=mapping[interpolation]),
                transforms.CenterCrop((h, w)) if not random_crop else transforms.RandomCrop((h, w)),
            ]
        )
    else:
        # expect rgb image in range 0-255, shape (h w c)
        import albumentations
        import cv2
        from albumentations import CenterCrop, HorizontalFlip, RandomCrop, SmallestMaxSize

        mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        transforms_list = [
            SmallestMaxSize(max_size=size, interpolation=mapping[interpolation]),
            CenterCrop(crop_size, crop_size) if not random_crop else RandomCrop(crop_size, crop_size),
        ]
        if not disable_flip:
            transforms_list.insert(0, HorizontalFlip(p=0.5))

        pixel_transforms = albumentations.Compose(transforms)

    return pixel_transforms
