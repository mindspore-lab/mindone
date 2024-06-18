import math

import decord

from mindspore import ops


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


IMG_EXTENSIONS = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class DecordInit(object):
    """Using Decord(https://github.com/dmlc/decord) to initialize the video_reader."""

    def __init__(self, num_threads=1):
        self.num_threads = num_threads
        self.ctx = decord.cpu(0)

    def __call__(self, filename):
        """Perform the Decord initialization.
        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        reader = decord.VideoReader(filename, ctx=self.ctx, num_threads=self.num_threads)
        return reader

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(" f"sr={self.sr}," f"num_threads={self.num_threads})"
        return repr_str


def pad_to_multiple(number, ds_stride):
    remainder = number % ds_stride
    if remainder == 0:
        return number
    else:
        padding = ds_stride - remainder
        return number + padding


class Collate:
    def __init__(self, args):
        self.max_image_size = args.max_image_size
        self.ae_stride = args.ae_stride
        self.ae_stride_t = args.ae_stride_t
        self.ae_stride_thw = (self.ae_stride_t, self.ae_stride, self.ae_stride)
        self.ae_stride_1hw = (1, self.ae_stride, self.ae_stride)

        self.patch_size = args.patch_size
        self.patch_size_t = args.patch_size_t
        self.patch_size_thw = (self.patch_size_t, self.patch_size, self.patch_size)
        self.patch_size_1hw = (1, self.patch_size, self.patch_size)

        self.num_frames = args.num_frames
        self.use_image_num = args.use_image_num
        self.max_thw = (self.num_frames, self.max_image_size, self.max_image_size)
        self.max_1hw = (1, self.max_image_size, self.max_image_size)

    def package(self, batch):
        batch_tubes_vid, input_ids_vid, cond_mask_vid = None, None, None
        batch_tubes_img, input_ids_img, cond_mask_img = None, None, None
        # import ipdb;ipdb.set_trace()
        if self.num_frames > 1:
            batch_tubes_vid = [i["video_data"]["video"] for i in batch]  # b [c t h w]
            input_ids_vid = ops.stack([i["video_data"]["input_ids"] for i in batch])  # b 1 l
            cond_mask_vid = ops.stack([i["video_data"]["cond_mask"] for i in batch])  # b 1 l
        if self.num_frames == 1 or self.use_image_num != 0:
            batch_tubes_img = [j for i in batch for j in i["image_data"]["image"]]  # b*num_img [c 1 h w]
            input_ids_img = ops.stack([i["image_data"]["input_ids"] for i in batch])  # b image_num l
            cond_mask_img = ops.stack([i["image_data"]["cond_mask"] for i in batch])  # b image_num l
        return batch_tubes_vid, input_ids_vid, cond_mask_vid, batch_tubes_img, input_ids_img, cond_mask_img

    def __call__(self, batch):
        batch_tubes_vid, input_ids_vid, cond_mask_vid, batch_tubes_img, input_ids_img, cond_mask_img = self.package(
            batch
        )

        ds_stride = self.ae_stride * self.patch_size
        t_ds_stride = self.ae_stride_t * self.patch_size_t
        if self.num_frames > 1 and self.use_image_num == 0:
            pad_batch_tubes, attention_mask = self.process(
                batch_tubes_vid,
                t_ds_stride,
                ds_stride,
                self.max_thw,
                self.ae_stride_thw,
                self.patch_size_thw,
                extra_1=True,
            )
            # attention_mask: b t h w
            # input_ids, cond_mask = input_ids_vid.squeeze(1), cond_mask_vid.squeeze(1)  # b 1 l -> b l
            input_ids, cond_mask = input_ids_vid, cond_mask_vid  # b 1 l
        elif self.num_frames > 1 and self.use_image_num != 0:
            pad_batch_tubes_vid, attention_mask_vid = self.process(
                batch_tubes_vid,
                t_ds_stride,
                ds_stride,
                self.max_thw,
                self.ae_stride_thw,
                self.patch_size_thw,
                extra_1=True,
            )
            # attention_mask_vid: b t h w

            pad_batch_tubes_img, attention_mask_img = self.process(
                batch_tubes_img, 1, ds_stride, self.max_1hw, self.ae_stride_1hw, self.patch_size_1hw, extra_1=False
            )
            # (b i) c 1 h w -> b c i h w
            pad_batch_tubes_img = pad_batch_tubes_img.sqeeuze(2)
            _, c, h, w = pad_batch_tubes_img.shape
            pad_batch_tubes_img = pad_batch_tubes_img.reshape(-1, self.use_image_num, c, h, w).permute(0, 2, 1, 3, 4)
            # (b i) 1 h w -> b i h w
            attention_mask_img = attention_mask_img.squeeze(2)
            _, h, w = attention_mask_img.shape
            attention_mask_img = attention_mask_img.reshape(-1, self.use_image_num, h, w)
            pad_batch_tubes = ops.cat(
                [pad_batch_tubes_vid, pad_batch_tubes_img], axis=2
            )  # concat at temporal, video first
            # attention_mask_img: b num_img h w
            attention_mask = ops.cat([attention_mask_vid, attention_mask_img], axis=1)  # b t+num_img h w
            input_ids = ops.cat([input_ids_vid, input_ids_img], axis=1)  # b 1+num_img hw
            cond_mask = ops.cat([cond_mask_vid, cond_mask_img], axis=1)  # b 1+num_img hw
        else:
            # import ipdb;ipdb.set_trace()
            pad_batch_tubes_img, attention_mask_img = self.process(
                batch_tubes_img, 1, ds_stride, self.max_1hw, self.ae_stride_1hw, self.patch_size_1hw, extra_1=False
            )
            # pad_batch_tubes = rearrange(pad_batch_tubes_img, '(b i) c 1 h w -> b c i h w', i=1)
            # attention_mask = rearrange(attention_mask_img, '(b i) 1 h w -> b i h w', i=1)
            pad_batch_tubes = pad_batch_tubes_img
            attention_mask = attention_mask_img
            input_ids, cond_mask = input_ids_img, cond_mask_img  # b 1 l
        return pad_batch_tubes, attention_mask, input_ids, cond_mask

    def process(self, batch_tubes, t_ds_stride, ds_stride, max_thw, ae_stride_thw, patch_size_thw, extra_1):
        # import ipdb;ipdb.set_trace()
        # pad to max multiple of ds_stride
        batch_input_size = [i.shape for i in batch_tubes]  # [(c t h w), (c t h w)]
        max_t, max_h, max_w = max_thw
        pad_max_t, pad_max_h, pad_max_w = (
            pad_to_multiple(max_t - 1 if extra_1 else max_t, t_ds_stride),
            pad_to_multiple(max_h, ds_stride),
            pad_to_multiple(max_w, ds_stride),
        )
        pad_max_t = pad_max_t + 1 if extra_1 else pad_max_t
        each_pad_t_h_w = [[pad_max_t - i.shape[1], pad_max_h - i.shape[2], pad_max_w - i.shape[3]] for i in batch_tubes]
        pad_batch_tubes = [
            ops.pad(im, (0, pad_w, 0, pad_h, 0, pad_t), value=0)
            for (pad_t, pad_h, pad_w), im in zip(each_pad_t_h_w, batch_tubes)
        ]
        pad_batch_tubes = ops.stack(pad_batch_tubes, axis=0)

        # make attention_mask
        # first_channel_first_frame, first_channel_other_frame = pad_batch_tubes[:, :1, :1], pad_batch_tubes[:, :1, 1:]  # first channel to make attention_mask
        # attention_mask_first_frame = F.max_pool3d(first_channel_first_frame, kernel_size=(1, *ae_stride_thw[1:]), stride=(1, *ae_stride_thw[1:]))
        # if first_channel_other_frame.numel() != 0:
        #     attention_mask_other_frame = F.max_pool3d(first_channel_other_frame, kernel_size=ae_stride_thw, stride=ae_stride_thw)
        #     attention_mask = ops.cat([attention_mask_first_frame, attention_mask_other_frame], axis=2)
        # else:
        #     attention_mask = attention_mask_first_frame
        # attention_mask = attention_mask[:, 0].bool().float()  # b t h w, do not channel

        max_tube_size = [pad_max_t, pad_max_h, pad_max_w]
        max_latent_size = [
            ((max_tube_size[0] - 1) // ae_stride_thw[0] + 1) if extra_1 else (max_tube_size[0] // ae_stride_thw[0]),
            max_tube_size[1] // ae_stride_thw[1],
            max_tube_size[2] // ae_stride_thw[2],
        ]
        valid_latent_size = [
            [
                int(math.ceil((i[1] - 1) / ae_stride_thw[0])) + 1
                if extra_1
                else int(math.ceil(i[1] / ae_stride_thw[0])),
                int(math.ceil(i[2] / ae_stride_thw[1])),
                int(math.ceil(i[3] / ae_stride_thw[2])),
            ]
            for i in batch_input_size
        ]
        attention_mask = [
            ops.pad(
                ops.ones(i),
                (0, max_latent_size[2] - i[2], 0, max_latent_size[1] - i[1], 0, max_latent_size[0] - i[0]),
                value=0,
            )
            for i in valid_latent_size
        ]
        attention_mask = ops.stack(attention_mask)  # b t h w

        return pad_batch_tubes, attention_mask
