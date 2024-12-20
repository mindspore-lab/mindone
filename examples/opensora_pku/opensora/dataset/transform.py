import html
import random
import re
import urllib.parse as ul

import albumentations
import ftfy
import numpy as np
from bs4 import BeautifulSoup

__all__ = ["create_video_transforms", "t5_text_preprocessing"]

# video (pixel) transform


def create_video_transforms(h, w, num_frames, interpolation="bicubic", backend="al", disable_flip=True):
    """
    pipeline: flip -> resize -> crop
    h, w : target resize height, weight
    NOTE: we change interpolation to bicubic for its better precision and used in SD. TODO: check impact on performance
    """
    if backend == "al":
        # expect rgb image in range 0-255, shape (h w c)
        import cv2
        from albumentations import CenterCrop, HorizontalFlip, SmallestMaxSize

        targets = {"image{}".format(i): "image" for i in range(num_frames)}
        mapping = {"bilinear": cv2.INTER_LINEAR, "bicubic": cv2.INTER_CUBIC}
        max_size = max(h, w)
        if disable_flip:
            # flip is not proper for horizontal motion learning
            pixel_transforms = albumentations.Compose(
                [
                    SmallestMaxSize(max_size=max_size, interpolation=mapping[interpolation]),
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
                    SmallestMaxSize(max_size=max_size, interpolation=mapping[interpolation]),
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


def crop(image, i, j, h, w):
    if len(image.shape) != 3:
        raise ValueError("image should be a 3D tensor")
    return image[i : i + h, j : j + w, ...]


def center_crop_th_tw(image, th, tw, top_crop, **kwargs):
    # input is a 3-d arrary (H, W, C)

    h, w = image.shape[0], image.shape[1]
    tr = th / tw
    if h / w > tr:
        new_h = int(w * tr)
        new_w = w
    else:
        new_h = h
        new_w = int(h / tr)

    i = 0 if top_crop else int(round((h - new_h) / 2.0))
    j = int(round((w - new_w) / 2.0))
    cropped_image = crop(image, i, j, new_h, new_w)
    return cropped_image


def resize(image, h, w, interpolation_mode):
    resize_func = albumentations.Resize(h, w, interpolation=interpolation_mode)

    return resize_func(image=image)["image"]


def get_params(h, w, stride):
    th, tw = h // stride * stride, w // stride * stride

    i = (h - th) // 2
    j = (w - tw) // 2

    return i, j, th, tw


def spatial_stride_crop_video(image, stride, **kwargs):
    """
    Args:
        image (numpy array): Video clip to be cropped. Size is (H, W, C)
    Returns:
        numpy array: cropped video clip by stride.
            size is (OH, OW, C)
    """
    h, w = image.shape[:2]
    i, j, h, w = get_params(h, w, stride)
    return crop(image, i, j, h, w)


def maxhxw_resize(image, max_hxw, interpolation_mode, **kwargs):
    """
        First use the h*w,
        then resize to the specified size
    Args:
        image (numpy array): Video clip to be cropped. Size is (H, W, C)
    Returns:
        numpy array: scale resized video clip.
    """
    h, w = image.shape[:2]
    if h * w > max_hxw:
        scale_factor = np.sqrt(max_hxw / (h * w))
        tr_h = int(h * scale_factor)
        tr_w = int(w * scale_factor)
    else:
        tr_h = h
        tr_w = w
    if h == tr_h and w == tr_w:
        return image
    resize_image = resize(image, tr_h, tr_w, interpolation_mode)
    return resize_image


# create text transform(preprocess)
bad_punct_regex = re.compile(
    r"[" + "#®•©™&@·º½¾¿¡§~" + r"\)" + r"\(" + r"\]" + r"\[" + r"\}" + r"\{" + r"\|" + r"\\" + r"\/" + r"\*" + r"]{1,}"
)  # noqa


def t5_text_preprocessing(text):
    # The exact text cleaning as was in the training stage:
    text = clean_caption(text)
    text = clean_caption(text)
    return text


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def clean_caption(caption):
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub("<person>", "person", caption)
    # urls:
    caption = re.sub(
        r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    caption = re.sub(
        r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",  # noqa
        "",
        caption,
    )  # regex for urls
    # html:
    caption = BeautifulSoup(caption, features="html.parser").text

    # @<nickname>
    caption = re.sub(r"@[\w\d]+\b", "", caption)

    # 31C0—31EF CJK Strokes
    # 31F0—31FF Katakana Phonetic Extensions
    # 3200—32FF Enclosed CJK Letters and Months
    # 3300—33FF CJK Compatibility
    # 3400—4DBF CJK Unified Ideographs Extension A
    # 4DC0—4DFF Yijing Hexagram Symbols
    # 4E00—9FFF CJK Unified Ideographs
    caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
    caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
    caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
    caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
    caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
    caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
    caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
    #######################################################

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",  # noqa
        "-",
        caption,
    )

    # кавычки к одному стандарту
    caption = re.sub(r"[`´«»“”¨]", '"', caption)
    caption = re.sub(r"[‘’]", "'", caption)

    # &quot;
    caption = re.sub(r"&quot;?", "", caption)
    # &amp
    caption = re.sub(r"&amp", "", caption)

    # ip adresses:
    caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

    # article ids:
    caption = re.sub(r"\d:\d\d\s+$", "", caption)

    # \n
    caption = re.sub(r"\\n", " ", caption)

    # "#123"
    caption = re.sub(r"#\d{1,3}\b", "", caption)
    # "#12345.."
    caption = re.sub(r"#\d{5,}\b", "", caption)
    # "123456.."
    caption = re.sub(r"\b\d{6,}\b", "", caption)
    # filenames:
    caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

    #
    caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

    caption = re.sub(bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

    # this-is-my-cute-cat / this_is_my_cute_cat
    regex2 = re.compile(r"(?:\-|\_)")
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, " ", caption)

    caption = basic_clean(caption)

    caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
    caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
    caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

    caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
    caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
    caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
    caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
    caption = re.sub(r"\bpage\s+\d+\b", "", caption)

    caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

    caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

    caption = re.sub(r"\b\s+\:\s+", r": ", caption)
    caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
    caption = re.sub(r"\s+", " ", caption)

    caption.strip()

    caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
    caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
    caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
    caption = re.sub(r"^\.\S+$", "", caption)

    return caption.strip()


#  ------------------------------------------------------------
#  ---------------------  Sampling  ---------------------------
#  ------------------------------------------------------------
class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    Args:
        size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index


class DynamicSampleDuration(object):
    """Temporally crop the given frame indices at a random location.

    Args:
        size (int): Desired length of frames will be seen in the model.
    """

    def __init__(self, t_stride, extra_1):
        self.t_stride = t_stride
        self.extra_1 = extra_1

    def __call__(self, t, h, w):
        if self.extra_1:
            t = t - 1
        truncate_t_list = list(range(t + 1))[t // 2 :][:: self.t_stride]  # need half at least
        truncate_t = random.choice(truncate_t_list)
        if self.extra_1:
            truncate_t = truncate_t + 1
        return 0, truncate_t


keywords = [
    " man ",
    " woman ",
    " person ",
    " people ",
    "human",
    " individual ",
    " child ",
    " kid ",
    " girl ",
    " boy ",
]
keywords += [i[:-1] + "s " for i in keywords]

masking_notices = [
    "Note: The faces in this image are blurred.",
    "This image contains faces that have been pixelated.",
    "Notice: Faces in this image are masked.",
    "Please be aware that the faces in this image are obscured.",
    "The faces in this image are hidden.",
    "This is an image with blurred faces.",
    "The faces in this image have been processed.",
    "Attention: Faces in this image are not visible.",
    "The faces in this image are partially blurred.",
    "This image has masked faces.",
    "Notice: The faces in this picture have been altered.",
    "This is a picture with obscured faces.",
    "The faces in this image are pixelated.",
    "Please note, the faces in this image have been blurred.",
    "The faces in this photo are hidden.",
    "The faces in this picture have been masked.",
    "Note: The faces in this picture are altered.",
    "This is an image where faces are not clear.",
    "Faces in this image have been obscured.",
    "This picture contains masked faces.",
    "The faces in this image are processed.",
    "The faces in this picture are not visible.",
    "Please be aware, the faces in this photo are pixelated.",
    "The faces in this picture have been blurred.",
]

webvid_watermark_notices = [
    "This video has a faint Shutterstock watermark in the center.",
    "There is a slight Shutterstock watermark in the middle of this video.",
    "The video contains a subtle Shutterstock watermark in the center.",
    "This video features a light Shutterstock watermark at its center.",
    "A faint Shutterstock watermark is present in the middle of this video.",
    "There is a mild Shutterstock watermark at the center of this video.",
    "This video has a slight Shutterstock watermark in the middle.",
    "You can see a faint Shutterstock watermark in the center of this video.",
    "A subtle Shutterstock watermark appears in the middle of this video.",
    "This video includes a light Shutterstock watermark at its center.",
]


high_aesthetic_score_notices_video = [
    "This video has a high aesthetic quality.",
    "The beauty of this video is exceptional.",
    "This video scores high in aesthetic value.",
    "With its harmonious colors and balanced composition.",
    "This video ranks highly for aesthetic quality",
    "The artistic quality of this video is excellent.",
    "This video is rated high for beauty.",
    "The aesthetic quality of this video is impressive.",
    "This video has a top aesthetic score.",
    "The visual appeal of this video is outstanding.",
]

low_aesthetic_score_notices_video = [
    "This video has a low aesthetic quality.",
    "The beauty of this video is minimal.",
    "This video scores low in aesthetic appeal.",
    "The aesthetic quality of this video is below average.",
    "This video ranks low for beauty.",
    "The artistic quality of this video is lacking.",
    "This video has a low score for aesthetic value.",
    "The visual appeal of this video is low.",
    "This video is rated low for beauty.",
    "The aesthetic quality of this video is poor.",
]


high_aesthetic_score_notices_image = [
    "This image has a high aesthetic quality.",
    "The beauty of this image is exceptional",
    "This photo scores high in aesthetic value.",
    "With its harmonious colors and balanced composition.",
    "This image ranks highly for aesthetic quality.",
    "The artistic quality of this photo is excellent.",
    "This image is rated high for beauty.",
    "The aesthetic quality of this image is impressive.",
    "This photo has a top aesthetic score.",
    "The visual appeal of this image is outstanding.",
]

low_aesthetic_score_notices_image = [
    "This image has a low aesthetic quality.",
    "The beauty of this image is minimal.",
    "This image scores low in aesthetic appeal.",
    "The aesthetic quality of this image is below average.",
    "This image ranks low for beauty.",
    "The artistic quality of this image is lacking.",
    "This image has a low score for aesthetic value.",
    "The visual appeal of this image is low.",
    "This image is rated low for beauty.",
    "The aesthetic quality of this image is poor.",
]

high_aesthetic_score_notices_image_human = [
    "High-quality image with visible human features and high aesthetic score.",
    "Clear depiction of an individual in a high-quality image with top aesthetics.",
    "High-resolution photo showcasing visible human details and high beauty rating.",
    "Detailed, high-quality image with well-defined human subject and strong aesthetic appeal.",
    "Sharp, high-quality portrait with clear human features and high aesthetic value.",
    "High-quality image featuring a well-defined human presence and exceptional aesthetics.",
    "Visible human details in a high-resolution photo with a high aesthetic score.",
    "Clear, high-quality image with prominent human subject and superior aesthetic rating.",
    "High-quality photo capturing a visible human with excellent aesthetics.",
    "Detailed, high-quality image of a human with high visual appeal and aesthetic value.",
]


def calculate_statistics(data):
    if len(data) == 0:
        return None
    data = np.array(data)
    mean = np.mean(data)
    variance = np.var(data)
    std_dev = np.std(data)
    minimum = np.min(data)
    maximum = np.max(data)

    return {"mean": mean, "variance": variance, "std_dev": std_dev, "min": minimum, "max": maximum}


def maxhwresize(ori_height, ori_width, max_hxw):
    if ori_height * ori_width > max_hxw:
        scale_factor = np.sqrt(max_hxw / (ori_height * ori_width))
        new_height = int(ori_height * scale_factor)
        new_width = int(ori_width * scale_factor)
    else:
        new_height = ori_height
        new_width = ori_width
    return new_height, new_width


def add_aesthetic_notice_video(caption, aesthetic_score):
    if aesthetic_score <= 4.25:
        notice = random.choice(low_aesthetic_score_notices_video)
        return random.choice([caption + " " + notice, notice + " " + caption])
    if aesthetic_score >= 5.75:
        notice = random.choice(high_aesthetic_score_notices_video)
        return random.choice([caption + " " + notice, notice + " " + caption])
    return caption


def add_aesthetic_notice_image(caption, aesthetic_score):
    if aesthetic_score <= 4.25:
        notice = random.choice(low_aesthetic_score_notices_image)
        return random.choice([caption + " " + notice, notice + " " + caption])
    if aesthetic_score >= 5.75:
        notice = random.choice(high_aesthetic_score_notices_image)
        return random.choice([caption + " " + notice, notice + " " + caption])
    return caption


def add_high_aesthetic_notice_image(caption):
    notice = random.choice(high_aesthetic_score_notices_image)
    return random.choice([caption + " " + notice, notice + " " + caption])


def add_high_aesthetic_notice_image_human(caption):
    notice = random.choice(high_aesthetic_score_notices_image_human)
    return random.choice([caption + " " + notice, notice + " " + caption])
