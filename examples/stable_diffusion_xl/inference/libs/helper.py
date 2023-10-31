import logging
import os
import warnings

import numpy as np
import PIL
from gm.helpers import load_model_from_config as load_model
from libs.util import set_random_seed
from PIL import Image

import mindspore as ms
from mindspore import ops

logger = logging.getLogger()

PIL_INTERPOLATION = {
    "linear": PIL.Image.LINEAR,
    "bilinear": PIL.Image.BILINEAR,
    "bicubic": PIL.Image.BICUBIC,
    "lanczos": PIL.Image.LANCZOS,
    "nearest": PIL.Image.NEAREST,
}


def load_model_from_config(config, ckpt=None, freeze=False, load_filter=False, amp_level="O0"):
    model = load_model(config, ckpt, amp_level=amp_level)
    if freeze:
        model.set_train(False)
        model.set_grad(False)
        for _, p in model.parameters_and_names():
            p.requires_grad = False
    if load_filter:
        raise NotImplementedError
    return model


def set_env(args):
    work_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"WORK DIR:{work_dir}")
    os.makedirs(args.output_path, exist_ok=True)
    outpath = args.output_path
    args.sample_path = os.path.join(outpath, "samples")
    os.makedirs(args.sample_path, exist_ok=True)
    args.base_count = len(os.listdir(args.sample_path))

    # set ms context
    device_id = int(os.getenv("DEVICE_ID", 0))
    ms.context.set_context(mode=args.ms_mode, device_id=device_id)

    set_random_seed(args.seed)


class VaeImageProcessor:
    def __init__(
        self,
        do_resize=True,
        vae_scale_factor=8,
        resample="lanczos",
        do_normalize=True,
        do_convert_rgb=False,
    ):
        super(VaeImageProcessor, self).__init__()
        self.do_resize = do_resize
        self.vae_scale_factor = vae_scale_factor
        self.resample = resample
        self.do_normalize = do_normalize
        self.do_convert_rgb = do_convert_rgb

    @staticmethod
    def numpy_to_pil(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    @staticmethod
    def pil_to_numpy(images):
        """
        Convert a PIL image or a list of PIL images to NumPy arrays.
        """
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    @staticmethod
    def numpy_to_ms(images):
        """
        Convert a NumPy image to a MindSpore tensor.
        """
        if images.ndim == 3:
            images = images[..., None]

        images = ms.Tensor(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def ms_to_numpy(images):
        """
        Convert a MindSpore tensor to a NumPy image.
        """
        images = images.asnumpy().transpose(0, 2, 3, 1)
        return images

    @staticmethod
    def normalize(images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @staticmethod
    def denormalize(images):
        """
        Denormalize an image array to [0,1].
        """
        return (images / 2 + 0.5).clamp(0, 1)

    @staticmethod
    def convert_to_rgb(image):
        """
        Converts an image to RGB format.
        """
        image = image.convert("RGB")
        return image

    def resize(self, image, height, width, resample=None):
        """
        Resize a PIL image. Both height and width are downscaled to the next integer multiple of `vae_scale_factor`.
        """
        if height is None:
            height = image.height
        if width is None:
            width = image.width

        width, height = (
            x - x % self.vae_scale_factor for x in (width, height)
        )  # resize to integer multiple of vae_scale_factor
        resample = resample if resample is not None else self.resample
        image = image.resize((width, height), resample=PIL_INTERPOLATION[resample])
        return image

    def preprocess(self, image, height, width):
        """
        Preprocess the image input. Accemsed formats are PIL images, NumPy arrays or MindSpore tensors.
        """
        supported_formats = (PIL.Image.Image, np.ndarray, ms.Tensor)
        if isinstance(image, supported_formats):
            image = [image]
        elif not (isinstance(image, list) and all(isinstance(i, supported_formats) for i in image)):
            raise ValueError(
                f"Input is in incorrect format: {[type(i) for i in image]}. Currently, we only support {', '.join(supported_formats)}"
            )

        if isinstance(image[0], PIL.Image.Image):
            if self.do_convert_rgb:
                image = [self.convert_to_rgb(i) for i in image]
            if self.do_resize:
                image = [self.resize(i, height, width) for i in image]
            image = self.pil_to_numpy(image)  # to np
            image = self.numpy_to_ms(image)  # to ms

        elif isinstance(image[0], np.ndarray):
            image = np.concatenate(image, axis=0) if image[0].ndim == 4 else np.stack(image, axis=0)
            image = self.numpy_to_ms(image)
            _, _, height, width = image.shape
            if self.do_resize and (height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0):
                raise ValueError(
                    f"Currently we only support resizing for PIL image - please resize your numpy array to be divisible by {self.vae_scale_factor}"
                    f"currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize omsion in VAEImageProcessor"
                )

        elif isinstance(image[0], ms.Tensor):
            image = ops.concat(image, axis=0) if image[0].ndim == 4 else ops.stack(image, axis=0)
            _, channel, height, width = image.shape

            # don't need any preprocess if the image is latents
            if channel == 4:
                return image

            if self.do_resize and (height % self.vae_scale_factor != 0 or width % self.vae_scale_factor != 0):
                raise ValueError(
                    f"Currently we only support resizing for PIL image - please resize your MindSpore tensor to be divisible by {self.vae_scale_factor}"
                    f"currently the sizes are {height} and {width}. You can also pass a PIL image instead to use resize omsion in VAEImageProcessor"
                )

        # expected range [0,1], normalize to [-1,1]
        do_normalize = self.do_normalize
        if image.min() < 0:
            warnings.warn(
                "Passing `image` as torch tensor with value range in [-1,1] is deprecated. The expected value range for image tensor is [0,1] "
                f"when passing as MindSpore tensor or numpy Array. You passed `image` with value range [{image.min()},{image.max()}]",
                FutureWarning,
            )
            do_normalize = False

        if do_normalize:
            image = self.normalize(image)

        return image

    def postprocess(self, image, output_type="pil"):
        if output_type not in ["latent", "ms", "np", "pil"]:
            output_type = "np"

        if output_type == "latent":
            return image

        if output_type == "ms":
            return image

        if isinstance(image, ms.Tensor):
            image = self.ms_to_numpy(image)
        elif isinstance(image, np.ndarray):
            image = image.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"Not support the image type: {type(image)}")

        if output_type == "np":
            return image

        if output_type == "pil":
            return self.numpy_to_pil(image)


def exists(x):
    return x is not None
