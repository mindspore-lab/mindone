import os
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from transformers.utils import is_safetensors_available

import mindspore
import mindspore as ms
from mindspore.nn.utils import no_init_parameters

from .annotator.canny import CannyDetector
from .controlnet import ControlNetFlux
from .model import Flux, FluxParams
from .modules.autoencoder import AutoEncoder, AutoEncoderParams
from .modules.conditioner import HFEmbedder


def load_safetensors(path):
    if path.endswith(".safetensors") and is_safetensors_available():
        # Check format of the archive
        with safe_open(path, framework="np") as f:
            metadata = f.metadata()
        if metadata is not None:
            format = metadata.get("format", None)
            if format is not None and format not in ["pt", "tf", "flax", "np"]:
                raise OSError(
                    f"The safetensors archive passed at {path} does not contain the valid metadata. Make sure "
                    "you save your model with the `save_pretrained` method."
                )
        return ms.load_checkpoint(path, format="safetensors")


def get_lora_rank(checkpoint):
    for k in checkpoint.keys():
        if k.endswith(".down.weight"):
            return checkpoint[k].shape[0]


def load_checkpoint(local_path, repo_id, name):
    if local_path is not None:
        if ".safetensors" in local_path:
            print(f"Loading .safetensors checkpoint from {local_path}")
            checkpoint = load_safetensors(local_path)
        else:
            print(f"Loading checkpoint from {local_path}")
            checkpoint = ms.load_checkpoint(local_path)
    elif repo_id is not None and name is not None:
        print(f"Loading checkpoint {name} from repo id {repo_id}")
        checkpoint = load_from_repo_id(repo_id, name)
    else:
        raise ValueError("LOADING ERROR: you must specify local_path or repo_id with name in HF to download")
    return checkpoint


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()


# Added upscale_method, mode params
def resize_image_with_pad(input_image, resolution, skip_hwc3=False, mode="edge"):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    if resolution == 0:
        return img, lambda x: x
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad


class Annotator:
    def __init__(self, name: str):
        if name == "canny":
            processor = CannyDetector()
        else:
            raise ValueError(f"Invalid annotator name: {name}")
        self.name = name
        self.processor = processor

    def __call__(self, image: Image, width: int, height: int):
        image = np.array(image)
        detect_resolution = max(width, height)
        image, remove_pad = resize_image_with_pad(image, detect_resolution)

        image = np.array(image)
        if self.name == "canny":
            result = self.processor(image, low_threshold=100, high_threshold=200)
        elif self.name == "hough":
            result = self.processor(image, thr_v=0.05, thr_d=5)
        elif self.name == "depth":
            result = self.processor(image)
            result, _ = result
        else:
            result = self.processor(image)

        result = HWC3(remove_pad(result))
        result = cv2.resize(result, (width, height))
        return result


@dataclass
class ModelSpec:
    params: FluxParams
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None
    repo_id: str | None
    repo_flow: str | None
    repo_ae: str | None
    repo_id_ae: str | None


configs = {
    "flux-dev": ModelSpec(
        repo_id="black-forest-labs/FLUX.1-dev",
        repo_id_ae="black-forest-labs/FLUX.1-dev",
        repo_flow="flux1-dev.safetensors",
        repo_ae="ae.safetensors",
        ckpt_path=os.getenv("FLUX_DEV"),
        params=FluxParams(
            in_channels=64,
            vec_in_dim=768,
            context_in_dim=4096,
            hidden_size=3072,
            mlp_ratio=4.0,
            num_heads=24,
            depth=19,
            depth_single_blocks=38,
            axes_dim=[16, 56, 56],
            theta=10_000,
            qkv_bias=True,
            guidance_embed=True,
        ),
        ae_path=os.getenv("AE"),
        ae_params=AutoEncoderParams(
            resolution=256,
            in_channels=3,
            ch=128,
            out_ch=3,
            ch_mult=[1, 2, 4, 4],
            num_res_blocks=2,
            z_channels=16,
            scale_factor=0.3611,
            shift_factor=0.1159,
        ),
    ),
}


def print_load_warning(missing: list[str], unexpected: list[str]) -> None:
    if len(missing) > 0 and len(unexpected) > 0:
        print(f"Got {len(missing)} missing keys: \n\t" + "\n\t".join(missing))
        print("\n" + "-" * 79 + "\n")
        print(f"Got {len(unexpected)} unexpected keys: \n\t" + "\n\t".join(unexpected))
    elif len(missing) > 0:
        print(f"Got {len(missing)} missing keys: \n\t" + "\n\t".join(missing))
    elif len(unexpected) > 0:
        print(f"Got {len(unexpected)} unexpected keys: \n\t" + "\n\t".join(unexpected))


def load_from_repo_id(repo_id, checkpoint_name):
    ckpt_path = hf_hub_download(repo_id, checkpoint_name)
    sd = load_safetensors(ckpt_path)
    return sd


def load_flow_model(name: str, hf_download: bool = True):
    # Loading Flux
    print(f"Init model of {name}")
    ckpt_path = configs[name].ckpt_path
    if ckpt_path is None and configs[name].repo_id is not None and configs[name].repo_flow is not None and hf_download:
        ckpt_path = hf_hub_download(configs[name].repo_id, configs[name].repo_flow)

    with no_init_parameters():
        model = Flux(configs[name].params)
        model = set_model_param_dtype(model, dtype=ms.bfloat16)
        model = model.to_float(ms.bfloat16)

    if ckpt_path is not None:
        print(f"Loading checkpoint of {name}")
        sd = load_safetensors(ckpt_path)
        # missing, unexpected = model.load_state_dict(sd, strict=False)
        missing, unexpected = ms.load_param_into_net(model, sd, strict_load=False)
        print_load_warning(missing, unexpected)
    return model


def load_flow_model_quantized(name: str, hf_download: bool = True):
    raise NotImplementedError("Quantization is not supported in mindspore.")


def load_controlnet(name, transformer=None):
    controlnet = ControlNetFlux(configs[name].params)
    if transformer is not None:
        controlnet.load_state_dict(transformer.state_dict(), strict=False)
    return controlnet


def load_t5(max_length: int = 512) -> HFEmbedder:
    # max length 64, 128, 256 and 512 should work (if your sequence is short enough)
    return HFEmbedder("xlabs-ai/xflux_text_encoders", max_length=max_length, mindspore_dtype=mindspore.bfloat16)


def load_clip() -> HFEmbedder:
    return HFEmbedder("openai/clip-vit-large-patch14", max_length=77, mindspore_dtype=mindspore.bfloat16)


def load_ae(name: str, hf_download: bool = True) -> AutoEncoder:
    ckpt_path = configs[name].ae_path
    if ckpt_path is None and configs[name].repo_id is not None and configs[name].repo_ae is not None and hf_download:
        ckpt_path = hf_hub_download(configs[name].repo_id_ae, configs[name].repo_ae)

    # Loading the autoencoder
    print("Init autoencoder")
    ae = AutoEncoder(configs[name].ae_params)

    if ckpt_path is not None:
        sd = load_safetensors(ckpt_path)
        # missing, unexpected = ae.load_state_dict(sd, strict=False)
        missing, unexpected = ms.load_param_into_net(ae, sd, strict_load=False)
        print_load_warning(missing, unexpected)
    return ae


def set_model_param_dtype(model, dtype=ms.bfloat16, keep_norm_fp32=False):
    if model is not None:
        assert isinstance(model, ms.nn.Cell)

        k_num, c_num = 0, 0
        for _, p in model.parameters_and_names():
            # filter norm/embedding position_ids param
            if keep_norm_fp32 and ("norm" in p.name):
                # print(f"param {p.name} keep {p.dtype}") # disable print
                k_num += 1
            elif "position_ids" in p.name:
                k_num += 1
            else:
                c_num += 1
                p.set_dtype(dtype)

        print(f"Convert '{type(model).__name__}' param to {dtype}, keep/modify num {k_num}/{c_num}.")

    return model


def process_mask(
    mask_path, height, width, dilate=False, dilation_kernel_size=(5, 5), fill_holes=False, closing_kernel_size=(5, 5)
):
    """
    Processes a mask image, optionally fills holes, dilates it, and returns a simple mask tensor.

    Args:
        mask_path (str): The path to the mask image.
        height (int): The desired height for the original image dimensions.
        width (int): The desired width for the original image dimensions.
        dilate (bool, optional): If True, performs a dilation operation to expand the
                                 mask area. Defaults to False.
        dilation_kernel_size (tuple, optional): The size of the kernel for dilation.
                                                Defaults to (5, 5).
        fill_holes (bool, optional): If True, performs a morphological closing operation
                                     to fill small holes within the mask. Defaults to False.
        closing_kernel_size (tuple, optional): The size of the kernel for the closing
                                               operation. Defaults to (5, 5).

    Returns:
        torch.Tensor: The processed simple mask tensor, ready for use.
    """
    # Read the mask image
    mask = cv2.imread(mask_path)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask file from: {mask_path}")

    # Convert the mask to grayscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # Downsample the mask to 1/16th of the target dimensions
    downsampled_mask = cv2.resize(mask, (width // 16, height // 16), interpolation=cv2.INTER_AREA)

    # Threshold the downsampled mask to a binary format (0 or 255)
    _, binary_downsampled_mask = cv2.threshold(downsampled_mask, 127, 255, cv2.THRESH_BINARY)

    # --- Optional Hole Filling Step ---
    # This operation is ideal for making masks contiguous and removing "pepper" noise.
    if fill_holes:
        # Create the kernel for the closing operation.
        kernel = np.ones(closing_kernel_size, np.uint8)
        # Apply morphological closing.
        binary_downsampled_mask = cv2.morphologyEx(binary_downsampled_mask, cv2.MORPH_CLOSE, kernel)

    # --- Optional Dilation Step ---
    # This expands the outer boundary of the mask.
    if dilate:
        # Create a kernel for the dilation.
        kernel = np.ones(dilation_kernel_size, np.uint8)
        # Apply the dilation operation.
        binary_downsampled_mask = cv2.dilate(binary_downsampled_mask, kernel, iterations=1)

    # Normalize the binary mask to have values of 0 and 1
    binary_downsampled_mask = (binary_downsampled_mask // 255).astype(np.uint8)

    # Invert the mask (object area becomes 0, background becomes 1)
    local_mask = 1 - binary_downsampled_mask

    # Convert the final mask to a PyTorch tensor
    local_mask_tensor = ms.tensor(local_mask, dtype=ms.float32)

    return local_mask_tensor


def plot_image_with_mask(image_path, mask_path_list, width, height, save_path):
    # Load the image
    image = cv2.imread(image_path)

    # Resize the image to the specified width and height
    image = cv2.resize(image, (width, height))

    # Convert the image from BGR to RGB for proper display in matplotlib
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    ax.axis("off")

    for mask_path in mask_path_list:
        # Load the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize the mask to match the resized image dimensions
        mask = cv2.resize(mask, (width, height))

        # Find the coordinates of the white region in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get the bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw the bounding rectangle as a red box
            rect = plt.Rectangle((x, y), w, h, edgecolor="red", fill=False, linewidth=4)
            ax.add_patch(rect)

    # Save the figure to the specified output path
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    return save_path
