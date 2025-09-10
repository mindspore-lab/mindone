"""Adapted from https://github.com/huggingface/diffusers/tree/main/src/diffusers/pipelines/consisid/consisid_utils.py."""

import importlib.util

import cv2
import numpy as np
from PIL import Image, ImageOps

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint

from mindone.diffusers.utils import get_logger, load_image
from mindone.utils._eva_clip import EvaCLIPVisionModelWithProjection
from mindone.utils._facexlib.parsing import init_parsing_model
from mindone.utils._facexlib.utils.face_restoration_helper import FaceRestoreHelper, normalize

logger = get_logger(__name__)

_insightface_available = importlib.util.find_spec("insightface") is not None

if _insightface_available:
    import insightface
    from insightface.app import FaceAnalysis
else:
    raise ImportError("insightface is not available. Please install it using 'pip install insightface'.")


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


def resize(img, size, interpolation="bilinear"):
    """
    Simplified, flattened and MindSpore-based version of torchvision.transforms.functional.resize.
    """
    assert isinstance(img, ms.Tensor) and img.ndim >= 2, "Input image must be a tensor with at least 2 dimensions."

    if isinstance(size, (list, tuple)):
        if len(size) not in [1, 2]:
            raise ValueError(
                f"Size must be an int or a 1 or 2 element tuple/list, not a {len(size)} element tuple/list"
            )
    if isinstance(size, int):
        size = [size]

    # _compute_resized_output_size
    height, width = img.shape[-2:]
    short, long = (width, height) if width <= height else (height, width)
    if len(size) == 1:  # specified size only for the smallest edge
        requested_new_short = size if isinstance(size, int) else size[0]
        new_short, new_long = requested_new_short, int(requested_new_short * long / short)
        new_w, new_h = (new_short, new_long) if width <= height else (new_long, new_short)
    else:  # specified both h and w
        new_w, new_h = size[1], size[0]
    output_size = [new_h, new_w]

    if [height, width] == output_size:
        return img

    # Do resize
    # _cast_squeeze_in
    req_dtypes = [ms.float32, ms.float64]
    need_squeeze = False
    # make image NCHW
    if img.ndim < 4:
        img = img.unsqueeze(dim=0)
        need_squeeze = True

    out_dtype = img.dtype
    need_cast = False
    if out_dtype not in req_dtypes:
        need_cast = True
        req_dtype = req_dtypes[0]
        img = img.to(req_dtype)

    # implementation of resize
    align_corners = False if interpolation in ["bilinear", "bicubic"] else None

    img = F.interpolate(img, size=output_size, mode=interpolation, align_corners=align_corners)

    if interpolation == "bicubic" and out_dtype == ms.uint8:
        img = img.clamp(min=0, max=255)

    # _cast_squeeze_out
    if need_squeeze:
        img = img.squeeze(dim=0)

    if need_cast:
        if out_dtype in (ms.uint8, ms.int8, ms.int16, ms.int32, ms.int64):
            # it is better to round before cast
            img = mint.round(img)
        img = img.to(out_dtype)

    return img


def resize_numpy_image_long(image, resize_long_edge=768):
    """
    Resize the input image to a specified long edge while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): Input image (H x W x C or H x W).
        resize_long_edge (int): The target size for the long edge of the image. Default is 768.

    Returns:
        numpy.ndarray: Resized image with the long edge matching `resize_long_edge`, while maintaining the aspect
        ratio.
    """

    h, w = image.shape[:2]
    if max(h, w) <= resize_long_edge:
        return image
    k = resize_long_edge / max(h, w)
    h = int(h * k)
    w = int(w * k)
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == "float64":
                img = img.astype("float32")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = ms.Tensor.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    return _totensor(imgs, bgr2rgb, float32)


def to_gray(img):
    """
    Converts an RGB image to grayscale by applying the standard luminosity formula.

    Args:
        img (mindspore.Tensor): The input image tensor with shape (batch_size, channels, height, width).
                             The image is expected to be in RGB format (3 channels).

    Returns:
        mindspore.Tensor: The grayscale image tensor with shape (batch_size, 3, height, width).
                      The grayscale values are replicated across all three channels.
    """
    x = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    x = x.tile((1, 3, 1, 1))
    return x


def process_face_embeddings(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    weight_dtype,
    image,
    original_id_image=None,
    is_align_face=True,
    **kwargs,
):
    """
    Process face embeddings from an image, extracting relevant features such as face embeddings, landmarks, and parsed
    face features using a series of face detection and alignment tools.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        clip_vision_model: Pre-trained CLIP vision model used for feature extraction.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        eva_transform_mean: Mean values for image normalization before passing to EVA model.
        eva_transform_std: Standard deviation values for image normalization before passing to EVA model.
        app: Application instance used for face detection.
        weight_dtype: Data type of the weights for precision (e.g., `mindspore.float32`).
        image: Input image in RGB format with pixel values in the range [0, 255].
        original_id_image: (Optional) Original image for feature extraction if `is_align_face` is False.
        is_align_face: Boolean flag indicating whether face alignment should be performed.

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding
            - id_vit_hidden: Hidden state of the CLIP vision model, a list of tensors.
            - return_face_features_image_2: Processed face features image after normalization and parsing.
            - face_kps: Keypoints of the face detected in the image.
    """

    face_helper_1.clean_all()
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # get antelopev2 embedding
    face_info = app.get(image_bgr)
    if len(face_info) > 0:
        face_info = sorted(face_info, key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]))[
            -1
        ]  # only use the maximum face
        id_ante_embedding = face_info["embedding"]  # (512,)
        face_kps = face_info["kps"]
    else:
        id_ante_embedding = None
        face_kps = None

    # using facexlib to detect and align face
    face_helper_1.read_image(image_bgr)
    face_helper_1.get_face_landmarks_5(only_center_face=True)
    if face_kps is None:
        face_kps = face_helper_1.all_landmarks_5[0]
    face_helper_1.align_warp_face()
    if len(face_helper_1.cropped_faces) == 0:
        raise RuntimeError("facexlib align face fail")
    align_face = face_helper_1.cropped_faces[0]  # (512, 512, 3)  # RGB

    # in case insightface didn't detect face
    if id_ante_embedding is None:
        logger.warning("Failed to detect face using insightface. Extracting embedding with align face")
        id_ante_embedding = face_helper_2.get_feat(align_face)

    id_ante_embedding = ms.Tensor.from_numpy(id_ante_embedding).to(weight_dtype)  # Size([512])
    if id_ante_embedding.ndim == 1:
        id_ante_embedding = id_ante_embedding.unsqueeze(0)  # Size([1, 512])

    # parsing
    if is_align_face:
        input = img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0  # Size([1, 3, 512, 512])
        parsing_out = face_helper_1.face_parse(normalize(input, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)  # Size([1, 1, 512, 512])
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = mint.ones_like(input)  # Size([1, 3, 512, 512])
        # only keep the face features
        return_face_features_image = mint.where(bg, white_image, to_gray(input))  # Size([1, 3, 512, 512])
        return_face_features_image_2 = mint.where(bg, white_image, input)  # Size([1, 3, 512, 512])
    else:
        original_image_bgr = cv2.cvtColor(original_id_image, cv2.COLOR_RGB2BGR)
        input = img2tensor(original_image_bgr, bgr2rgb=True).unsqueeze(0) / 255.0  # Size([1, 3, 512, 512])
        return_face_features_image = return_face_features_image_2 = input

    # transform img before sending to eva-clip-vit
    face_features_image = resize(
        return_face_features_image, clip_vision_model.config.image_size, "bicubic"
    )  # Size([1, 3, 336, 336])
    face_features_image = normalize(face_features_image, eva_transform_mean, eva_transform_std)
    clip_vision_model_outputs = clip_vision_model(face_features_image.to(weight_dtype), output_hidden_states=True)
    id_cond_vit, id_vit_hidden = (
        clip_vision_model_outputs.image_embeds,
        clip_vision_model_outputs.hidden_states[4:21:4],  # (4, 8, 12, 16, 20)
    )  # Size([1, 768]),  list(Size([1, 577, 1024]))
    id_cond_vit_norm = mint.norm(id_cond_vit, 2, 1, True)
    id_cond_vit = mint.div(id_cond_vit, id_cond_vit_norm)

    id_cond = mint.cat([id_ante_embedding, id_cond_vit], dim=-1)  # Size([1, 512]), Size([1, 768])  ->  Size([1, 1280])

    return (
        id_cond,
        id_vit_hidden,
        return_face_features_image_2,
        face_kps,
    )  # Size([1, 1280]), list(Size([1, 577, 1024]))


def process_face_embeddings_infer(
    face_helper_1,
    clip_vision_model,
    face_helper_2,
    eva_transform_mean,
    eva_transform_std,
    app,
    weight_dtype,
    img_file_path,
    is_align_face=True,
    **kwargs,
):
    """
    Process face embeddings from an input image for inference, including alignment, feature extraction, and embedding
    concatenation.

    Args:
        face_helper_1: Face helper object (first helper) for alignment and landmark detection.
        clip_vision_model: Pre-trained CLIP vision model used for feature extraction.
        face_helper_2: Face helper object (second helper) for embedding extraction.
        eva_transform_mean: Mean values for image normalization before passing to EVA model.
        eva_transform_std: Standard deviation values for image normalization before passing to EVA model.
        app: Application instance used for face detection.
        weight_dtype: Data type of the weights for precision (e.g., `mindspore.float32`).
        img_file_path: Path to the input image file (string) or a numpy array representing an image.
        is_align_face: Boolean flag indicating whether face alignment should be performed (default: True).

    Returns:
        Tuple:
            - id_cond: Concatenated tensor of Ante face embedding and CLIP vision embedding.
            - id_vit_hidden: Hidden state of the CLIP vision model, a list of tensors.
            - image: Processed face image after feature extraction and alignment.
            - face_kps: Keypoints of the face detected in the image.
    """

    # Load and preprocess the input image
    if isinstance(img_file_path, str):
        image = np.array(load_image(image=img_file_path).convert("RGB"))
    else:
        image = np.array(ImageOps.exif_transpose(Image.fromarray(img_file_path)).convert("RGB"))

    # Resize image to ensure the longer side is 1024 pixels
    image = resize_numpy_image_long(image, 1024)
    original_id_image = image

    # Process the image to extract face embeddings and related features
    id_cond, id_vit_hidden, align_crop_face_image, face_kps = process_face_embeddings(
        face_helper_1,
        clip_vision_model,
        face_helper_2,
        eva_transform_mean,
        eva_transform_std,
        app,
        weight_dtype,
        image,
        original_id_image,
        is_align_face,
    )

    # Convert the aligned cropped face image (mindspore tensor) to a numpy array
    tensor = align_crop_face_image
    tensor = tensor.squeeze()
    tensor = tensor.permute(1, 2, 0)
    tensor = tensor.numpy() * 255
    tensor = tensor.astype(np.uint8)
    image = ImageOps.exif_transpose(Image.fromarray(tensor))

    return id_cond, id_vit_hidden, image, face_kps


def prepare_face_models(model_path, dtype, **kwargs):
    """
    Prepare all face models for the facial recognition task.

    Parameters:
    - model_path: Path to the directory containing model files.
    - dtype: Data type (e.g., mindspore.float32) for model inference.

    Returns:
    - face_helper_1: First face restoration helper.
    - face_helper_2: Second face restoration helper.
    - face_clip_model: CLIP model for face extraction.
    - eva_transform_mean: Mean value for image normalization.
    - eva_transform_std: Standard deviation value for image normalization.
    - face_main_model: Main face analysis model.
    """
    # download model weights
    from huggingface_hub import snapshot_download

    repo_id = model_path
    model_path = snapshot_download(repo_id)

    # get helper model
    face_helper_1 = FaceRestoreHelper(
        upscale_factor=1,
        face_size=512,
        crop_ratio=(1, 1),
        det_model="retinaface_resnet50",
        save_ext="png",
        model_rootpath=repo_id,
        subfolder="face_helper_1",
    )
    face_helper_1.face_parse = None
    face_helper_1.face_parse = init_parsing_model(
        model_name="bisenet", model_rootpath=repo_id, subfolder="face_helper_1"
    )
    face_helper_2 = insightface.model_zoo.get_model(
        f"{model_path}/models/glintr100.onnx", providers=["CPUExecutionProvider"]
    )
    face_helper_2.prepare(ctx_id=0)

    # get local facial extractor part 1
    face_clip_model = EvaCLIPVisionModelWithProjection.from_pretrained(repo_id, subfolder="face_clip_model")
    eva_transform_mean = getattr(face_clip_model, "image_mean", OPENAI_DATASET_MEAN)
    eva_transform_std = getattr(face_clip_model, "image_std", OPENAI_DATASET_STD)
    if not isinstance(eva_transform_mean, (list, tuple)):
        eva_transform_mean = (eva_transform_mean,) * 3
    if not isinstance(eva_transform_std, (list, tuple)):
        eva_transform_std = (eva_transform_std,) * 3
    eva_transform_mean = eva_transform_mean
    eva_transform_std = eva_transform_std

    # get local facial extractor part 2
    face_main_model = FaceAnalysis(name="", root=model_path, providers=["CPUExecutionProvider"])
    face_main_model.prepare(ctx_id=0, det_size=(640, 640))

    face_helper_1.face_det.set_train(False)
    face_helper_1.face_parse.set_train(False)
    face_clip_model.set_train(False)
    face_clip_model.to(dtype=dtype)

    return face_helper_1, face_helper_2, face_clip_model, face_main_model, eva_transform_mean, eva_transform_std
