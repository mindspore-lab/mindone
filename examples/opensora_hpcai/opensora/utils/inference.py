import os
from itertools import zip_longest

import numpy as np
from PIL import Image

from mindspore import Tensor, nn, tensor
from mindspore.common import dtype as mstype

from mindone.visualize.videos import save_videos

from ..utils.cond_data import get_references


def process_and_save(x: Tensor, ids: list[int], sub_ids: list[int], save_dir: str, fps: int = 24):
    """
    x: B C T H W
    """
    x = x.to(mstype.float32).numpy()
    x = (((x.clip(-1, 1) + 1) / 2) * 255 + 0.5).clip(0, 255).transpose(0, 2, 3, 4, 1).astype(np.uint8)

    ext, is_image = ".mp4", False
    if x.shape[1] == 1:
        ext, is_image = ".png", True
        x = x.squeeze(axis=1)

    paths = []
    for im, id_, sid in zip_longest(x, ids, sub_ids):
        paths.append(os.path.join(save_dir, f"{id_:05d}{f'_{sid:03d}' if sid else ''}{ext}"))
        if is_image:
            Image.fromarray(im).save(paths[-1])
        else:
            save_videos(im, paths[-1], fps, normalize=False)

    return paths


def collect_references_batch(
    reference_paths: list[str],
    cond_type: str,
    model_ae: nn.Cell,
    image_size: tuple[int, int],
    is_causal=False,
):
    refs_x = []  # refs_x: [batch, ref_num, C, T, H, W]
    for reference_path in reference_paths:
        if reference_path == "":
            refs_x.append(None)
            continue
        ref_path = reference_path.split(";")
        ref = []

        if "v2v" in cond_type:
            r = get_references([ref_path[0]], image_size)[0][0]  # size [C, T, H, W]
            actual_t = r.shape[1]
            target_t = (
                64 if (actual_t >= 64 and "easy" in cond_type) else 32
            )  # if reference not long enough, default to shorter ref
            if is_causal:
                target_t += 1
            assert actual_t >= target_t, f"need at least {target_t} reference frames for v2v generation"
            if "head" in cond_type:  # v2v head
                r = r[:, :target_t]
            elif "tail" in cond_type:  # v2v tail
                r = r[:, -target_t:]
            else:
                raise NotImplementedError
            r_x = model_ae.encode(tensor(r, dtype=mstype.float32))
            r_x = r_x.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x)
        elif cond_type == "i2v_head":  # take the 1st frame from first ref_path
            r = get_references([ref_path[0]], image_size)[0][0]  # size [C, T, H, W]
            r = r[:, :, :1]
            r_x = model_ae.encode(tensor(r, dtype=mstype.bfloat16)).to(mstype.float32)  # FIXME:
            r_x = r_x.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x)
        elif cond_type == "i2v_tail":  # take the last frame from last ref_path
            r = get_references([ref_path[-1]], image_size)[0][0]  # size [C, T, H, W]
            r = r[:, -1:]
            r_x = model_ae.encode(tensor(r, dtype=mstype.float32))
            r_x = r_x.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x)
        elif cond_type == "i2v_loop":
            # first frame
            r_head = get_references([ref_path[0]], image_size)[0][0]  # size [C, T, H, W]
            r_head = r_head[:, :1]
            r_x_head = model_ae.encode(tensor(r_head, dtype=mstype.float32))
            r_x_head = r_x_head.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x_head)
            # last frame
            r_tail = get_references([ref_path[-1]], image_size)[0][0]  # size [C, T, H, W]
            r_tail = r_tail[:, -1:]
            r_x_tail = model_ae.encode(tensor(r_tail, dtype=mstype.float32))
            r_x_tail = r_x_tail.squeeze(0)  # size [C, T, H, W]
            ref.append(r_x_tail)
        else:
            raise NotImplementedError(f"Unknown condition type {cond_type}")

        refs_x.append(ref)
    return refs_x


def prepare_inference_condition(
    z: np.ndarray,
    mask_cond: str,
    ref_list: list[list[Tensor]] = None,
    causal: bool = True,
    dtype: mstype.Type = mstype.float32,
) -> tuple[Tensor, Tensor]:
    """
    Prepare the visual condition for the model, using causal vae.

    Args:
        z (torch.Tensor): The latent noise tensor, of shape [B, C, T, H, W]
        mask_cond (dict): The condition configuration.
        ref_list: list of lists of media (image/video) for i2v and v2v condition, of shape [C, T', H, W];
        len(ref_list)==B; ref_list[i] is the list of media for the generation in batch idx i, we use a list of media for
        each batch item so that it can have multiple references. For example, ref_list[i] could be [ref_image_1, ref_image_2] for i2v_loop condition.

    Returns:
        torch.Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B, C, T, H, W = z.shape

    masks = np.zeros((B, 1, T, H, W), dtype=np.int32)
    masked_z = tensor(np.zeros((B, C, T, H, W)), dtype=dtype)

    if ref_list is None:
        assert mask_cond == "t2v", f"reference is required for {mask_cond}"

    for i in range(B):
        ref = ref_list[i]

        # warning message
        if ref is None and mask_cond != "t2v":
            print("no reference found. will default to cond_type t2v!")

        if ref is not None and T > 1:  # video
            # Apply the selected mask condition directly on the masks tensor
            if mask_cond == "i2v_head":  # equivalent to masking the first timestep
                masks[i, :, 0, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
            elif mask_cond == "i2v_tail":  # mask the last timestep
                masks[i, :, -1, :, :] = 1
                masked_z[i, :, -1, :, :] = ref[-1][:, -1, :, :]
            elif mask_cond == "v2v_head":
                k = 8 + int(causal)
                masks[i, :, :k, :, :] = 1
                masked_z[i, :, :k, :, :] = ref[0][:, :k, :, :]
            elif mask_cond == "v2v_tail":
                k = 8 + int(causal)
                masks[i, :, -k:, :, :] = 1
                masked_z[i, :, -k:, :, :] = ref[0][:, -k:, :, :]
            elif mask_cond == "v2v_head_easy":
                k = 16 + int(causal)
                masks[i, :, :k, :, :] = 1
                masked_z[i, :, :k, :, :] = ref[0][:, :k, :, :]
            elif mask_cond == "v2v_tail_easy":
                k = 16 + int(causal)
                masks[i, :, -k:, :, :] = 1
                masked_z[i, :, -k:, :, :] = ref[0][:, -k:, :, :]
            elif mask_cond == "i2v_loop":  # mask first and last timesteps
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                masked_z[i, :, 0, :, :] = ref[0][:, 0, :, :]
                masked_z[i, :, -1, :, :] = ref[-1][:, -1, :, :]  # last frame of last referenced content
            else:
                # "t2v" is the fallback case where no specific condition is specified
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"

    masks = tensor(masks, dtype=dtype)
    return masks, masked_z
