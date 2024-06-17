import os
from typing import List, Tuple, Union

import imageio
import numpy as np


def save_videos(videos: np.ndarray, path: str, fps=8, concat=False):
    # videos: (b f H W 3), normalized to [0, 1]
    videos = (videos * 255).round().clip(0, 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if len(videos.shape) == 4:
        imageio.mimsave(path, videos, fps=fps)
    else:
        b, f, h, w, _ = videos.shape
        if b > 1:
            if concat:
                canvas = np.array((f, h, w * b, 3), dtype=np.uint8)
                for idx in range(b):
                    canvas[:, :, (w * idx) : (w * (idx + 1)), :] = videos[idx]
                imageio.mimsave(path, canvas, fps=fps)
            else:
                for idx in range(b):
                    # concat in Width
                    imageio.mimsave(path.replace(".gif", f"-{idx}.gif"), videos[idx], fps=fps)
        else:
            imageio.mimsave(path, videos[0], fps=fps)


def video_tensor_to_gif(images: np.ndarray, path: str, duration: int = 120, save_frames: bool = False) -> None:
    """images: f x h x w x c"""
    images = (images * 255).round().clip(0, 255).astype(np.uint8)
    images = [x for x in images]
    imageio.mimwrite(path, images, duration=duration)
    if save_frames:
        root = os.path.join(os.path.splitext(path)[0], "frames")
        if not os.path.isdir(root):
            os.makedirs(root)
        for i, x in enumerate(images):
            imageio.imwrite(os.path.join(root, f"{i:04d}.jpg"), x)


def process_mask_strategies(
    mask_strategies: List[Union[str, None]]
) -> List[Union[List[List[Union[int, float]]], None]]:
    default_strategy = [1, 0, 0, 0, 1, 0.0]
    processed = []
    for mst in mask_strategies:  # iterate over all samples
        if mst:
            substrategies = []
            for substrategy in mst.split(";"):  # iterate over strategies for each loop
                substrategy = substrategy.split(",")
                assert 1 <= len(substrategy) <= 6, f"Invalid mask strategy: {substrategy}"
                # the first 5 elements are indexes => int, the last one is the edit ratio => float
                substrategy = [int(s) if i < 5 else float(s) for i, s in enumerate(substrategy)]
                substrategies.append(substrategy + default_strategy[len(substrategy) :])
            processed.append(substrategies)
        else:  # None or empty string
            processed.append(None)
    return processed


def apply_mask_strategy(
    z: np.ndarray,
    references: List[List[np.ndarray]],
    mask_strategies: List[Union[List[Union[int, float]], None]],
    loop_i: int,
) -> Tuple[np.ndarray, np.ndarray]:
    masks = np.ones((z.shape[0], z.shape[2]), dtype=np.float32)
    for batch_id, mask_strategy in enumerate(mask_strategies):
        if mask_strategy is not None:
            for mst in mask_strategy:
                loop_id, ref_id, ref_start, target_start, length, edit_ratio = mst
                if loop_id == loop_i:
                    ref = references[batch_id][ref_id]
                    if ref_start < 0:
                        ref_start = ref.shape[1] + ref_start  # ref: [C, T, H, W]
                    if target_start < 0:
                        target_start = z.shape[2] + target_start  # z: [B, C, T, H, W]
                    z[batch_id, :, target_start : target_start + length] = ref[:, ref_start : ref_start + length]
                    masks[batch_id, target_start : target_start + length] = edit_ratio
    return z, masks


def process_prompts(prompts: List[str], num_loop: int) -> List[List[str]]:
    ret_prompts = []
    for prompt in prompts:
        if prompt.startswith("|0|"):
            prompt_list = prompt.split("|")[1:]
            text_list = []
            for i in range(0, len(prompt_list), 2):
                start_loop = int(prompt_list[i])
                text = prompt_list[i + 1]
                end_loop = int(prompt_list[i + 2]) if i + 2 < len(prompt_list) else num_loop
                text_list.extend([text] * (end_loop - start_loop))
            assert len(text_list) == num_loop, f"Prompt loop mismatch: {len(text_list)} != {num_loop}"
            ret_prompts.append(text_list)
        else:
            ret_prompts.append([prompt] * num_loop)
    return ret_prompts
