import math
import random
from typing import Callable, Union

import numpy as np
from src.model import Flux
from tqdm import tqdm

import mindspore as ms
from mindspore import Tensor, mint


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    dtype: ms.dtype,
    seed: int,
):
    ms.set_seed(seed)
    return ms.mint.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        dtype=dtype,
    )


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = ms.mint.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # eastimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


# Add
def denoise_fireflow(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info,
    guidance: float = 4.0,
):
    if inverse:
        timesteps = timesteps[::-1]
    guidance_vec = ms.ops.full((img.shape[0],), guidance, dtype=img.dtype)

    next_step_velocity = None
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = ms.ops.full((img.shape[0],), t_curr, dtype=img.dtype)
        info["t"] = t_prev if inverse else t_curr
        info["inverse"] = inverse
        info["second_order"] = False

        if inverse is True:
            if next_step_velocity is None:
                block_res_samples = info["controlnet"](
                    img=img,
                    img_ids=img_ids,
                    controlnet_cond=info["controlnet_cond"],
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )

                pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt,
                    txt_ids=txt_ids,
                    y=vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    block_controlnet_hidden_states=[ij * info["controlnet_gs"] for ij in block_res_samples],
                )

            else:
                pred = next_step_velocity

            img_mid = img + (t_prev - t_curr) / 2 * pred

            t_vec_mid = ms.ops.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype)
            info["second_order"] = True

            block_res_samples = info["controlnet"](
                img=img_mid,
                img_ids=img_ids,
                controlnet_cond=info["controlnet_cond"],
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec,
            )

            pred_mid = model(
                img=img_mid,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec_mid,
                guidance=guidance_vec,
                block_controlnet_hidden_states=[ij * info["controlnet_gs"] for ij in block_res_samples],
            )

            next_step_velocity = pred_mid
            img = img + (t_prev - t_curr) * pred_mid
            info[t_curr] = img

    return img, info


def process_mask(input_mask, height, width, latent_image, kernel_size=1):
    """
    Process the input mask and return processed_mask, dilated_mask, and flattened_mask.

    Args:
        input_mask (torch.Tensor or None): Input mask tensor or None.
        height (int): Height to be used for processing.
        width (int): Width to be used for processing.
        latent_image (torch.Tensor): Source image latent tensor (used for dtype).
        kernel_size (int): Size of the dilation kernel (default is 1).

    Returns:
        tuple: (processed_mask, dilated_mask, flattened_mask)
    """
    # Initialize the processed mask based on the input mask
    if input_mask is None:
        processed_mask = ms.mint.ones((1, int(height / 16) * int(width / 16), 1))
    else:
        processed_mask = input_mask.copy()
    # Ensure processed_mask has the correct dtype and is on GPU
    processed_mask = processed_mask.to(latent_image.dtype)
    # original_mask = processed_mask.copy()
    # Convert processed_mask to numpy and prepare for dilation
    processed_mask_np = (1 - processed_mask.copy().float()).asnumpy()
    processed_mask_np = np.squeeze(processed_mask_np).reshape(int(height / 16), int(width / 16))
    # Kernel size and number of iterations for dilation
    # iterations = 1
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    # Perform dilation (currently commented out in the original code)
    dilated_mask_np = processed_mask_np  # Example: cv2.dilate(processed_mask_np, kernel, iterations=iterations)
    dilated_mask_np_larger = processed_mask_np  # Example: cv2.dilate(processed_mask_np, (4 * int(height / 512), 4 * int(height / 512)), iterations=iterations)
    # Convert dilated masks back to torch tensors
    dilated_mask = ms.tensor(dilated_mask_np, dtype=ms.float32).flatten().unsqueeze(1)
    dilated_mask_larger = ms.tensor(dilated_mask_np_larger, dtype=ms.float32).flatten().unsqueeze(1)
    # Update processed_mask and dilated_mask_larger
    processed_mask = 1 - dilated_mask
    dilated_mask = 1 - dilated_mask_larger
    # Compute flattened_mask
    flattened_mask = (1 - processed_mask).flatten()
    return processed_mask, dilated_mask, flattened_mask


def generate_combined_noise(local_mask1_flat, x2, epsilon, idx, info, timesteps):
    """
    Generate combined noise based on a local mask and input noise tensors.

    Args:
        local_mask1_flat (torch.Tensor): A 1D binary tensor indicating the mask (0s and 1s).
        x2 (torch.Tensor): Noise tensor to be scaled and combined.
        epsilon (float): Scaling factor for the second noise tensor.
    Returns:
        torch.Tensor: Combined noise tensor.
    """
    # Count the number of 1s in the mask
    num_ones = int(mint.sum(local_mask1_flat).item())
    # Create a tensor `b` with the same size as `local_mask1_flat` and initialize it with 0
    b = mint.zeros_like(local_mask1_flat)
    # Get the indices of 0s and 1s in `local_mask1_flat`
    zero_indices = mint.where(local_mask1_flat == 0)[0]
    one_indices = mint.where(local_mask1_flat == 1)[0]

    # Handle sampling based on the number of zeros and ones
    if len(zero_indices) >= num_ones:
        # Randomly select `num_ones` indices from the 0 region
        selected_indices = mint.randperm(len(zero_indices))[:num_ones]
        b[zero_indices[selected_indices]] = 1
    else:
        # Select all 0 indices and calculate remaining
        b[zero_indices] = 1
        remaining = num_ones - len(zero_indices)
        # Randomly select the remaining indices from the 1 region
        selected_indices = mint.randperm(len(one_indices))[:remaining]
        b[one_indices[selected_indices]] = 1

    # Extract and normalize noise1 based on the updated mask
    noise1 = info[timesteps[idx + 10]][:, b.bool(), :]
    mean = mint.mean(noise1)
    std = mint.std(noise1)
    noise1 = (noise1 - mean) / std

    # Extract and normalize noise2 based on the original mask
    noise2 = x2[:, local_mask1_flat.bool(), :]
    mean = mint.mean(noise2)
    std = mint.std(noise2)
    noise2 = (noise2 - mean) / std
    # Combine noise1 and scaled noise2
    combined_noise = noise1 + epsilon * noise2
    # Normalize the combined noise
    mean = mint.mean(combined_noise)
    std = mint.std(combined_noise)
    combined_noise = (combined_noise - mean) / std

    return combined_noise


def denoise_cannyedit_removal(
    model: Flux,
    controlnet: None,
    source_image_latent: Tensor,
    source_image_latent_rg: Tensor,
    img: Tensor,
    img_ids: Tensor,
    # source prompt-related embeddings
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # local prompt 1-related embeddings
    txt2: Tensor,
    txt_ids2: Tensor,
    vec2: Tensor,
    # target prompt-related embeddings
    txt3: Tensor,
    txt_ids3: Tensor,
    vec3: Tensor,
    # additional local prompts-related embeddings
    txt_addition: list[Tensor],
    txt_ids_addition: list[Tensor],
    vec_addition: list[Tensor],
    # negative prompt-related embeddings
    neg_txt: Tensor,
    neg_txt_ids: Tensor,
    neg_vec: Tensor,
    local_mask,
    local_mask_addition,
    controlnet_cond,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    true_gs=1,
    controlnet_gs=0.7,
    controlnet_gs2=0.5,
    timestep_to_start_cfg=0,
    # ip-adapter parameters
    image_proj: Tensor = None,
    neg_image_proj: Tensor = None,
    ip_scale: Union[Tensor, float] = 1,
    neg_ip_scale: Union[Tensor, float] = 1,
    seed=random.randint(0, 99999),
    generate_save_path=None,
    inversion_save_path=None,
    stage="stage_removal",
):
    guidance_vec = ms.ops.full((img.shape[0],), guidance, dtype=img.dtype)
    t_length = len(timesteps)
    if generate_save_path is not None:
        info_generate = []

    time_to_start = 2
    i = 0

    for t_curr, t_prev in tqdm(
        zip(timesteps[:-1], timesteps[1:]), total=len(timesteps) - 1, desc="CannyEdit Denoising Steps"
    ):
        if i == 0:
            # -------------------------Inversion-------------------------------------------------------------------------

            if stage == "stage_removal":
                timesteps_inv = timesteps  # [time_to_start-2:]
                info = {}
                info["controlnet_cond"] = controlnet_cond
                info["controlnet"] = controlnet
                info["controlnet_gs"] = controlnet_gs2

                z, info = denoise_fireflow(
                    model,
                    source_image_latent_rg,
                    img_ids,
                    txt,
                    txt_ids,
                    vec,
                    timesteps_inv,
                    guidance=1,
                    inverse=True,
                    info=info,
                )

                if inversion_save_path is not None and stage == "stage_removal":
                    np.save(inversion_save_path, info)

            if stage == "stage_removal_regen" and inversion_save_path is not None:
                print("load previous inversion results")
                info = np.load(inversion_save_path, allow_pickle=True).item()
            # print('Inversion End....')

            # -----------------------End of Inversion----------------------------------------------------------------------

            # ---------------------Processing mask---------------------------------------------------------------------
            # print('Denoising Start....')
            bs, c, h, w = source_image_latent.shape
            H_use = int(h * 8)
            W_use = int(w * 8)

            b, c, h, w = source_image_latent.shape
            sh = h // 2  # ph=2
            sw = w // 2  # pw=2
            # source_image_latent = rearrange(source_image_latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2) # keep fpr debugging
            source_image_latent = source_image_latent.reshape(b, c, sh, 2, sw, 2)
            source_image_latent = source_image_latent.permute(0, 2, 4, 1, 3, 5)
            source_image_latent = source_image_latent.reshape(b, sh * sw, c * 4)

            source_image_latent = source_image_latent.broadcast_to((bs, *source_image_latent.shape[1:]))
            # source_image_latent = repeat(source_image_latent, "1 ... -> bs ...", bs=bs) # keep fpr debugging

            # process the first local mask
            # after processing, the value inside the edit region is 0 and is 1 elsewhere in local_mask1_proceed and local_mask1_dilate;
            # for local_mask1_flat, inside edit region is 1 instead
            local_mask1_proceed, local_mask1_dilate, local_mask1_flat = process_mask(
                local_mask, H_use, W_use, source_image_latent, kernel_size=1
            )
            # process the additional local masks
            local_mask_add_proceed = []
            local_mask_add_dilate = []
            local_mask_add_flat = []
            local_mask_all_dilate = []
            local_mask_all_dilate.append(local_mask1_dilate)
            if local_mask_addition != []:
                for local_mask1 in local_mask_addition:
                    local_mask2_proceed, local_mask2_dilate, local_mask2_flat = process_mask(
                        local_mask1, H_use, W_use, source_image_latent, kernel_size=1
                    )
                    local_mask_add_proceed.append(local_mask2_proceed)
                    local_mask_add_dilate.append(local_mask2_dilate)
                    local_mask_add_flat.append(local_mask2_flat)
                    local_mask_all_dilate.append(local_mask2_dilate)

            # initialize the mask (union_mask) used for canny control relaxation and blending, where the value inside the union of the edit
            #  regions is 0 and is 1 elsewhere.
            if local_mask_addition == []:
                union_mask = local_mask1_dilate
            elif local_mask_addition != []:
                union_inverted = 1 - local_mask1_dilate
                for mask_dilate in local_mask_add_dilate:
                    mask_dilate_inverted = 1 - mask_dilate
                    union_inverted = ms.mint.logical_or(union_inverted.bool(), mask_dilate_inverted.bool())
                union_inverted = union_inverted.int()
                union_mask = 1 - union_inverted
            # ------------------End of processing mask------------------------------------------------------------------

            # ------------------Handle attention mask-------------------------------------------------------------------
            # len(local_mask_add_proceed)=number of additional local edit prompts + 1 (the first local edit prompt) + 1 (target prompt)
            conds = [None] * (len(local_mask_add_proceed) + 2)
            masks = [None] * (len(local_mask_add_proceed) + 2)

            # the first local prompt and mask for the first local edit region
            conds[1] = txt2
            masks[1] = 1 - local_mask1_proceed.flatten().unsqueeze(1).repeat(1, conds[1].shape[1])
            # the additional local prompts and their corresponding local edit regions
            for indd in range(len(local_mask_add_proceed)):
                conds[2 + indd] = txt_addition[indd]
                masks[2 + indd] = 1 - local_mask_add_proceed[indd].flatten().unsqueeze(1).repeat(
                    1, conds[2 + indd].shape[1]
                )
            # the target prompt and its mask see the whole image
            conds[0] = txt3
            masks[0] = ms.mint.ones_like(masks[1])

            regional_embeds = ms.mint.cat(conds, dim=1)
            encoder_seq_len = regional_embeds.shape[1]
            hidden_seq_len = source_image_latent.shape[1]
            txt_ids_region = ms.mint.zeros((regional_embeds.shape[1], 3)).to(dtype=txt_ids.dtype).unsqueeze(0)

            # initialize attention mask
            regional_attention_mask = ms.mint.zeros(
                (encoder_seq_len + hidden_seq_len, encoder_seq_len + hidden_seq_len), dtype=ms.bool
            )
            num_of_regions = len(masks)
            each_prompt_seq_len = encoder_seq_len // num_of_regions

            # ================================
            # T2T, T2I and I2T attention mask
            # Each text can only see itself
            # Local prompt can only see/be seen by the local edit region
            # Target prompt can see/be seen by the whole image
            for ij in range(num_of_regions):
                # t2t mask txt attends to itself
                regional_attention_mask[
                    ij * each_prompt_seq_len : (ij + 1) * each_prompt_seq_len,
                    ij * each_prompt_seq_len : (ij + 1) * each_prompt_seq_len,
                ] = True
                # t2i and i2t mask
                regional_attention_mask[
                    ij * each_prompt_seq_len : (ij + 1) * each_prompt_seq_len, encoder_seq_len:
                ] = masks[ij].transpose(-1, -2)
                regional_attention_mask[
                    encoder_seq_len:, ij * each_prompt_seq_len : (ij + 1) * each_prompt_seq_len
                ] = masks[ij]

            # ================================
            # I2I mask
            # I2I attention mask:
            # initialization

            attention_mask_i2i = ms.mint.zeros(
                (int(H_use / 16) * int(W_use / 16), int(H_use / 16) * int(W_use / 16)), dtype=ms.int
            )
            # background region can only see background region
            # Find the union of regions where both masks are 0
            zero_union_mask = local_mask1_flat == 0
            # Iterate over all masks in the list and combine their conditions
            for local_mask2_flat in local_mask_add_flat:
                zero_union_mask &= local_mask2_flat == 0
            zero_union_indices = ms.mint.nonzero(zero_union_mask, as_tuple=True)[0]
            # attention_mask_i2i[zero_union_indices[:, None],] = 1
            attention_mask_i2i[zero_union_indices[:, None], zero_union_indices] = 1

            # edited region can only see the whole image
            mask1_indices = ms.mint.nonzero(local_mask1_flat, as_tuple=True)[0]
            attention_mask_i2i[mask1_indices, :] = 1

            for local_mask2_flat in local_mask_add_flat:
                mask2_indices = ms.mint.nonzero(local_mask2_flat, as_tuple=True)[0]
                attention_mask_i2i[mask2_indices, :] = 1

            regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = ms.mint.ones_like(
                regional_attention_mask[encoder_seq_len:, encoder_seq_len:]
            )
            # regional_attention_mask[encoder_seq_len:, encoder_seq_len:] = attention_mask_i2i
            # ================================

        # ------------------End of Handle attention mask-------------------------------------------------------------------

        apply_local_point = 0.7
        apply_extenda_point = 0.5

        # 0818
        try:
            tempp = timesteps[i + 10]
        except IndexError:
            tempp = timesteps[i]

        if i >= time_to_start:
            if i == time_to_start:
                img = info[tempp]
                # ------------------Reinitialize each local edit region--------------------------------------------------

                x2 = get_noise(1, H_use, W_use, dtype=ms.bfloat16, seed=seed)
                bs, c, h, w = x2.shape

                b, c, h, w = x2.shape
                sh = h // 2  # ph=2
                sw = w // 2  # pw=2
                # x2 = rearrange(x2, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)  # keep for debugging
                x2 = x2.reshape(b, c, sh, 2, sw, 2)
                x2 = x2.permute(0, 2, 4, 1, 3, 5)
                x2 = x2.reshape(b, sh * sw, c * 4)

                # x2 = repeat(x2, "1 ... -> bs ...", bs=bs) # keep for debugging
                x2 = x2.broadcast_to((bs, *x2.shape[1:]))
                epsilon = 3
                combined_noise = generate_combined_noise(local_mask1_flat, x2, epsilon, i, info, timesteps)
                img[:, local_mask1_flat.bool(), :] = combined_noise
                seed += 1

                for local_mask2_flat in local_mask_add_flat:
                    x3 = get_noise(1, H_use, W_use, dtype=ms.bfloat16, seed=seed)
                    bs, c, h, w = x3.shape

                    b, c, h, w = x3.shape
                    sh = h // 2  # ph=2
                    sw = w // 2  # pw=2
                    # x3 = rearrange(x3, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)  # keep for debugging
                    x3 = x3.reshape(b, c, sh, 2, sw, 2)
                    x3 = x3.permute(0, 2, 4, 1, 3, 5)
                    x3 = x3.reshape(b, sh * sw, c * 4)

                    # x3= repeat(x3, "1 ... -> bs ...", bs=bs)
                    x3 = x3.broadcast_to((bs, *x3.shape[1:]))
                    combined_noise = generate_combined_noise(local_mask2_flat, x3, epsilon, i, info, timesteps)
                    img[:, local_mask2_flat.bool(), :] = combined_noise
                    seed += 1

                # ------------------END of Reinitialize each local edit region-------------------------------------------

            # ================================== Start Denoising =================================================

            t_vec = ms.mint.full((img.shape[0],), t_curr, dtype=img.dtype)
            imgg = info[tempp]

            block_res_samples = controlnet(
                # use img=imgg if we want to use the original image+noise in the controlnet
                img=imgg,
                img_ids=img_ids,
                controlnet_cond=controlnet_cond,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )
            # ------------------Selective Canny Masking-------------------------------------------------------------------
            soft_masks = []
            # Generate a Gaussian soft mask for each tensor
            for tensor in block_res_samples:
                soft_masks.append(union_mask.to(dtype=tensor.dtype))  # Add to list
            # print("soft_masks applied!")
            block_res_samples = [block_res_samples[i] * soft_masks[i] for i in range(len(block_res_samples))]
            # ------------------End of Selective Canny Masking--------------------------------------------------------------

            # Stage 1: Regional Denoising
            if i < int(apply_local_point * t_length):
                attention_kwargs = {}
                # *********************************************
                # union of all local mask
                attention_kwargs["union_mask"] = union_mask
                # input each local mask
                attention_kwargs["local_mask_all_dilate"] = local_mask_all_dilate
                # number of local edit regions
                attention_kwargs["num_edit_region"] = 1 + len(txt_addition)
                # attention_mask
                attention_kwargs["regional_attention_mask"] = regional_attention_mask.bool()
                # amplify the attention between the local text promt and local edit region
                attention_kwargs["local_t2i_strength"] = 1 + 0.4 * (1 - (i / (apply_local_point * t_length)))
                # amplify the attention between the target prompt and the whole image
                attention_kwargs["context_t2i_strength"] = 1.0
                # amplify the attention within each edit region
                attention_kwargs["local_i2i_strength"] = 1.0
                # amplify the attention between each edit region and other regions
                attention_kwargs["local2out_i2i_strength"] = 1.0 + (0.35) * (1 - (i / (apply_local_point * t_length)))
                attention_kwargs["image_size"] = int(H_use / 16) * int(W_use / 16)
                # *********************************************
                if i <= int(t_length * apply_extenda_point):
                    controlnet_control = [ij * controlnet_gs2 for ij in block_res_samples]
                else:
                    controlnet_control = None
                pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=regional_embeds,
                    txt_ids=txt_ids_region,
                    y=ms.mint.zeros_like(vec3),
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    block_controlnet_hidden_states=controlnet_control,
                    image_proj=image_proj,
                    ip_scale=ip_scale,
                    attention_kwargs=attention_kwargs,
                )

            # Stage 2: Normal Denoising
            else:
                attention_kwargs = {}
                if i <= int(t_length * apply_extenda_point):
                    controlnet_control = [ij * controlnet_gs2 for ij in block_res_samples]
                else:
                    controlnet_control = None

                pred = model(
                    img=img,
                    img_ids=img_ids,
                    txt=txt3,
                    txt_ids=txt_ids3,
                    y=vec3,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    block_controlnet_hidden_states=controlnet_control,
                    image_proj=image_proj,
                    ip_scale=ip_scale,
                    attention_kwargs=attention_kwargs,
                )

            cfg_step = 30
            if i < cfg_step:
                neg_pred1 = model(
                    img=img,
                    img_ids=img_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                    block_controlnet_hidden_states=controlnet_control,
                    image_proj=image_proj,
                    ip_scale=ip_scale,
                )

                true_gs1 = 3 + (true_gs) * (1 - (i / (cfg_step)))
                pred = mint.where(union_mask == 1, pred, neg_pred1 + true_gs1 * (pred - neg_pred1))

            img = img + (t_prev - t_curr) * pred

            # Cyclical Blending
            if i < 10 or (i <= 30 and i % 5 == 0):
                img = mint.where(union_mask == 1, 0.5 * info[tempp] + 0.5 * img, img)
            elif i <= 40 and i % 10 == 0:
                img = mint.where(union_mask == 1, 0.3 * info[tempp] + 0.7 * img, img)

            if generate_save_path is not None:
                info_generate[i] = img

        i += 1

    if generate_save_path is not None and stage == "stage_generate":
        np.save(generate_save_path, info_generate)
    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    b = x.shape[0]
    h = math.ceil(height / 16)
    w = math.ceil(width / 16)
    # return rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)",
    #                  h=h, w=w, ph=2, pw=2)  # keep for debugging
    x = x.reshape(b, h * w, -1)  # -1 will infer the correct size for c*ph*pw
    c = x.shape[2] // 4  # since ph=pw=2, total is 4
    x = x.reshape(b, h, w, c, 2, 2)
    x = x.transpose(0, 3, 1, 4, 2, 5)
    x = x.reshape(b, c, h * 2, w * 2)
    return x
