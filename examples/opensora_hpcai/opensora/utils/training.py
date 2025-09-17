import random
from dataclasses import dataclass

from mindspore import Tensor, mint, nn


@dataclass
class Condition:
    i2v_head: int = 0
    i2v_loop: int = 0
    i2v_tail: int = 0
    v2v_head: int = 0
    v2v_tail: int = 0
    v2v_head_easy: int = 0
    v2v_tail_easy: int = 0
    t2v: int = 0


@dataclass
class TrainingOptions:
    steps: int = 100


def prepare_visual_condition_uncausal(
    x: Tensor, condition_config: dict[str, int], ae: nn.Cell, pad: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Prepare the visual condition for the model.

    Args:
        x: (Tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        ae (torch.nn.Module): The video encoder module.

    Returns:
        Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B = x.shape[0]
    C = ae.cfg.latent_channels
    T, H, W = ae.get_latent_size(x.shape[-3:])

    # Initialize masks tensor to match the shape of x, but only the time dimension will be masked
    masks = mint.zeros(
        (B, 1, T, H, W), dtype=x.dtype
    )  # broadcasting over channel, concat to masked_x with 1 + 16 = 17 channesl
    # to prevent information leakage, image must be encoded separately and copied to latent
    latent = mint.zeros((B, C, T, H, W), dtype=x.dtype)
    x_0 = mint.zeros((B, C, T, H, W), dtype=x.dtype)
    if T > 1:  # video
        # certain v2v conditions not are applicable for short videos
        if T <= 64 // ae.time_compression_ratio:
            condition_config.pop("v2v_head_easy", None)  # given first 64 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 64 frames
        if T <= 32 // ae.time_compression_ratio:
            condition_config.pop("v2v_head", None)  # given first 32 frames
            condition_config.pop("v2v_tail", None)  # given last 32 frames

        mask_cond_options = list(condition_config.keys())  # list of mask conditions
        mask_cond_weights = list(condition_config.values())  # corresponding probabilities

        for i in range(B):
            # Randomly select a mask condition based on the provided probabilities
            mask_cond = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            # Apply the selected mask condition directly on the masks tensor
            if mask_cond == "i2v_head":  # NOTE: modify video, mask first latent frame
                # padded video such that the first latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                if pad:
                    pad_num = ae.time_compression_ratio - 1  # 32 --> new video: 7 + (1+31-7)
                    padded_x = mint.cat([x[i, :, :1]] * pad_num + [x[i, :, :-pad_num]], dim=1).unsqueeze(0)
                    x_0[i] = ae.encode(padded_x)[0]
                else:
                    x_0[i] = ae.encode(x[i : i + 1])[0]
                # condition: encode the image only
                latent[i, :, :1, :, :] = ae.encode(
                    x[i, :, :1, :, :].unsqueeze(0)
                )  # since the first dimension of right hand side is singleton, torch auto-ignores it
            elif mask_cond == "i2v_loop":  # # NOTE: modify video, mask first and last latent frame
                # pad video such that first and last latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                if pad:
                    pad_num = ae.time_compression_ratio - 1
                    padded_x = mint.cat(
                        [x[i, :, :1]] * pad_num
                        + [x[i, :, : -pad_num * 2]]
                        + [x[i, :, -pad_num * 2 - 1].unsqueeze(1)] * pad_num,
                        dim=1,
                    ).unsqueeze(
                        0
                    )  # remove the last pad_num * 2 frames from the end of the video
                    x_0[i] = ae.encode(padded_x)[0]
                    # condition: encode the image only
                    latent[i, :, :1, :, :] = ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                    latent[i, :, -1:, :, :] = ae.encode(x[i, :, -pad_num * 2 - 1, :, :].unsqueeze(1).unsqueeze(0))
                else:
                    x_0[i] = ae.encode(x[i : i + 1])[0]
                    latent[i, :, :1, :, :] = ae.encode(x[i, :, :1, :, :].unsqueeze(0))
                    latent[i, :, -1:, :, :] = ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == "i2v_tail":  # mask the last latent frame
                masks[i, :, -1, :, :] = 1
                if pad:
                    pad_num = ae.time_compression_ratio - 1
                    padded_x = mint.cat([x[i, :, pad_num:]] + [x[i, :, -1:]] * pad_num, dim=1).unsqueeze(0)
                    x_0[i] = ae.encode(padded_x)[0]
                    latent[i, :, -1:, :, :] = ae.encode(x[i, :, -pad_num * 2 - 1, :, :].unsqueeze(1).unsqueeze(0))
                else:
                    x_0[i] = ae.encode(x[i : i + 1])[0]
                    latent[i, :, -1:, :, :] = ae.encode(x[i, :, -1:, :, :].unsqueeze(0))
            elif mask_cond == "v2v_head":  # mask the first 32 video frames
                assert T > 32 // ae.time_compression_ratio
                conditioned_t = 32 // ae.time_compression_ratio
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = x_0[i, :, :conditioned_t, :, :]
            elif mask_cond == "v2v_tail":  # mask the last 32 video frames
                assert T > 32 // ae.time_compression_ratio
                conditioned_t = 32 // ae.time_compression_ratio
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = x_0[i, :, -conditioned_t:, :, :]
            elif mask_cond == "v2v_head_easy":  # mask the first 64 video frames
                assert T > 64 // ae.time_compression_ratio
                conditioned_t = 64 // ae.time_compression_ratio
                masks[i, :, :conditioned_t, :, :] = 1
                x_0[i] = ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, :conditioned_t, :, :] = x_0[i, :, :conditioned_t, :, :]
            elif mask_cond == "v2v_tail_easy":  # mask the last 64 video frames
                assert T > 64 // ae.time_compression_ratio
                conditioned_t = 64 // ae.time_compression_ratio
                masks[i, :, -conditioned_t:, :, :] = 1
                x_0[i] = ae.encode(x[i].unsqueeze(0))[0]
                latent[i, :, -conditioned_t:, :, :] = x_0[i, :, -conditioned_t:, :, :]
            # elif mask_cond == "v2v_head":  # mask from the beginning to a random point
            #     masks[i, :, : random.randint(1, T - 2), :, :] = 1
            # elif mask_cond == "v2v_tail":  # mask from a random point to the end
            #     masks[i, :, -random.randint(1, T - 2) :, :, :] = 1
            else:
                # "t2v" is the fallback case where no specific condition is specified
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"
                x_0[i] = ae.encode(x[i].unsqueeze(0))[0]
    else:  # image
        x_0 = ae.encode(x)  # latent video

    latent = masks * latent  # condition latent
    # merge the masks and the masked_x into a single tensor
    cond = mint.cat((masks, latent), dim=1)
    return x_0, cond


def prepare_visual_condition_causal(
    x: Tensor, out_shape: tuple[int, int, int, int, int], condition_config: dict[str, int], ae: nn.Cell
) -> Tensor:
    """
    Prepare the visual condition for the model.

    Args:
        x: (Tensor): The input video tensor.
        condition_config (dict): The condition configuration.
        ae (torch.nn.Module): The video encoder module.

    Returns:
        Tensor: The visual condition tensor.
    """
    # x has shape [b, c, t, h, w], where b is the batch size
    B, C, T, H, W = out_shape

    # Initialize mask tensor to match the shape of x, but only the time dimension will be masked
    # broadcasting over the channel dim, concat to masked_x with 1 + 16 = 17 channels
    masks = mint.zeros((B, 1, T, H, W), dtype=x.dtype)
    # to prevent information leakage, image must be encoded separately and copied to latent
    latent = mint.zeros((B, C, T, H, W), dtype=x.dtype)
    if T > 1:  # video
        # certain v2v conditions not are applicable for short videos
        if T <= (64 // ae.time_compression_ratio) + 1:
            condition_config.pop("v2v_head_easy", None)  # given first 65 frames
            condition_config.pop("v2v_tail_easy", None)  # given last 65 frames
        if T <= (32 // ae.time_compression_ratio) + 1:
            condition_config.pop("v2v_head", None)  # given first 33 frames
            condition_config.pop("v2v_tail", None)  # given last 33 frames

        mask_cond_options = list(condition_config.keys())  # list of mask conditions
        mask_cond_weights = list(condition_config.values())  # corresponding probabilities

        for i in range(B):
            # Randomly select a mask condition based on the provided probabilities
            mask_cond: str = random.choices(mask_cond_options, weights=mask_cond_weights, k=1)[0]
            # Apply the selected mask condition directly on the masks tensor

            if mask_cond == "i2v_head":  # NOTE: modify video, mask first latent frame
                masks[i, :, 0, :, :] = 1
                # condition: encode the image only
                latent[i, :, :1, :, :] = ae.encode(x[i, :, :1, :, :].unsqueeze(0)).squeeze(0)

            elif mask_cond == "i2v_loop":  # # NOTE: modify video, mask first and last latent frame
                # pad video such that first and last latent frame correspond to image only
                masks[i, :, 0, :, :] = 1
                masks[i, :, -1, :, :] = 1
                # condition: encode the image only
                latent[i, :, :1, :, :] = ae.encode(x[i, :, :1, :, :].unsqueeze(0)).squeeze(0)
                latent[i, :, -1:, :, :] = ae.encode(x[i, :, -1:, :, :].unsqueeze(0)).squeeze(0)

            elif mask_cond == "i2v_tail":  # mask the last latent frame
                masks[i, :, -1, :, :] = 1
                # condition: encode the last image only
                latent[i, :, -1:, :, :] = ae.encode(x[i, :, -1:, :, :].unsqueeze(0)).squeeze(0)

            elif mask_cond.startswith("v2v_head"):  # mask the first 33 video frames
                ref_t = 33 if not mask_cond.endswith("easy") else 65
                assert (ref_t - 1) % ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // ae.time_compression_ratio + 1
                masks[i, :, :conditioned_t, :, :] = 1
                # encode the first ref_t frame video separately
                latent[i, :, :conditioned_t, :, :] = ae.encode(x[i, :, :ref_t, :, :].unsqueeze(0)).squeeze(0)

            elif mask_cond.startswith("v2v_tail"):  # mask the last 32 video frames
                ref_t = 33 if not mask_cond.endswith("easy") else 65
                assert (ref_t - 1) % ae.time_compression_ratio == 0
                conditioned_t = (ref_t - 1) // ae.time_compression_ratio + 1
                masks[i, :, -conditioned_t:, :, :] = 1
                # encode the first ref_t frame video separately
                latent[i, :, -conditioned_t:, :, :] = ae.encode(x[i, :, -ref_t:, :, :].unsqueeze(0)).squeeze(0)
            else:
                # "t2v" is the fallback case where no specific condition is specified
                assert mask_cond == "t2v", f"Unknown mask condition {mask_cond}"

    latent = masks * latent  # condition latent
    # merge the masks and the masked_x into a single tensor
    return mint.cat((masks, latent), dim=1)
