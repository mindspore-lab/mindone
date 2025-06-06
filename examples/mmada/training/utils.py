import math
import random
from typing import Any, List, Tuple

from omegaconf import DictConfig, ListConfig, OmegaConf

import mindspore as ms
import mindspore.mint.nn.functional as F
from mindspore import mint


##################################################
#              config utils
##################################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def flatten_omega_conf(cfg: Any, resolve: bool = False) -> List[Tuple[str, Any]]:
    ret = []

    def handle_dict(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{k1}", v1) for k1, v1 in flatten_omega_conf(value, resolve=resolve)]

    def handle_list(key: Any, value: Any, resolve: bool) -> List[Tuple[str, Any]]:
        return [(f"{key}.{idx}", v1) for idx, v1 in flatten_omega_conf(value, resolve=resolve)]

    if isinstance(cfg, DictConfig):
        for k, v in cfg.items_ex(resolve=resolve):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(k, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(k, v, resolve=resolve))
            else:
                ret.append((str(k), v))
    elif isinstance(cfg, ListConfig):
        for idx, v in enumerate(cfg._iter_ex(resolve=resolve)):
            if isinstance(v, DictConfig):
                ret.extend(handle_dict(idx, v, resolve=resolve))
            elif isinstance(v, ListConfig):
                ret.extend(handle_list(idx, v, resolve=resolve))
            else:
                ret.append((str(idx), v))
    else:
        assert False

    return ret


##################################################
#              training utils
##################################################
def soft_target_cross_entropy(logits, targets, soft_targets):
    # ignore the first token from logits and targets (class id token)
    logits = logits[:, 1:]
    targets = targets[:, 1:]

    logits = logits[..., : soft_targets.shape[-1]]

    log_probs = F.log_softmax(logits, dim=-1)
    padding_mask = targets.eq(-100)

    loss = mint.sum(-soft_targets * log_probs, dim=-1)
    loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.to(ms.int64).sum()
    loss = loss.sum() / num_active_elements
    return loss


def get_loss_weight(t, mask, min_val=0.3):
    return 1 - (1 - mask) * ((1 - t) * (1 - min_val))[:, None]


def mask_or_random_replace_tokens(image_tokens, mask_id, config, mask_schedule, is_train=True, seed=None):
    batch_size, seq_len = image_tokens.shape

    if not is_train and seed is not None:
        # 保存当前随机状态
        rng_state = ms.get_rng_state()
        python_rng_state = random.getstate()

        # 设置固定种子
        ms.manual_seed(seed)

        random.seed(seed)
        # print(f"Set seed to {seed}")

    if not is_train and config.training.get("eval_mask_ratios", None):
        mask_prob = random.choices(config.training.eval_mask_ratios, k=batch_size)
        mask_prob = ms.tensor(
            mask_prob,
        )
    else:
        # Sample a random timestep for each image
        timesteps = mint.rand(
            batch_size,
        )
        # Sample a random mask probability for each image using timestep and cosine schedule
        mask_prob = mask_schedule(timesteps)
        mask_prob = mask_prob.clip(config.training.min_masking_rate)

    # creat a random mask for each image
    num_token_masked = mint.clamp((seq_len * mask_prob).round(), min=1)

    mask_contiguous_region_prob = config.training.get("mask_contiguous_region_prob", None)

    if mask_contiguous_region_prob is None:
        mask_contiguous_region = False
    else:
        mask_contiguous_region = random.random() < mask_contiguous_region_prob

    if not mask_contiguous_region:
        batch_randperm = mint.argsort(mint.rand(batch_size, seq_len), dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
    else:
        resolution = int(seq_len**0.5)
        mask = mint.zeros((batch_size, resolution, resolution))

        # TODO - would be nice to vectorize
        for batch_idx, num_token_masked_ in enumerate(num_token_masked):
            num_token_masked_ = int(num_token_masked_.item())

            # NOTE: a bit handwavy with the bounds but gets a rectangle of ~num_token_masked_
            num_token_masked_height = random.randint(
                math.ceil(num_token_masked_ / resolution), min(resolution, num_token_masked_)
            )
            num_token_masked_height = min(num_token_masked_height, resolution)

            num_token_masked_width = math.ceil(num_token_masked_ / num_token_masked_height)
            num_token_masked_width = min(num_token_masked_width, resolution)

            start_idx_height = random.randint(0, resolution - num_token_masked_height)
            start_idx_width = random.randint(0, resolution - num_token_masked_width)

            mask[
                batch_idx,
                start_idx_height : start_idx_height + num_token_masked_height,
                start_idx_width : start_idx_width + num_token_masked_width,
            ] = 1

        mask = mask.reshape(batch_size, seq_len)
        mask = mask.to(ms.bool_)

    # mask images and create input and labels
    if config.training.get("noise_type", "mask"):
        input_ids = mint.where(mask, mask_id, image_tokens)
    elif config.training.get("noise_type", "random_replace"):
        # sample random tokens from the vocabulary
        random_tokens = mint.randint_like(image_tokens, low=0, high=config.model.codebook_size)
        input_ids = mint.where(mask, random_tokens, image_tokens)
    else:
        raise ValueError(f"noise_type {config.training.noise_type} not supported")

    if (
        config.training.get("predict_all_tokens", False)
        or config.training.get("noise_type", "mask") == "random_replace"
    ):
        labels = image_tokens
        loss_weight = get_loss_weight(mask_prob, mask.to(ms.int64))
    else:
        labels = mint.where(mask, image_tokens, -100)
        loss_weight = None

    if not is_train and seed is not None:
        # 恢复随机状态
        ms.set_rng_state(rng_state)

        random.setstate(python_rng_state)

    return input_ids, labels, loss_weight, mask_prob


##################################################
#              misc
##################################################
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


import mindspore.dataset.vision as transforms
from mindspore.dataset.vision import Inter


def image_transform(image, resolution=256, normalize=True):
    image = transforms.Resize(resolution, interpolation=Inter.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = (image - 0.5) / 0.5
    return image


def image_transform_squash(image, resolution=256, normalize=True):
    image = transforms.Resize((resolution, resolution), interpolation=Inter.BICUBIC)(image)
    image = transforms.ToTensor()(image)
    if normalize:
        image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(image)
    return image
