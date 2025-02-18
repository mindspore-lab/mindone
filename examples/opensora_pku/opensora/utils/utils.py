import argparse
import collections
import html
import json
import logging
import random
import re
import urllib.parse as ul
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

import mindspore as ms

try:
    from bs4 import BeautifulSoup
except ImportError:
    is_bs4_available = False

try:
    import ftfy
except ImportError:
    is_ftfy_available = False

logger = logging.getLogger(__name__)


# Custom JSON Encoder to serialize everything as strings
class StringifyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Convert the object to a string
        return str(obj)


def save_diffusers_json(config, filename):
    if not isinstance(config, dict):
        config = dict(config)

    # Save the regular dictionary to a JSON file using the custom encoder
    with open(filename, "w") as json_file:
        json.dump(config, json_file, cls=StringifyJSONEncoder, indent=4)
    logger.info(f"Save config file to {filename}")


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)


def explicit_uniform_sampling(T, n, rank, bsz):
    """
    Explicit Uniform Sampling with integer timesteps and MindSpore.

    Args:
        T (int): Maximum timestep value.
        n (int): Number of ranks (data parallel processes).
        rank (int): The rank of the current process (from 0 to n-1).
        bsz (int): Batch size, number of timesteps to return.

    Returns:
        ms.Tensor: A tensor of shape (bsz,) containing uniformly sampled integer timesteps
                      within the rank's interval.
    """
    interval_size = T / n  # Integer division to ensure boundaries are integers
    lower_bound = interval_size * rank - 0.5
    upper_bound = interval_size * (rank + 1) - 0.5
    sampled_timesteps = [round(random.uniform(lower_bound, upper_bound)) for _ in range(bsz)]

    # Uniformly sample within the rank's interval, returning integers
    sampled_timesteps = ms.Tensor([round(random.uniform(lower_bound, upper_bound)) for _ in range(bsz)], dtype=ms.int32)
    # sampled_timesteps = sampled_timesteps.long()
    return sampled_timesteps


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=ms.float32):
    sigmas = noise_scheduler.sigmas.to(dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps

    step_indices = [(schedule_timesteps == t).nonzero() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def get_experiment_dir(root_dir, args):
    # if args.pretrained is not None and 'Latte-XL-2-256x256.pt' not in args.pretrained:
    #     root_dir += '-WOPRE'
    if args.use_compile:
        root_dir += "-Compile"  # speedup by torch compile
    if args.attention_mode:
        root_dir += f"-{args.attention_mode.upper()}"
    # if args.enable_xformers_memory_efficient_attention:
    #     root_dir += '-Xfor'
    if args.gradient_checkpointing:
        root_dir += "-Gc"
    if args.mixed_precision:
        root_dir += f"-{args.mixed_precision.upper()}"
    root_dir += f"-{args.max_image_size}"
    return root_dir


def get_precision(mixed_precision):
    if mixed_precision == "bf16":
        dtype = ms.bfloat16
    elif mixed_precision == "fp16":
        dtype = ms.float16
    else:
        dtype = ms.float32
    return dtype


def process_key(key_val):
    """Processes a single key-value pair from the source data."""
    k, val = key_val
    val = val.detach().float().numpy().astype(np.float32)
    return k, ms.Parameter(ms.Tensor(val, dtype=ms.float32))


def load_torch_state_dict_to_ms_ckpt(ckpt_file, num_workers=8, exclude_prefix=None, include_prefix=None):
    import torch

    source_data = torch.load(ckpt_file, map_location="cpu", weights_only=True)
    if "state_dict" in source_data:
        source_data = source_data["state_dict"]
    if "ema" in source_data:
        source_data = source_data["ema"]

    if exclude_prefix is not None:
        if isinstance(exclude_prefix, str):
            exclude_prefix = [exclude_prefix]
        assert (
            isinstance(exclude_prefix, list)
            and len(exclude_prefix) > 0
            and isinstance(exclude_prefix[0], str)
            and len(exclude_prefix[0]) > 0
        )
    if exclude_prefix is not None and len(exclude_prefix) > 0:
        keys_to_remove = [key for key in source_data if any(key.startswith(prefix) for prefix in exclude_prefix)]
        for key in keys_to_remove:
            del source_data[key]

    if include_prefix is not None:
        if isinstance(include_prefix, str):
            include_prefix = [include_prefix]
        assert (
            isinstance(include_prefix, list)
            and len(include_prefix) > 0
            and isinstance(include_prefix[0], str)
            and len(include_prefix[0]) > 0
        )
    if include_prefix is not None and len(include_prefix) > 0:
        keys_to_retain = [key for key in source_data if any(key.startswith(prefix) for prefix in include_prefix)]
        for key in source_data.keys():
            if key not in keys_to_retain:
                del source_data[key]
    assert len(source_data.keys()), "state dict is empty!"
    # Use multiprocessing to process keys in parallel
    with Pool(processes=num_workers) as pool:
        target_data = dict(
            tqdm(pool.imap(process_key, source_data.items()), total=len(source_data), desc="Checkpoint Conversion")
        )

    return target_data


#################################################################################
#                          Pixart-alpha  Utils                                  #
#################################################################################


bad_punct_regex = re.compile(
    r"[" + "#®•©™&@·º½¾¿¡§~" + r"\)" + r"\(" + r"\]" + r"\[" + r"\}" + r"\{" + r"\|" + r"\\" + r"\/" + r"\*" + r"]{1,}"
)  # noqa


def text_preprocessing(text, support_Chinese=True):
    # The exact text cleaning as was in the training stage:
    text = clean_caption(text, support_Chinese=support_Chinese)
    text = clean_caption(text, support_Chinese=support_Chinese)
    return text


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def clean_caption(caption, support_Chinese=True):
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
    if not support_Chinese:
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)  # Chinese
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


def _check_cfgs_in_parser(cfgs: dict, parser: argparse.ArgumentParser):
    actions_dest = [action.dest for action in parser._actions]
    defaults_key = parser._defaults.keys()
    for k in cfgs.keys():
        if k not in actions_dest and k not in defaults_key:
            raise KeyError(f"{k} does not exist in ArgumentParser!")


def remove_invalid_characters(file_name):
    file_name = file_name.replace(" ", "-")
    valid_pattern = r"[^a-zA-Z0-9_.-]"
    cleaned_file_name = re.sub(valid_pattern, "-", file_name)
    return cleaned_file_name
