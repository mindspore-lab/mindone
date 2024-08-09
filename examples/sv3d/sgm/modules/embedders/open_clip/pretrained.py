# reference to https://github.com/mlfoundations/open_clip

import hashlib
import os
import urllib
import warnings
from functools import partial
from typing import Dict, Union

from tqdm import tqdm

try:
    from huggingface_hub import hf_hub_download

    hf_hub_download = partial(hf_hub_download, library_name="open_clip")
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


def _pcfg(url="", hf_hub="", mean=None, std=None):
    return dict(
        url=url,
        hf_hub=hf_hub,
        mean=mean,
        std=std,
    )


def _clean_tag(tag: str):
    # normalize pretrained tags
    return tag.lower().replace("-", "_")


_PRETRAINED = {
    "ViT-H-14": dict(
        laion2b_s32b_b79k=_pcfg(hf_hub="laion/CLIP-ViT-H-14-laion2B-s32B-b79K/"),
    ),
    "ViT-H-14-Text": dict(),
}


def list_pretrained_tags_by_model(model: str):
    """return all pretrain tags for the specified model architecture"""
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def is_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return False
    return _clean_tag(tag) in _PRETRAINED[model]


def get_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return {}
    model_pretrained = _PRETRAINED[model]
    return model_pretrained.get(_clean_tag(tag), {})


def download_pretrained_from_url(
    url: str,
    cache_dir: Union[str, None] = None,
):
    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if "openaipublic" in url:
        expected_sha256 = url.split("/")[-2]
    elif "mlfoundations" in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ""

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match. re-downloading the file"
                )
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit="iB", unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(
        expected_sha256
    ):
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            "Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`."
        )
    return _has_hf_hub


def download_pretrained_from_hf(
    model_id: str,
    filename: str = "open_clip_pytorch_model.bin",
    revision=None,
    cache_dir: Union[str, None] = None,
):
    has_hf_hub(True)
    cached_file = hf_hub_download(model_id, filename, revision=revision, cache_dir=cache_dir)
    return cached_file


def download_pretrained(
    cfg: Dict,
    force_hf_hub: bool = False,
    cache_dir: Union[str, None] = None,
):
    target = ""
    if not cfg:
        return target

    download_url = cfg.get("url", "")
    download_hf_hub = cfg.get("hf_hub", "")
    if download_hf_hub and force_hf_hub:
        # use HF hub even if url exists
        download_url = ""

    if download_url:
        target = download_pretrained_from_url(download_url, cache_dir=cache_dir)
    elif download_hf_hub:
        has_hf_hub(True)
        # we assume the hf_hub entries in pretrained config combine model_id + filename in
        # 'org/model_name/filename.pt' form. To specify just the model id w/o filename and
        # use 'open_clip_pytorch_model.bin' default, there must be a trailing slash 'org/model_name/'.
        model_id, filename = os.path.split(download_hf_hub)
        if filename:
            target = download_pretrained_from_hf(model_id, filename=filename, cache_dir=cache_dir)
        else:
            target = download_pretrained_from_hf(model_id, cache_dir=cache_dir)

    return target
