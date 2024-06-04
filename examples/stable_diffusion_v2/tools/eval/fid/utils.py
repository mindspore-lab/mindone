"""Utility"""
import bz2
import gzip
import hashlib
import logging
import os
import pathlib
import ssl
import tarfile
import urllib
import urllib.error
import urllib.request
import zipfile
from copy import deepcopy
from typing import Callable, Dict, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from mindspore import load_checkpoint, load_param_into_net

_logger = logging.getLogger(__name__)
# The default root directory where we save downloaded files.
# Use Get/Set to R/W this variable.
_DEFAULT_DOWNLOAD_ROOT = os.path.join(os.path.expanduser("~"), ".mindspore")

IMAGE_EXTENSIONS = {"bmp", "jpg", "jpeg", "pgm", "png", "ppm", "tif", "tiff", "webp", "JPEG"}


def get_default_download_root():
    return deepcopy(_DEFAULT_DOWNLOAD_ROOT)


def set_default_download_root(path):
    global _DEFAULT_DOWNLOAD_ROOT
    _DEFAULT_DOWNLOAD_ROOT = path


def get_checkpoint_download_root():
    return os.path.join(get_default_download_root(), "models")


FILE_TYPE_ALIASES = {
    ".tbz": (".tar", ".bz2"),
    ".tbz2": (".tar", ".bz2"),
    ".tgz": (".tar", ".gz"),
}

ARCHIVE_TYPE_SUFFIX = [
    ".tar",
    ".zip",
]

COMPRESS_TYPE_SUFFIX = [
    ".bz2",
    ".gz",
]


def detect_file_type(filename: str):  # pylint: disable=inconsistent-return-statements
    """Detect file type by suffixes and return tuple(suffix, archive_type, compression)."""
    suffixes = pathlib.Path(filename).suffixes
    if not suffixes:
        raise RuntimeError(f"File `{filename}` has no suffixes that could be used to detect.")
    suffix = suffixes[-1]

    # Check if the suffix is a known alias.
    if suffix in FILE_TYPE_ALIASES:
        return suffix, FILE_TYPE_ALIASES[suffix][0], FILE_TYPE_ALIASES[suffix][1]

    # Check if the suffix is an archive type.
    if suffix in ARCHIVE_TYPE_SUFFIX:
        return suffix, suffix, None

    # Check if the suffix is a compression.
    if suffix in COMPRESS_TYPE_SUFFIX:
        # Check for suffix hierarchy.
        if len(suffixes) > 1:
            suffix2 = suffixes[-2]
            # Check if the suffix2 is an archive type.
            if suffix2 in ARCHIVE_TYPE_SUFFIX:
                return suffix2 + suffix, suffix2, suffix
        return suffix, None, suffix


class Download:
    """Base utility class for downloading."""

    USER_AGENT: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/92.0.4515.131 Safari/537.36"
    )

    @staticmethod
    def calculate_md5(file_path: str, chunk_size: int = 1024 * 1024) -> str:
        """Calculate md5 value."""
        md5 = hashlib.md5()
        with open(file_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def check_md5(self, file_path: str, md5: Optional[str] = None) -> bool:
        """Check md5 value."""
        return md5 == self.calculate_md5(file_path)

    @staticmethod
    def extract_tar(from_path: str, to_path: Optional[str] = None, compression: Optional[str] = None) -> None:
        """Extract tar format file."""

        with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
            tar.extractall(to_path)

    @staticmethod
    def extract_zip(from_path: str, to_path: Optional[str] = None, compression: Optional[str] = None) -> None:
        """Extract zip format file."""

        compression_mode = zipfile.ZIP_BZIP2 if compression else zipfile.ZIP_STORED
        with zipfile.ZipFile(from_path, "r", compression=compression_mode) as zip_file:
            zip_file.extractall(to_path)

    def extract_archive(self, from_path: str, to_path: str = None) -> str:
        """Extract and  archive from path to path."""
        archive_extractors = {
            ".tar": self.extract_tar,
            ".zip": self.extract_zip,
        }
        compress_file_open = {
            ".bz2": bz2.open,
            ".gz": gzip.open,
        }

        if not to_path:
            to_path = os.path.dirname(from_path)

        suffix, archive_type, compression = detect_file_type(from_path)  # pylint: disable=unused-variable

        if not archive_type:
            to_path = from_path.replace(suffix, "")
            compress = compress_file_open[compression]
            with compress(from_path, "rb") as rf, open(to_path, "wb") as wf:
                wf.write(rf.read())
            return to_path

        extractor = archive_extractors[archive_type]
        extractor(from_path, to_path, compression)

        return to_path

    def download_file(self, url: str, file_path: str, chunk_size: int = 1024):
        """Download a file."""

        # no check certificate
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # Define request headers.
        headers = {"User-Agent": self.USER_AGENT}

        _logger.info(f"Downloading from {url} to {file_path} ...")
        with open(file_path, "wb") as f:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, context=ctx) as response:
                with tqdm(total=response.length, unit="B") as pbar:
                    for chunk in iter(lambda: response.read(chunk_size), b""):
                        if not chunk:
                            break
                        pbar.update(chunk_size)
                        f.write(chunk)

    def download_url(
        self,
        url: str,
        path: Optional[str] = None,
        filename: Optional[str] = None,
        md5: Optional[str] = None,
    ):
        """Download a file from a url and place it in root."""
        if path is None:
            path = get_default_download_root()
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)

        if not filename:
            filename = os.path.basename(url)

        file_path = os.path.join(path, filename)

        # Check if the file is exists.
        if os.path.isfile(file_path):
            if not md5 or self.check_md5(file_path, md5):
                return file_path

        # Download the file.
        try:
            self.download_file(url, file_path)
        except (urllib.error.URLError, IOError) as e:
            if url.startswith("https"):
                url = url.replace("https", "http")
                try:
                    self.download_file(url, file_path)
                except (urllib.error.URLError, IOError):
                    # pylint: disable=protected-access
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.download_file(url, file_path)
                    ssl._create_default_https_context = ssl.create_default_context
            else:
                raise e

        return file_path

    def download_and_extract_archive(
        self,
        url: str,
        download_path: Optional[str] = None,
        extract_path: Optional[str] = None,
        filename: Optional[str] = None,
        md5: Optional[str] = None,
        remove_finished: bool = False,
    ) -> None:
        """Download and extract archive."""
        if download_path is None:
            download_path = get_default_download_root()
        download_path = os.path.expanduser(download_path)

        if not filename:
            filename = os.path.basename(url)

        self.download_url(url, download_path, filename, md5)

        archive = os.path.join(download_path, filename)
        self.extract_archive(archive, extract_path)

        if remove_finished:
            os.remove(archive)


def download_model(url):
    """Download the pretrained ckpt from url to local path"""

    # download files
    download_path = get_checkpoint_download_root()
    os.makedirs(download_path, exist_ok=True)
    file_path = Download().download_url(url, path=download_path)
    return file_path


def load_model(
    network,
    load_from: Optional[str] = None,
    filter_fn: Optional[Callable[[Dict], Dict]] = None,
):
    """
    Load the checkpoint into the model

    Args:
        network: network
        load_from: a string that can be url or local path to a checkpoint, that will be loaded to the network.
        filter_fn: a function filtering the parameters that will be loading into the network. If it is None,
            all parameters will be loaded.
    """
    if load_from is None:
        return

    if load_from[:4] == "http":
        local_ckpt_path = download_model(load_from)
    else:
        local_ckpt_path = load_from

    assert local_ckpt_path and os.path.exists(local_ckpt_path), (
        f"Failed to load checkpoint. `{local_ckpt_path}` NOT exist. \n"
        "Please check the path and set it in `eval-ckpt_load_path` or `model-pretrained` in the yaml config file "
    )

    params = load_checkpoint(local_ckpt_path)

    if filter_fn is not None:
        params = filter_fn(params)

    load_param_into_net(network, params)


def get_image_paths(img_dir):
    path = pathlib.Path(img_dir)
    files = sorted([file for ext in IMAGE_EXTENSIONS for file in path.glob("*.{}".format(ext))])

    return files


def compute_torchmetric_fid(gen_imgs, gt_imgs):
    import torch
    import torchmetrics as tm

    real_images = [np.array(Image.open(path).convert("RGB")) for path in gt_imgs]
    fake_images = [np.array(Image.open(path).convert("RGB")) for path in gen_imgs]

    def preprocess_image(image):
        image = cv2.resize(image, (299, 299))
        image = torch.tensor(image).unsqueeze(0)
        image = image.permute(0, 3, 1, 2) / 255.0
        return image

    real_images = torch.cat([preprocess_image(image) for image in real_images])
    # print(real_images.shape)
    fake_images = torch.cat([preprocess_image(image) for image in fake_images])
    # print(fake_images.shape)

    # torch
    fid = tm.image.fid.FrechetInceptionDistance(normalize=True)

    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)

    return float(fid.compute())
