"""Utility"""
import os
import ssl
import urllib
import urllib.error
import urllib.request
from copy import deepcopy
from typing import Callable, Dict, Optional

from mindspore import load_checkpoint, load_param_into_net

# The default root directory where we save downloaded files.
# Use Get/Set to R/W this variable.
_DEFAULT_DOWNLOAD_ROOT = os.path.join(os.path.expanduser("~"), ".mindspore")


def get_default_download_root():
    return deepcopy(_DEFAULT_DOWNLOAD_ROOT)


def set_default_download_root(path):
    global _DEFAULT_DOWNLOAD_ROOT
    _DEFAULT_DOWNLOAD_ROOT = path


def get_checkpoint_download_root():
    return os.path.join(get_default_download_root(), "models")


class Download:
    """Base utility class for downloading."""

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
