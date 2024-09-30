"""Utility"""
import hashlib
import os
import shutil
import ssl
import urllib
import urllib.error
import urllib.request
from copy import deepcopy

from tqdm import tqdm

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


def verify_sha256(path, sha):
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            sha256.update(data)

    return sha256.hexdigest()[:8] == sha


def locate_model(model_name="nsfw", backend="ms"):
    path = get_checkpoint_download_root()
    extension = ".ckpt" if backend == "ms" else ".pth"
    if not os.path.exists(path):
        os.makedirs(path)
    for file_name in os.listdir(path):
        if model_name in file_name and file_name.endswith(extension):
            file_path = os.path.join(path, file_name)
            break
    else:
        if backend == "ms":
            url = (
                "https://download.mindspore.cn/toolkits/mindone/stable_diffusion/safety_checker/l14_nsfw-c7c99ae7.ckpt"
            )
        else:
            url = "https://github.com/LAION-AI/CLIP-based-NSFW-Detector/files/10250461/clip_autokeras_binary_nsfw.zip"
        file_name = url.split("/")[-1]
        file_path = os.path.join(path, file_name)

        # no check certificate
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        # Define request headers.
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/92.0.4515.131 Safari/537.36"
            )
        }

        chunk_size = 1024 * 1024
        print(f"Downloading from {url} to {file_path} ...")
        with open(file_path, "wb") as f:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, context=ctx) as response:
                with tqdm(total=response.length, unit="B") as pbar:
                    for chunk in iter(lambda: response.read(chunk_size), b""):
                        if not chunk:
                            break
                        pbar.update(chunk_size)
                        f.write(chunk)

        if backend == "ms":
            if not verify_sha256(file_path, file_name.split("-")[-1][:-5]):
                os.remove(file_path)
                raise ConnectionError("sha256 not matched, cannot download model weights")
        else:
            shutil.unpack_archive(file_path, path)
            os.remove(file_path)

    return file_path
