# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib.machinery
import importlib.metadata
import importlib.util
from collections import OrderedDict
from functools import lru_cache
from typing import Tuple, Union

# hf exists
from transformers.utils import logging

from ..mindspore_adapter.utils import _is_ascend

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# TODO: This doesn't work for all packages (`bs4`, `faiss`, etc.) Talk to Sylvain to see how to do with it better.
def _is_package_available(pkg_name: str, return_version: bool = False) -> Union[Tuple[bool, str], bool]:
    # Check if the package spec exists and grab its version to avoid importing a local directory
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            # Primary method to get the package version
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            if pkg_name == "mindspore":
                try:
                    package = importlib.import_module(pkg_name)
                    temp_version = getattr(package, "__version__", "N/A")
                    # Check if the version contains "dev"
                    if "dev" in temp_version:
                        package_version = temp_version
                        package_exists = True
                    else:
                        package_exists = False
                except ImportError:
                    # If the package can't be imported, it's not available
                    package_exists = False
            else:
                package_exists = False
        logger.debug(f"Detected {pkg_name} version: {package_version}")
    if return_version:
        return package_exists, package_version
    else:
        return package_exists


_av_available = importlib.util.find_spec("av") is not None
_decord_available = importlib.util.find_spec("decord") is not None
_pandas_available = _is_package_available("pandas")
_scipy_available = _is_package_available("scipy")
_cv2_available = importlib.util.find_spec("cv2") is not None
_yt_dlp_available = importlib.util.find_spec("yt_dlp") is not None
_soundfile_available = _is_package_available("soundfile")
_librosa_available = _is_package_available("librosa")
_pytesseract_available = _is_package_available("pytesseract")


def is_mindspore_available():
    _mindspore_available, _mindspore_version = _is_package_available("mindspore", return_version=True)
    return _mindspore_available


def get_mindspore_version():
    _mindspore_available, _mindspore_version = _is_package_available("mindspore", return_version=True)
    return _mindspore_version


def is_scipy_available():
    return _scipy_available


def is_av_available():
    return _av_available


def is_decord_available():
    return _decord_available


def is_cv2_available():
    return _cv2_available


def is_yt_dlp_available():
    return _yt_dlp_available


def is_pandas_available():
    return _pandas_available


@lru_cache
def is_vision_available():
    _pil_available = importlib.util.find_spec("PIL") is not None
    if _pil_available:
        try:
            package_version = importlib.metadata.version("Pillow")
        except importlib.metadata.PackageNotFoundError:
            try:
                package_version = importlib.metadata.version("Pillow-SIMD")
            except importlib.metadata.PackageNotFoundError:
                return False
        logger.debug(f"Detected PIL version {package_version}")
    return _pil_available


def is_pytesseract_available():
    return _pytesseract_available


# docstyle-ignore
AV_IMPORT_ERROR = """
{0} requires the PyAv library but it was not found in your environment. You can install it with:
```
pip install av
```
Please note that you may need to restart your runtime after installation.
"""

MINDSPORE_IMPORT_ERROR_WITH_TF = """
{0} requires the MindSpore library but it was not found in your environment.
However, we were able to find a TensorFlow installation. TensorFlow classes begin
with "TF", but are otherwise identically named to our MindSpore classes. This
means that the TF equivalent of the class you tried to import would be "TF{0}".
If you want to use TensorFlow, please use TF classes instead!
"""

PANDAS_IMPORT_ERROR = """
{0} requires the pandas library but it was not found in your environment. You can install it with pip as
explained here: https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html.
Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
PYTESSERACT_IMPORT_ERROR = """
{0} requires the PyTesseract library but it was not found in your environment. You can install it with pip:
`pip install pytesseract`. Please note that you may need to restart your runtime after installation.
"""

SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip:
`pip install scipy`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
VISION_IMPORT_ERROR = """
{0} requires the PIL library but it was not found in your environment. You can install it with pip:
`pip install pillow`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
CV2_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with:
```
pip install opencv-python
```
Please note that you may need to restart your runtime after installation.
"""

BACKENDS_MAPPING = OrderedDict(
    [
        ("av", (is_av_available, AV_IMPORT_ERROR)),
        ("mindspore", (is_mindspore_available, MINDSPORE_IMPORT_ERROR_WITH_TF)),
        ("pandas", (is_pandas_available, PANDAS_IMPORT_ERROR)),
        ("pytesseract", (is_pytesseract_available, PYTESSERACT_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("vision", (is_vision_available, VISION_IMPORT_ERROR)),
        ("cv2", (is_cv2_available, CV2_IMPORT_ERROR)),
    ]
)


def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]

    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


def is_flash_attn_2_available():
    if _is_ascend():
        return True

    return False


def is_sdpa_available():
    return False


def is_soundfile_available():
    return _soundfile_available


def is_librosa_available():
    return _librosa_available
