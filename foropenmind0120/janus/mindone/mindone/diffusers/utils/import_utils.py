# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import importlib.util
import operator as op
import os
import sys
from collections import OrderedDict
from itertools import chain
from types import ModuleType
from typing import Any, Union

from huggingface_hub.utils import is_jinja_available  # noqa: F401
from packaging.version import Version, parse

from . import logging

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

_mindspore_version = "N/A"
_mindspore_available = importlib.util.find_spec("mindspore") is not None
if _mindspore_available:
    try:
        _mindspore_version = importlib_metadata.version("mindspore")
        logger.info(f"MindSpore version {_mindspore_version} available.")
    except importlib_metadata.PackageNotFoundError:
        _mindspore_available = False


_transformers_available = importlib.util.find_spec("transformers") is not None
try:
    _transformers_version = importlib_metadata.version("transformers")
    logger.debug(f"Successfully imported transformers version {_transformers_version}")
except importlib_metadata.PackageNotFoundError:
    _transformers_available = False

# (sayakpaul): importlib.util.find_spec("opencv-python") returns None even when it's installed.
# _opencv_available = importlib.util.find_spec("opencv-python") is not None
try:
    candidates = (
        "opencv-python",
        "opencv-contrib-python",
        "opencv-python-headless",
        "opencv-contrib-python-headless",
    )
    _opencv_version = None
    for pkg in candidates:
        try:
            _opencv_version = importlib_metadata.version(pkg)
            break
        except importlib_metadata.PackageNotFoundError:
            pass
    _opencv_available = _opencv_version is not None
    if _opencv_available:
        logger.debug(f"Successfully imported cv2 version {_opencv_version}")
except importlib_metadata.PackageNotFoundError:
    _opencv_available = False


_scipy_available = importlib.util.find_spec("scipy") is not None
try:
    _scipy_version = importlib_metadata.version("scipy")
    logger.debug(f"Successfully imported scipy version {_scipy_version}")
except importlib_metadata.PackageNotFoundError:
    _scipy_available = False


_ftfy_available = importlib.util.find_spec("ftfy") is not None
try:
    _ftfy_version = importlib_metadata.version("ftfy")
    logger.debug(f"Successfully imported ftfy version {_ftfy_version}")
except importlib_metadata.PackageNotFoundError:
    _ftfy_available = False


_bs4_available = importlib.util.find_spec("bs4") is not None
try:
    # importlib metadata under different name
    _bs4_version = importlib_metadata.version("beautifulsoup4")
    logger.debug(f"Successfully imported ftfy version {_bs4_version}")
except importlib_metadata.PackageNotFoundError:
    _bs4_available = False


_invisible_watermark_available = importlib.util.find_spec("imwatermark") is not None
try:
    _invisible_watermark_version = importlib_metadata.version("invisible-watermark")
    logger.debug(f"Successfully imported invisible-watermark version {_invisible_watermark_version}")
except importlib_metadata.PackageNotFoundError:
    _invisible_watermark_available = False


_sentencepiece_available = importlib.util.find_spec("sentencepiece") is not None
try:
    _sentencepiece_version = importlib_metadata.version("sentencepiece")
    logger.info(f"Successfully imported sentencepiece version {_sentencepiece_version}")
except importlib_metadata.PackageNotFoundError:
    _sentencepiece_available = False


_matplotlib_available = importlib.util.find_spec("matplotlib") is not None
try:
    _matplotlib_version = importlib_metadata.version("matplotlib")
    logger.debug(f"Successfully imported matplotlib version {_matplotlib_version}")
except importlib_metadata.PackageNotFoundError:
    _matplotlib_available = False


_imageio_available = importlib.util.find_spec("imageio") is not None
if _imageio_available:
    try:
        _imageio_version = importlib_metadata.version("imageio")
        logger.debug(f"Successfully imported imageio version {_imageio_version}")

    except importlib_metadata.PackageNotFoundError:
        _imageio_available = False


def is_transformers_available():
    return _transformers_available


def is_opencv_available():
    return _opencv_available


def is_scipy_available():
    return _scipy_available


def is_ftfy_available():
    return _ftfy_available


def is_bs4_available():
    return _bs4_available


def is_matplotlib_available():
    return _matplotlib_available


def is_imageio_available():
    return _imageio_available


def is_invisible_watermark_available():
    return _invisible_watermark_available


def is_sentencepiece_available():
    return _sentencepiece_available


# docstyle-ignore
OPENCV_IMPORT_ERROR = """
{0} requires the OpenCV library but it was not found in your environment. You can install it with pip: `pip
install opencv-python`
"""


# docstyle-ignore
SCIPY_IMPORT_ERROR = """
{0} requires the scipy library but it was not found in your environment. You can install it with pip: `pip install
scipy`
"""


# docstyle-ignore
TRANSFORMERS_IMPORT_ERROR = """
{0} requires the transformers library but it was not found in your environment. You can install it with pip: `pip
install transformers`
"""


# docstyle-ignore
BS4_IMPORT_ERROR = """
{0} requires the Beautiful Soup library but it was not found in your environment. You can install it with pip:
`pip install beautifulsoup4`. Please note that you may need to restart your runtime after installation.
"""

# docstyle-ignore
FTFY_IMPORT_ERROR = """
{0} requires the ftfy library but it was not found in your environment. Checkout the instructions on the
installation section: https://github.com/rspeer/python-ftfy/tree/master#installing and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
"""


# docstyle-ignore
IMAGEIO_IMPORT_ERROR = """
{0} requires the imageio library and ffmpeg but it was not found in your environment. You can install it with pip: `pip install imageio imageio-ffmpeg`
"""

# docstyle-ignore
INVISIBLE_WATERMARK_IMPORT_ERROR = """
{0} requires the invisible-watermark library but it was not found in your environment. You can install it with pip: `pip install invisible-watermark>=0.2.0`
"""

# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the sentencepiece library but it was not found in your environment. You can install it with pip: `pip install sentencepiece`
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        ("opencv", (is_opencv_available, OPENCV_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("transformers", (is_transformers_available, TRANSFORMERS_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        ("imageio", (is_imageio_available, IMAGEIO_IMPORT_ERROR)),
        ("invisible_watermark", (is_invisible_watermark_available, INVISIBLE_WATERMARK_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
    ]
)


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Args:
    Compares a library version to some requirement using a given operation.
        library_or_version (`str` or `packaging.version.Version`):
            A library name or a version to check.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`.
        requirement_version (`str`):
            The version to compare the library version against
    """
    if operation not in STR_OPERATION_TO_FUNC.keys():
        raise ValueError(f"`operation` must be one of {list(STR_OPERATION_TO_FUNC.keys())}, received {operation}")
    operation = STR_OPERATION_TO_FUNC[operation]
    if isinstance(library_or_version, str):
        library_or_version = parse(importlib_metadata.version(library_or_version))
    return operation(library_or_version, parse(requirement_version))


def is_mindspore_version(operation: str, version: str):
    """
    Args:
    Compares the current MindSpore version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of MindSpore
    """
    return compare_versions(parse(_mindspore_version), operation, version)


def is_peft_version(operation: str, version: str):
    """
    Args:
    Compares the current PEFT version to a given reference with an operation.
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    from mindone.diffusers._peft import __version__ as _peft_version

    if not _peft_version:
        return False
    return compare_versions(parse(_peft_version), operation, version)


def maybe_import_module_in_mindone(module_name: str, force_original: bool = False):
    if force_original:
        return importlib.import_module(module_name)

    if module_name.startswith("diffusers") or module_name.startswith("transformers"):
        return importlib.import_module(f"mindone.{module_name}")
    else:
        return importlib.import_module(module_name)


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(self, name, module_file, import_structure, module_spec=None, extra_objects=None):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
        return result

    def __getattr__(self, name: str) -> Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
