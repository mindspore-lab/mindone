# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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
import inspect
import operator as op
import os
import sys
from collections import OrderedDict, defaultdict
from itertools import chain
from types import ModuleType
from typing import Any, Tuple, Union

from huggingface_hub.utils import is_jinja_available  # noqa: F401
from packaging.version import Version, parse

from . import logging

# The package importlib_metadata is in a different place, depending on the python version.
if sys.version_info < (3, 8):
    import importlib_metadata
else:
    import importlib.metadata as importlib_metadata
try:
    _package_map = importlib_metadata.packages_distributions()  # load-once to avoid expensive calls
except Exception:
    _package_map = None

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_MINDSPORE = os.environ.get("USE_MINDSPORE", "AUTO").upper()
USE_SAFETENSORS = os.environ.get("USE_SAFETENSORS", "AUTO").upper()
DIFFUSERS_SLOW_IMPORT = os.environ.get("DIFFUSERS_SLOW_IMPORT", "FALSE").upper()
DIFFUSERS_SLOW_IMPORT = DIFFUSERS_SLOW_IMPORT in ENV_VARS_TRUE_VALUES

STR_OPERATION_TO_FUNC = {">": op.gt, ">=": op.ge, "==": op.eq, "!=": op.ne, "<=": op.le, "<": op.lt}

_is_google_colab = "google.colab" in sys.modules or any(k.startswith("COLAB_") for k in os.environ)


def _is_package_available(pkg_name: str, get_dist_name: bool = False) -> Tuple[bool, str]:
    global _package_map
    pkg_exists = importlib.util.find_spec(pkg_name) is not None
    pkg_version = "N/A"

    if pkg_exists:
        if _package_map is None:
            _package_map = defaultdict(list)
            try:
                # Fallback for Python < 3.10
                for dist in importlib_metadata.distributions():
                    _top_level_declared = (dist.read_text("top_level.txt") or "").split()
                    _infered_opt_names = {
                        f.parts[0] if len(f.parts) > 1 else inspect.getmodulename(f) for f in (dist.files or [])
                    } - {None}
                    _top_level_inferred = filter(lambda name: "." not in name, _infered_opt_names)
                    for pkg in _top_level_declared or _top_level_inferred:
                        _package_map[pkg].append(dist.metadata["Name"])
            except Exception as _:  # noqa
                pass
        try:
            if get_dist_name and pkg_name in _package_map and _package_map[pkg_name]:
                if len(_package_map[pkg_name]) > 1:
                    logger.warning(
                        f"Multiple distributions found for package {pkg_name}. Picked distribution: {_package_map[pkg_name][0]}"
                    )
                pkg_name = _package_map[pkg_name][0]
            pkg_version = importlib_metadata.version(pkg_name)
            logger.debug(f"Successfully imported {pkg_name} version {pkg_version}")
        except (ImportError, importlib_metadata.PackageNotFoundError):
            pkg_exists = False

    return pkg_exists, pkg_version


if USE_MINDSPORE in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _mindspore_available, _mindspore_version = _is_package_available("mindspore")

else:
    logger.info("Disabling MindSpore because USE_MINDSPORE is set")
    _mindspore_available = False
    _mindspore_version = "N/A"

if USE_SAFETENSORS in ENV_VARS_TRUE_AND_AUTO_VALUES:
    _safetensors_available, _safetensors_version = _is_package_available("safetensors")

else:
    logger.info("Disabling Safetensors because USE_SAFETENSORS is set")
    _safetensors_available = False

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

_transformers_available, _transformers_version = _is_package_available("transformers")
_hf_hub_available, _hf_hub_version = _is_package_available("huggingface_hub")
_kernels_available, _kernels_version = _is_package_available("kernels")
_inflect_available, _inflect_version = _is_package_available("inflect")
_unidecode_available, _unidecode_version = _is_package_available("unidecode")
_note_seq_available, _note_seq_version = _is_package_available("note_seq")
_wandb_available, _wandb_version = _is_package_available("wandb")
_tensorboard_available, _tensorboard_version = _is_package_available("tensorboard")
_sentencepiece_available, _sentencepiece_version = _is_package_available("sentencepiece")
_matplotlib_available, _matplotlib_version = _is_package_available("matplotlib")
_imageio_available, _imageio_version = _is_package_available("imageio")
_ftfy_available, _ftfy_version = _is_package_available("ftfy")
_scipy_available, _scipy_version = _is_package_available("scipy")
_librosa_available, _librosa_version = _is_package_available("librosa")
_better_profanity_available, _better_profanity_version = _is_package_available("better_profanity")
_nltk_available, _nltk_version = _is_package_available("nltk")


def is_mindspore_available():
    return _mindspore_available


def is_transformers_available():
    return _transformers_available


def is_inflect_available():
    return _inflect_available


def is_unidecode_available():
    return _unidecode_available


def is_opencv_available():
    return _opencv_available


def is_scipy_available():
    return _scipy_available


def is_librosa_available():
    return _librosa_available


def is_kernels_available():
    return _kernels_available


def is_note_seq_available():
    return _note_seq_available


def is_wandb_available():
    return _wandb_available


def is_tensorboard_available():
    return _tensorboard_available


def is_ftfy_available():
    return _ftfy_available


def is_bs4_available():
    return _bs4_available


def is_invisible_watermark_available():
    return _invisible_watermark_available


def is_matplotlib_available():
    return _matplotlib_available


def is_safetensors_available():
    return _safetensors_available


def is_google_colab():
    return _is_google_colab


def is_sentencepiece_available():
    return _sentencepiece_available


def is_imageio_available():
    return _imageio_available


def is_better_profanity_available():
    return _better_profanity_available


def is_nltk_available():
    return _nltk_available


# docstyle-ignore
INFLECT_IMPORT_ERROR = """
{0} requires the inflect library but it was not found in your environment. You can install it with pip: `pip install
inflect`
"""

# docstyle-ignore
MINDSPORE_IMPORT_ERROR = """
{0} requires the MindSpore library but it was not found in your environment. Checkout the instructions on the
installation page: https://www.mindspore.cn/install/ and follow the ones that match your environment.
"""

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
LIBROSA_IMPORT_ERROR = """
{0} requires the librosa library but it was not found in your environment.  Checkout the instructions on the
installation page: https://librosa.org/doc/latest/install.html and follow the ones that match your environment.
"""

# docstyle-ignore
TRANSFORMERS_IMPORT_ERROR = """
{0} requires the transformers library but it was not found in your environment. You can install it with pip: `pip
install transformers`
"""

# docstyle-ignore
UNIDECODE_IMPORT_ERROR = """
{0} requires the unidecode library but it was not found in your environment. You can install it with pip: `pip install
Unidecode`
"""

# docstyle-ignore
NOTE_SEQ_IMPORT_ERROR = """
{0} requires the note-seq library but it was not found in your environment. You can install it with pip: `pip
install note-seq`
"""

# docstyle-ignore
WANDB_IMPORT_ERROR = """
{0} requires the wandb library but it was not found in your environment. You can install it with pip: `pip
install wandb`
"""

# docstyle-ignore
TENSORBOARD_IMPORT_ERROR = """
{0} requires the tensorboard library but it was not found in your environment. You can install it with pip: `pip
install tensorboard`
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
INVISIBLE_WATERMARK_IMPORT_ERROR = """
{0} requires the invisible-watermark library but it was not found in your environment. You can install it with pip: `pip install invisible-watermark>=0.2.0`
"""

# docstyle-ignore
SAFETENSORS_IMPORT_ERROR = """
{0} requires the safetensors library but it was not found in your environment. You can install it with pip: `pip install safetensors`
"""

# docstyle-ignore
SENTENCEPIECE_IMPORT_ERROR = """
{0} requires the sentencepiece library but it was not found in your environment. You can install it with pip: `pip install sentencepiece`
"""

# docstyle-ignore
IMAGEIO_IMPORT_ERROR = """
{0} requires the imageio library and ffmpeg but it was not found in your environment. You can install it with pip: `pip install imageio imageio-ffmpeg`
"""

# docstyle-ignore
BETTER_PROFANITY_IMPORT_ERROR = """
{0} requires the better_profanity library but it was not found in your environment. You can install it with pip: `pip install better_profanity`
"""

# docstyle-ignore
NLTK_IMPORT_ERROR = """
{0} requires the nltk library but it was not found in your environment. You can install it with pip: `pip install nltk`
"""


BACKENDS_MAPPING = OrderedDict(
    [
        ("bs4", (is_bs4_available, BS4_IMPORT_ERROR)),
        ("inflect", (is_inflect_available, INFLECT_IMPORT_ERROR)),
        ("opencv", (is_opencv_available, OPENCV_IMPORT_ERROR)),
        ("scipy", (is_scipy_available, SCIPY_IMPORT_ERROR)),
        ("transformers", (is_transformers_available, TRANSFORMERS_IMPORT_ERROR)),
        ("unidecode", (is_unidecode_available, UNIDECODE_IMPORT_ERROR)),
        ("librosa", (is_librosa_available, LIBROSA_IMPORT_ERROR)),
        ("note_seq", (is_note_seq_available, NOTE_SEQ_IMPORT_ERROR)),
        ("wandb", (is_wandb_available, WANDB_IMPORT_ERROR)),
        ("tensorboard", (is_tensorboard_available, TENSORBOARD_IMPORT_ERROR)),
        ("ftfy", (is_ftfy_available, FTFY_IMPORT_ERROR)),
        ("invisible_watermark", (is_invisible_watermark_available, INVISIBLE_WATERMARK_IMPORT_ERROR)),
        ("safetensors", (is_safetensors_available, SAFETENSORS_IMPORT_ERROR)),
        ("sentencepiece", (is_sentencepiece_available, SENTENCEPIECE_IMPORT_ERROR)),
        ("imageio", (is_imageio_available, IMAGEIO_IMPORT_ERROR)),
        ("better_profanity", (is_better_profanity_available, BETTER_PROFANITY_IMPORT_ERROR)),
        ("nltk", (is_nltk_available, NLTK_IMPORT_ERROR)),
    ]
)


# This function was copied from: https://github.com/huggingface/accelerate/blob/874c4967d94badd24f893064cc3bef45f57cadf7/src/accelerate/utils/versions.py#L319
def compare_versions(library_or_version: Union[str, Version], operation: str, requirement_version: str):
    """
    Compares a library version to some requirement using a given operation.

    Args:
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
    Compares the current MindSpore version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A string version of MindSpore
    """
    return compare_versions(parse(_mindspore_version), operation, version)


def is_hf_hub_version(operation: str, version: str):
    """
    Compares the current Hugging Face Hub version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    if not _hf_hub_available:
        return False
    return compare_versions(parse(_hf_hub_version), operation, version)


def is_peft_version(operation: str, version: str):
    """
    Compares the current PEFT version to a given reference with an operation.

    Args:
        operation (`str`):
            A string representation of an operator, such as `">"` or `"<="`
        version (`str`):
            A version string
    """
    from mindone.peft import __version__ as _peft_version

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
