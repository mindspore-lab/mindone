# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
import importlib
import os
import re
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import requests
from huggingface_hub import DDUFEntry, ModelCard, model_info, snapshot_download
from huggingface_hub.utils import OfflineModeIsEnabled, validate_hf_hub_args
from packaging import version
from requests.exceptions import HTTPError

import mindspore as ms
from mindspore import nn

from mindone.transformers import MSPreTrainedModel

from ..utils import (
    FLAX_WEIGHTS_NAME,
    ONNX_EXTERNAL_WEIGHTS_NAME,
    ONNX_WEIGHTS_NAME,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    deprecate,
    get_class_from_dynamic_module,
    is_transformers_available,
    logging,
    maybe_import_module_in_mindone,
)
from .transformers_loading_utils import _load_tokenizer_from_dduf, _load_transformers_model_from_dduf

if is_transformers_available():
    import transformers
    from transformers import PreTrainedTokenizerBase
    from transformers.utils import FLAX_WEIGHTS_NAME as TRANSFORMERS_FLAX_WEIGHTS_NAME
    from transformers.utils import SAFE_WEIGHTS_NAME as TRANSFORMERS_SAFE_WEIGHTS_NAME
    from transformers.utils import WEIGHTS_NAME as TRANSFORMERS_WEIGHTS_NAME

INDEX_FILE = "diffusion_pytorch_model.bin"
CUSTOM_PIPELINE_FILE_NAME = "pipeline.py"
DUMMY_MODULES_FOLDER = "diffusers.utils"
TRANSFORMERS_DUMMY_MODULES_FOLDER = "transformers.utils"
CONNECTED_PIPES_KEYS = ["prior"]

logger = logging.get_logger(__name__)

LOADABLE_CLASSES = {
    "diffusers": {
        "ModelMixin": ["save_pretrained", "from_pretrained"],
        "SchedulerMixin": ["save_pretrained", "from_pretrained"],
        "DiffusionPipeline": ["save_pretrained", "from_pretrained"],
    },
    "transformers": {
        "MSPreTrainedModel": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizer": ["save_pretrained", "from_pretrained"],
        "PreTrainedTokenizerFast": ["save_pretrained", "from_pretrained"],
        "FeatureExtractionMixin": ["save_pretrained", "from_pretrained"],
        "ProcessorMixin": ["save_pretrained", "from_pretrained"],
        "ImageProcessingMixin": ["save_pretrained", "from_pretrained"],
    },
}

ALL_IMPORTABLE_CLASSES = {}
for library in LOADABLE_CLASSES:
    ALL_IMPORTABLE_CLASSES.update(LOADABLE_CLASSES[library])


def is_safetensors_compatible(filenames, passed_components=None, folder_names=None, variant=None) -> bool:
    """
    Checking for safetensors compatibility:
    - The model is safetensors compatible only if there is a safetensors file for each model component present in
      filenames.

    Converting default pytorch serialized filenames to safetensors serialized filenames:
    - For models from the diffusers library, just replace the ".bin" extension with ".safetensors"
    - For models from the transformers library, the filename changes from "pytorch_model" to "model", and the ".bin"
      extension is replaced with ".safetensors"
    """
    weight_names = [
        WEIGHTS_NAME,
        SAFETENSORS_WEIGHTS_NAME,
        FLAX_WEIGHTS_NAME,
        ONNX_WEIGHTS_NAME,
        ONNX_EXTERNAL_WEIGHTS_NAME,
    ]

    if is_transformers_available():
        weight_names += [TRANSFORMERS_WEIGHTS_NAME, TRANSFORMERS_SAFE_WEIGHTS_NAME, TRANSFORMERS_FLAX_WEIGHTS_NAME]

    # model_pytorch, diffusion_model_pytorch, ...
    weight_prefixes = [w.split(".")[0] for w in weight_names]
    # .bin, .safetensors, ...
    weight_suffixs = [w.split(".")[-1] for w in weight_names]
    # -00001-of-00002
    transformers_index_format = r"\d{5}-of-\d{5}"
    # `diffusion_pytorch_model.bin` as well as `model-00001-of-00002.safetensors`
    variant_file_re = re.compile(
        rf"({'|'.join(weight_prefixes)})\.({variant}|{variant}-{transformers_index_format})\.({'|'.join(weight_suffixs)})$"
    )
    non_variant_file_re = re.compile(
        rf"({'|'.join(weight_prefixes)})(-{transformers_index_format})?\.({'|'.join(weight_suffixs)})$"
    )

    passed_components = passed_components or []
    if folder_names:
        filenames = {f for f in filenames if os.path.split(f)[0] in folder_names}

    # extract all components of the pipeline and their associated files
    components = {}
    for filename in filenames:
        if not len(filename.split("/")) == 2:
            continue

        component, component_filename = filename.split("/")
        if component in passed_components:
            continue

        components.setdefault(component, [])
        components[component].append(component_filename)

    # If there are no component folders check the main directory for safetensors files
    filtered_filenames = set()
    if not components:
        if variant is not None:
            filtered_filenames = filter_with_regex(filenames, variant_file_re)

        # If no variant filenames exist check if non-variant files are available
        if not filtered_filenames:
            filtered_filenames = filter_with_regex(filenames, non_variant_file_re)
        return any(".safetensors" in filename for filename in filtered_filenames)

    # iterate over all files of a component
    # check if safetensor files exist for that component
    for component, component_filenames in components.items():
        matches = []
        filtered_component_filenames = set()
        # if variant is provided check if the variant of the safetensors exists
        if variant is not None:
            filtered_component_filenames = filter_with_regex(component_filenames, variant_file_re)

        # if variant safetensor files do not exist check for non-variants
        if not filtered_component_filenames:
            filtered_component_filenames = filter_with_regex(component_filenames, non_variant_file_re)
        for component_filename in filtered_component_filenames:
            filename, extension = os.path.splitext(component_filename)

            match_exists = extension == ".safetensors"
            matches.append(match_exists)

        if not any(matches):
            return False

    return True


def filter_model_files(filenames):
    """Filter model repo files for just files/folders that contain model weights"""
    weight_names = [
        WEIGHTS_NAME,
        SAFETENSORS_WEIGHTS_NAME,
        FLAX_WEIGHTS_NAME,
        ONNX_WEIGHTS_NAME,
        ONNX_EXTERNAL_WEIGHTS_NAME,
    ]

    if is_transformers_available():
        weight_names += [TRANSFORMERS_WEIGHTS_NAME, TRANSFORMERS_SAFE_WEIGHTS_NAME, TRANSFORMERS_FLAX_WEIGHTS_NAME]

    allowed_extensions = [wn.split(".")[-1] for wn in weight_names]

    return [f for f in filenames if any(f.endswith(extension) for extension in allowed_extensions)]


def filter_with_regex(filenames, pattern_re):
    return {f for f in filenames if pattern_re.match(f.split("/")[-1]) is not None}


def variant_compatible_siblings(filenames, variant=None, ignore_patterns=None) -> Union[List[os.PathLike], str]:
    weight_names = [
        WEIGHTS_NAME,
        SAFETENSORS_WEIGHTS_NAME,
        FLAX_WEIGHTS_NAME,
        ONNX_WEIGHTS_NAME,
        ONNX_EXTERNAL_WEIGHTS_NAME,
    ]

    if is_transformers_available():
        weight_names += [TRANSFORMERS_WEIGHTS_NAME, TRANSFORMERS_SAFE_WEIGHTS_NAME, TRANSFORMERS_FLAX_WEIGHTS_NAME]

    # model_pytorch, diffusion_model_pytorch, ...
    weight_prefixes = [w.split(".")[0] for w in weight_names]
    # .bin, .safetensors, ...
    weight_suffixs = [w.split(".")[-1] for w in weight_names]
    # -00001-of-00002
    transformers_index_format = r"\d{5}-of-\d{5}"

    if variant is not None:
        # `diffusion_pytorch_model.fp16.bin` as well as `model.fp16-00001-of-00002.safetensors`
        variant_file_re = re.compile(
            rf"({'|'.join(weight_prefixes)})\.({variant}|{variant}-{transformers_index_format})\.({'|'.join(weight_suffixs)})$"
        )
        # `text_encoder/pytorch_model.bin.index.fp16.json`
        variant_index_re = re.compile(
            rf"({'|'.join(weight_prefixes)})\.({'|'.join(weight_suffixs)})\.index\.{variant}\.json$"
        )
        legacy_variant_file_re = re.compile(rf".*-{transformers_index_format}\.{variant}\.[a-z]+$")
        legacy_variant_index_re = re.compile(
            rf"({'|'.join(weight_prefixes)})\.({'|'.join(weight_suffixs)})\.{variant}\.index\.json$"
        )

    # `diffusion_pytorch_model.bin` as well as `model-00001-of-00002.safetensors`
    non_variant_file_re = re.compile(
        rf"({'|'.join(weight_prefixes)})(-{transformers_index_format})?\.({'|'.join(weight_suffixs)})$"
    )
    # `text_encoder/pytorch_model.bin.index.json`
    non_variant_index_re = re.compile(rf"({'|'.join(weight_prefixes)})\.({'|'.join(weight_suffixs)})\.index\.json")

    def filter_for_compatible_extensions(filenames, ignore_patterns=None):
        if not ignore_patterns:
            return filenames

        # ignore patterns uses glob style patterns e.g *.safetensors but we're only
        # interested in the extension name
        return {f for f in filenames if not any(f.endswith(pat.lstrip("*.")) for pat in ignore_patterns)}

    # Group files by component
    components = {}
    for filename in filenames:
        if not len(filename.split("/")) == 2:
            components.setdefault("", []).append(filename)
            continue

        component, _ = filename.split("/")
        components.setdefault(component, []).append(filename)

    usable_filenames = set()
    variant_filenames = set()
    for component, component_filenames in components.items():
        component_filenames = filter_for_compatible_extensions(component_filenames, ignore_patterns=ignore_patterns)

        component_variants = set()
        component_legacy_variants = set()
        component_non_variants = set()
        if variant is not None:
            component_variants = filter_with_regex(component_filenames, variant_file_re)
            component_variant_index_files = filter_with_regex(component_filenames, variant_index_re)

            component_legacy_variants = filter_with_regex(component_filenames, legacy_variant_file_re)
            component_legacy_variant_index_files = filter_with_regex(component_filenames, legacy_variant_index_re)

        if component_variants or component_legacy_variants:
            variant_filenames.update(
                component_variants | component_variant_index_files
                if component_variants
                else component_legacy_variants | component_legacy_variant_index_files
            )

        else:
            component_non_variants = filter_with_regex(component_filenames, non_variant_file_re)
            component_variant_index_files = filter_with_regex(component_filenames, non_variant_index_re)

            usable_filenames.update(component_non_variants | component_variant_index_files)

    usable_filenames.update(variant_filenames)

    if len(variant_filenames) == 0 and variant is not None:
        error_message = (
            f"You are trying to load model files of the `variant={variant}`, but no such modeling files are available. "
        )
        raise ValueError(error_message)

    if len(variant_filenames) > 0 and usable_filenames != variant_filenames:
        logger.warning(
            f"\nA mixture of {variant} and non-{variant} filenames will be loaded.\nLoaded {variant} filenames:\n"
            f"[{', '.join(variant_filenames)}]\nLoaded non-{variant} filenames:\n"
            f"[{', '.join(usable_filenames - variant_filenames)}\nIf this behavior is not "
            f"expected, please check your folder structure."
        )

    return usable_filenames, variant_filenames


@validate_hf_hub_args
def warn_deprecated_model_variant(pretrained_model_name_or_path, token, variant, revision, model_filenames):
    info = model_info(
        pretrained_model_name_or_path,
        token=token,
        revision=None,
    )
    filenames = {sibling.rfilename for sibling in info.siblings}
    comp_model_filenames, _ = variant_compatible_siblings(filenames, variant=revision)
    comp_model_filenames = [".".join(f.split(".")[:1] + f.split(".")[2:]) for f in comp_model_filenames]

    if set(model_filenames).issubset(set(comp_model_filenames)):
        warnings.warn(
            f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'` even though you can load it via `variant=`{revision}`. Loading model variants via `revision='{revision}'` is deprecated and will be removed in diffusers v1. Please use `variant='{revision}'` instead.",  # noqa: E501
            FutureWarning,
        )
    else:
        warnings.warn(
            f"You are loading the variant {revision} from {pretrained_model_name_or_path} via `revision='{revision}'`. This behavior is deprecated and will be removed in diffusers v1. One should use `variant='{revision}'` instead. However, it appears that {pretrained_model_name_or_path} currently does not have the required variant filenames in the 'main' branch. \n The Diffusers team and community would be very grateful if you could open an issue: https://github.com/huggingface/diffusers/issues/new with the title '{pretrained_model_name_or_path} is missing {revision} files' so that the correct variant file can be added.",  # noqa: E501
            FutureWarning,
        )


def _unwrap_model(model):
    """Unwraps a model."""
    from mindone.peft import PeftModel

    if isinstance(model, PeftModel):
        model = model.base_model.model
    return model


def maybe_raise_or_warn(library_name, class_name, importable_classes, passed_class_obj, name, is_pipeline_module):
    """Simple helper method to raise or warn in case incorrect module has been passed"""
    if not is_pipeline_module:
        if library_name == "diffusers":
            library = maybe_import_module_in_mindone(library_name)
            class_obj = getattr(library, class_name)
            class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}
        elif library_name == "transformers":
            library = maybe_import_module_in_mindone(library_name)
            if hasattr(library, class_name):
                class_obj = getattr(library, class_name)
                class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}
            else:  # class_name is not implemented in mindone, try get it from huggingface library
                library = maybe_import_module_in_mindone(library_name, force_original=True)
                class_obj = getattr(library, class_name, None)
                if class_obj is None or issubclass(class_obj, library.PreTrainedModel):
                    # if class_name is a kind of model, we should notify the users.
                    # 1. huggingface/transformers w/o torch; 2. w/ torch; what if with tensorflow/flax?
                    raise NotImplementedError(f"{class_name} has not been implemented in mindone.transformers yet")
                class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}
        else:
            raise NotImplementedError(f"{library_name} has not been implemented in mindone yet.")

        expected_class_obj = None
        for class_name, class_candidate in class_candidates.items():
            if class_candidate is not None and issubclass(class_obj, class_candidate):
                expected_class_obj = class_candidate

        # Dynamo wraps the original model in a private class.
        # I didn't find a public API to get the original class.
        sub_model = passed_class_obj[name]
        unwrapped_sub_model = _unwrap_model(sub_model)
        model_cls = unwrapped_sub_model.__class__

        if not issubclass(model_cls, expected_class_obj):
            raise ValueError(f"{passed_class_obj[name]} is of type: {model_cls}, but should be {expected_class_obj}")
    else:
        logger.warning(
            f"You have passed a non-standard module {passed_class_obj[name]}. We cannot verify whether it"
            " has the correct type"
        )


# a simpler version of get_class_obj_and_candidates, it won't work with custom code
def simple_get_class_obj(library_name, class_name):
    from mindone.diffusers import pipelines

    is_pipeline_module = hasattr(pipelines, library_name)

    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)
        class_obj = getattr(pipeline_module, class_name)
    else:
        library = importlib.import_module(library_name)
        class_obj = getattr(library, class_name)

    return class_obj


def get_class_obj_and_candidates(
    library_name, class_name, importable_classes, pipelines, is_pipeline_module, component_name=None, cache_dir=None
):
    """Simple helper method to retrieve class object of module as well as potential parent class objects"""
    component_folder = os.path.join(cache_dir, component_name) if component_name and cache_dir else None

    if is_pipeline_module:
        pipeline_module = getattr(pipelines, library_name)

        class_obj = getattr(pipeline_module, class_name)
        class_candidates = dict.fromkeys(importable_classes.keys(), class_obj)
    elif component_folder and os.path.isfile(os.path.join(component_folder, library_name + ".py")):
        # load custom component
        class_obj = get_class_from_dynamic_module(
            component_folder, module_file=library_name + ".py", class_name=class_name
        )
        class_candidates = dict.fromkeys(importable_classes.keys(), class_obj)
    else:
        # else we just import it from the library.
        if library_name == "diffusers":
            library = maybe_import_module_in_mindone(library_name)
            class_obj = getattr(library, class_name)
            class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}
        elif library_name == "transformers":
            library = maybe_import_module_in_mindone(library_name)
            if hasattr(library, class_name):
                class_obj = getattr(library, class_name)
                class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}
            else:  # class_name is not implemented in mindone, try get it from huggingface library
                library = maybe_import_module_in_mindone(library_name, force_original=True)
                class_obj = getattr(library, class_name, None)
                if class_obj is None or issubclass(class_obj, library.PreTrainedModel):
                    # if class_name is a kind of model, we should notify the users.
                    # 1. huggingface/transformers w/o torch; 2. w/ torch; what if with tensorflow/flax?
                    raise NotImplementedError(f"{class_name} has not been implemented in mindone.transformers yet")
                class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}
        else:
            # we just import it from the library.
            import importlib

            library = importlib.import_module(library_name)
            class_obj = getattr(library, class_name)
            class_candidates = {c: getattr(library, c, None) for c in importable_classes.keys()}

    return class_obj, class_candidates


def _get_custom_pipeline_class(
    custom_pipeline,
    repo_id=None,
    hub_revision=None,
    class_name=None,
    cache_dir=None,
    revision=None,
):
    if custom_pipeline.endswith(".py"):
        path = Path(custom_pipeline)
        # decompose into folder & file
        file_name = path.name
        custom_pipeline = path.parent.absolute()
    elif repo_id is not None:
        file_name = f"{custom_pipeline}.py"
        custom_pipeline = repo_id
    else:
        file_name = CUSTOM_PIPELINE_FILE_NAME

    if repo_id is not None and hub_revision is not None:
        # if we load the pipeline code from the Hub
        # make sure to overwrite the `revision`
        revision = hub_revision

    return get_class_from_dynamic_module(
        custom_pipeline,
        module_file=file_name,
        class_name=class_name,
        cache_dir=cache_dir,
        revision=revision,
    )


def _get_pipeline_class(
    class_obj,
    config=None,
    load_connected_pipeline=False,
    custom_pipeline=None,
    repo_id=None,
    hub_revision=None,
    class_name=None,
    cache_dir=None,
    revision=None,
):
    if custom_pipeline is not None:
        return _get_custom_pipeline_class(
            custom_pipeline,
            repo_id=repo_id,
            hub_revision=hub_revision,
            class_name=class_name,
            cache_dir=cache_dir,
            revision=revision,
        )

    if class_obj.__name__ != "DiffusionPipeline" and class_obj.__name__ != "ModularPipeline":
        return class_obj

    diffusers_module = maybe_import_module_in_mindone(class_obj.__module__.split(".")[1])
    class_name = class_name or config["_class_name"]
    if not class_name:
        raise ValueError(
            "The class name could not be found in the configuration file. Please make sure to pass the correct `class_name`."
        )

    class_name = class_name[4:] if class_name.startswith("Flax") else class_name

    pipeline_cls = getattr(diffusers_module, class_name)

    if load_connected_pipeline:
        raise NotImplementedError("connected pipeline is not supported for now.")

    return pipeline_cls


def load_sub_model(
    library_name: str,
    class_name: str,
    importable_classes: List[Any],
    pipelines: Any,
    is_pipeline_module: bool,
    pipeline_class: Any,
    mindspore_dtype: ms.Type,
    model_variants: Dict[str, str],
    name: str,
    variant: str,
    cached_folder: Union[str, os.PathLike],
    use_safetensors: bool,
    dduf_entries: Optional[Dict[str, DDUFEntry]],
):
    """Helper method to load the module `name` from `library_name` and `class_name`"""

    # retrieve class candidates

    class_obj, class_candidates = get_class_obj_and_candidates(
        library_name,
        class_name,
        importable_classes,
        pipelines,
        is_pipeline_module,
        component_name=name,
        cache_dir=cached_folder,
    )

    load_method_name = None
    # retrieve load method name
    for class_name, class_candidate in class_candidates.items():
        if class_candidate is not None and issubclass(class_obj, class_candidate):
            load_method_name = importable_classes[class_name][1]

    # if load method name is None, then we have a dummy module -> raise Error
    if load_method_name is None:
        none_module = class_obj.__module__
        is_dummy_path = none_module.startswith(DUMMY_MODULES_FOLDER) or none_module.startswith(
            TRANSFORMERS_DUMMY_MODULES_FOLDER
        )
        if is_dummy_path and "dummy" in none_module:
            # call class_obj for nice error message of missing requirements
            class_obj()

        raise ValueError(
            f"The component {class_obj} of {pipeline_class} cannot be loaded as it does not seem to have"
            f" any of the loading methods defined in {ALL_IMPORTABLE_CLASSES}."
        )

    load_method = _get_load_method(class_obj, load_method_name, is_dduf=dduf_entries is not None)

    # add kwargs to loading method
    diffusers_module = maybe_import_module_in_mindone(__name__.split(".")[1])
    loading_kwargs = {}
    if issubclass(class_obj, nn.Cell):
        loading_kwargs["mindspore_dtype"] = mindspore_dtype

    is_diffusers_model = issubclass(class_obj, diffusers_module.ModelMixin)

    transformers_version = version.parse(version.parse(transformers.__version__).base_version)
    is_transformers_model = issubclass(class_obj, MSPreTrainedModel) and transformers_version >= version.parse("4.20.0")

    # When loading a transformers model, if the device_map is None, the weights will be initialized as opposed to diffusers.
    # To make default loading faster we set the `low_cpu_mem_usage=low_cpu_mem_usage` flag which is `True` by default.
    # This makes sure that the weights won't be initialized which significantly speeds up loading.
    if is_diffusers_model or is_transformers_model:
        loading_kwargs["variant"] = model_variants.pop(name, None)
        loading_kwargs["use_safetensors"] = use_safetensors

        # the following can be deleted once the minimum required `transformers` version
        # is higher than 4.27
        if (
            is_transformers_model
            and loading_kwargs["variant"] is not None
            and transformers_version < version.parse("4.27.0")
        ):
            raise ImportError(
                f"When passing `variant='{variant}'`, please make sure to upgrade your `transformers` version to at least 4.27.0.dev0"
            )
        elif is_transformers_model and loading_kwargs["variant"] is None:
            loading_kwargs.pop("variant")

    # check if the module is in a subdirectory
    if dduf_entries:
        loading_kwargs["dduf_entries"] = dduf_entries
        loaded_sub_model = load_method(name, **loading_kwargs)
    elif os.path.isdir(os.path.join(cached_folder, name)):
        loaded_sub_model = load_method(os.path.join(cached_folder, name), **loading_kwargs)
    else:
        # else load from the root directory
        loaded_sub_model = load_method(cached_folder, **loading_kwargs)

    return loaded_sub_model


def _get_load_method(class_obj: object, load_method_name: str, is_dduf: bool) -> Callable:
    """
    Return the method to load the sub model.

    In practice, this method will return the `"from_pretrained"` (or `load_method_name`) method of the class object
    except if loading from a DDUF checkpoint. In that case, transformers models and tokenizers have a specific loading
    method that we need to use.
    """
    if is_dduf:
        if issubclass(class_obj, PreTrainedTokenizerBase):
            return lambda *args, **kwargs: _load_tokenizer_from_dduf(class_obj, *args, **kwargs)
        if issubclass(class_obj, MSPreTrainedModel):
            return lambda *args, **kwargs: _load_transformers_model_from_dduf(class_obj, *args, **kwargs)
    return getattr(class_obj, load_method_name)


def _fetch_class_library_tuple(module):
    # import it here to avoid circular import
    diffusers_module = maybe_import_module_in_mindone(__name__.split(".")[1])
    pipelines = getattr(diffusers_module, "pipelines")

    # register the config from the original module, not the dynamo compiled one
    not_compiled_module = _unwrap_model(module)
    library = not_compiled_module.__module__.split(".")[0]
    if library == "mindone":  # give the subpackage name like diffusers/transformers
        library = not_compiled_module.__module__.split(".")[1]

    # check if the module is a pipeline module
    module_path_items = not_compiled_module.__module__.split(".")
    pipeline_dir = module_path_items[-2] if len(module_path_items) > 2 else None

    path = not_compiled_module.__module__.split(".")
    is_pipeline_module = pipeline_dir in path and hasattr(pipelines, pipeline_dir)

    # if library is not in LOADABLE_CLASSES, then it is a custom module.
    # Or if it's a pipeline module, then the module is inside the pipeline
    # folder so we set the library to module name.
    if is_pipeline_module:
        library = pipeline_dir
    elif library not in LOADABLE_CLASSES:
        library = not_compiled_module.__module__

    # retrieve class_name
    if isinstance(not_compiled_module, type):
        class_name = not_compiled_module.__name__
    else:
        class_name = not_compiled_module.__class__.__name__

    return (library, class_name)


def _identify_model_variants(folder: str, variant: str, config: dict) -> dict:
    model_variants = {}
    if variant is not None:
        for sub_folder in os.listdir(folder):
            folder_path = os.path.join(folder, sub_folder)
            is_folder = os.path.isdir(folder_path) and sub_folder in config
            variant_exists = is_folder and any(p.split(".")[1].startswith(variant) for p in os.listdir(folder_path))
            if variant_exists:
                model_variants[sub_folder] = variant
    return model_variants


def _resolve_custom_pipeline_and_cls(folder, config, custom_pipeline):
    custom_class_name = None
    if os.path.isfile(os.path.join(folder, f"{custom_pipeline}.py")):
        custom_pipeline = os.path.join(folder, f"{custom_pipeline}.py")
    elif isinstance(config["_class_name"], (list, tuple)) and os.path.isfile(
        os.path.join(folder, f"{config['_class_name'][0]}.py")
    ):
        custom_pipeline = os.path.join(folder, f"{config['_class_name'][0]}.py")
        custom_class_name = config["_class_name"][1]

    return custom_pipeline, custom_class_name


def _maybe_raise_warning_for_inpainting(pipeline_class, pretrained_model_name_or_path: str, config: dict):
    if pipeline_class.__name__ == "StableDiffusionInpaintPipeline" and version.parse(
        version.parse(config["_diffusers_version"]).base_version
    ) <= version.parse("0.5.1"):
        from diffusers import StableDiffusionInpaintPipeline, StableDiffusionInpaintPipelineLegacy

        pipeline_class = StableDiffusionInpaintPipelineLegacy

        deprecation_message = (
            "You are using a legacy checkpoint for inpainting with Stable Diffusion, therefore we are loading the"
            f" {StableDiffusionInpaintPipelineLegacy} class instead of {StableDiffusionInpaintPipeline}. For"
            " better inpainting results, we strongly suggest using Stable Diffusion's official inpainting"
            " checkpoint: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting instead or adapting your"
            f" checkpoint {pretrained_model_name_or_path} to the format of"
            " https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-inpainting. Note that we do not actively maintain"
            " the {StableDiffusionInpaintPipelineLegacy} class and will likely remove it in version 1.0.0."
        )
        deprecate("StableDiffusionInpaintPipelineLegacy", "1.0.0", deprecation_message, standard_warn=False)


def _update_init_kwargs_with_connected_pipeline(
    init_kwargs: dict, passed_pipe_kwargs: dict, passed_class_objs: dict, folder: str, **pipeline_loading_kwargs
) -> dict:
    from .pipeline_utils import DiffusionPipeline

    modelcard = ModelCard.load(os.path.join(folder, "README.md"))
    connected_pipes = {prefix: getattr(modelcard.data, prefix, [None])[0] for prefix in CONNECTED_PIPES_KEYS}

    # We don't scheduler argument to match the existing logic:
    # https://github.com/huggingface/diffusers/blob/867e0c919e1aa7ef8b03c8eb1460f4f875a683ae/src/diffusers/pipelines/pipeline_utils.py#L906C13-L925C14
    pipeline_loading_kwargs_cp = pipeline_loading_kwargs.copy()
    if pipeline_loading_kwargs_cp is not None and len(pipeline_loading_kwargs_cp) >= 1:
        for k in pipeline_loading_kwargs:
            if "scheduler" in k:
                _ = pipeline_loading_kwargs_cp.pop(k)

    def get_connected_passed_kwargs(prefix):
        connected_passed_class_obj = {
            k.replace(f"{prefix}_", ""): w for k, w in passed_class_objs.items() if k.split("_")[0] == prefix
        }
        connected_passed_pipe_kwargs = {
            k.replace(f"{prefix}_", ""): w for k, w in passed_pipe_kwargs.items() if k.split("_")[0] == prefix
        }

        connected_passed_kwargs = {**connected_passed_class_obj, **connected_passed_pipe_kwargs}
        return connected_passed_kwargs

    connected_pipes = {
        prefix: DiffusionPipeline.from_pretrained(
            repo_id, **pipeline_loading_kwargs_cp, **get_connected_passed_kwargs(prefix)
        )
        for prefix, repo_id in connected_pipes.items()
        if repo_id is not None
    }

    for prefix, connected_pipe in connected_pipes.items():
        # add connected pipes to `init_kwargs` with <prefix>_<component_name>, e.g. "prior_text_encoder"
        init_kwargs.update(
            {"_".join([prefix, name]): component for name, component in connected_pipe.components.items()}
        )

    return init_kwargs


def _get_custom_components_and_folders(
    pretrained_model_name: str,
    config_dict: Dict[str, Any],
    filenames: Optional[List[str]] = None,
    variant_filenames: Optional[List[str]] = None,
    variant: Optional[str] = None,
):
    config_dict = config_dict.copy()

    # retrieve all folder_names that contain relevant files
    folder_names = [k for k, v in config_dict.items() if isinstance(v, list) and k != "_class_name"]

    # diffusers -> mindone.diffusers
    diffusers_module = importlib.import_module(".".join(__name__.split(".")[:2]))
    pipelines = getattr(diffusers_module, "pipelines")

    # optionally create a custom component <> custom file mapping
    custom_components = {}
    for component in folder_names:
        module_candidate = config_dict[component][0]

        if module_candidate is None or not isinstance(module_candidate, str):
            continue

        # We compute candidate file path on the Hub. Do not use `os.path.join`.
        candidate_file = f"{component}/{module_candidate}.py"

        if candidate_file in filenames:
            custom_components[component] = module_candidate
        elif module_candidate not in LOADABLE_CLASSES and not hasattr(pipelines, module_candidate):
            raise ValueError(
                f"{candidate_file} as defined in `model_index.json` does not exist in {pretrained_model_name} and is not a module in 'diffusers/pipelines'."
            )

    return custom_components, folder_names


def _get_ignore_patterns(
    passed_components,
    model_folder_names: List[str],
    model_filenames: List[str],
    use_safetensors: bool,
    from_flax: bool,
    allow_pickle: bool,
    use_onnx: bool,
    is_onnx: bool,
    variant: Optional[str] = None,
) -> List[str]:
    if (
        use_safetensors
        and not allow_pickle
        and not is_safetensors_compatible(
            model_filenames, passed_components=passed_components, folder_names=model_folder_names, variant=variant
        )
    ):
        raise EnvironmentError(
            f"Could not find the necessary `safetensors` weights in {model_filenames} (variant={variant})"
        )

    if from_flax:
        ignore_patterns = ["*.bin", "*.safetensors", "*.onnx", "*.pb"]

    elif use_safetensors and is_safetensors_compatible(
        model_filenames, passed_components=passed_components, folder_names=model_folder_names, variant=variant
    ):
        ignore_patterns = ["*.bin", "*.msgpack"]

        use_onnx = use_onnx if use_onnx is not None else is_onnx
        if not use_onnx:
            ignore_patterns += ["*.onnx", "*.pb"]

    else:
        ignore_patterns = ["*.safetensors", "*.msgpack"]

        use_onnx = use_onnx if use_onnx is not None else is_onnx
        if not use_onnx:
            ignore_patterns += ["*.onnx", "*.pb"]

    return ignore_patterns


def _download_dduf_file(
    pretrained_model_name: str,
    dduf_file: str,
    pipeline_class_name: str,
    cache_dir: str,
    proxies: str,
    local_files_only: bool,
    token: str,
    revision: str,
):
    model_info_call_error = None
    if not local_files_only:
        try:
            info = model_info(pretrained_model_name, token=token, revision=revision)
        except (HTTPError, OfflineModeIsEnabled, requests.ConnectionError) as e:
            logger.warning(f"Couldn't connect to the Hub: {e}.\nWill try to load from local cache.")
            local_files_only = True
            model_info_call_error = e  # save error to reraise it if model is not cached locally

    if (
        not local_files_only
        and dduf_file is not None
        and dduf_file not in (sibling.rfilename for sibling in info.siblings)
    ):
        raise ValueError(f"Requested {dduf_file} file is not available in {pretrained_model_name}.")

    try:
        user_agent = {"pipeline_class": pipeline_class_name, "dduf": True}
        cached_folder = snapshot_download(
            pretrained_model_name,
            cache_dir=cache_dir,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            allow_patterns=[dduf_file],
            user_agent=user_agent,
        )
        return cached_folder
    except FileNotFoundError:
        # Means we tried to load pipeline with `local_files_only=True` but the files have not been found in local cache.
        # This can happen in two cases:
        # 1. If the user passed `local_files_only=True`                    => we raise the error directly
        # 2. If we forced `local_files_only=True` when `model_info` failed => we raise the initial error
        if model_info_call_error is None:
            # 1. user passed `local_files_only=True`
            raise
        else:
            # 2. we forced `local_files_only=True` when `model_info` failed
            raise EnvironmentError(
                f"Cannot load model {pretrained_model_name}: model is not cached locally and an error occurred"
                " while trying to fetch metadata from the Hub. Please check out the root cause in the stacktrace"
                " above."
            ) from model_info_call_error
