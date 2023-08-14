# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Hub utilities: utilities related to download and cache models
"""
import json
import os
import re
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import urlparse
from uuid import uuid4

import requests
from requests.exceptions import HTTPError

from . import __version__, logging
from .import_utils import (
    ENV_VARS_TRUE_VALUES,
    _tf_version,
    _torch_version,
    is_tf_available,
    is_torch_available,
    is_training_run_on_sagemaker,
)
from .logging import tqdm

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

_is_offline_mode = True if os.environ.get("TRANSFORMERS_OFFLINE", "0").upper() in ENV_VARS_TRUE_VALUES else False


def is_offline_mode():
    return _is_offline_mode


torch_cache_home = os.getenv("TORCH_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "torch"))
old_default_cache_path = os.path.join(torch_cache_home, "transformers")
# New default cache, shared with the Datasets library
hf_cache_home = os.path.expanduser(
    os.getenv("HF_HOME", os.path.join(os.getenv("XDG_CACHE_HOME", "~/.cache"), "huggingface"))
)
default_cache_path = os.path.join(hf_cache_home, "hub")

PYTORCH_PRETRAINED_BERT_CACHE = os.getenv("PYTORCH_PRETRAINED_BERT_CACHE", default_cache_path)
PYTORCH_TRANSFORMERS_CACHE = os.getenv("PYTORCH_TRANSFORMERS_CACHE", PYTORCH_PRETRAINED_BERT_CACHE)
HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE", PYTORCH_TRANSFORMERS_CACHE)
TRANSFORMERS_CACHE = os.getenv("TRANSFORMERS_CACHE", HUGGINGFACE_HUB_CACHE)
HF_MODULES_CACHE = os.getenv("HF_MODULES_CACHE", os.path.join(hf_cache_home, "modules"))
TRANSFORMERS_DYNAMIC_MODULE_NAME = "transformers_modules"
SESSION_ID = uuid4().hex
DISABLE_TELEMETRY = os.getenv("DISABLE_TELEMETRY", False) in ENV_VARS_TRUE_VALUES

S3_BUCKET_PREFIX = "https://s3.amazonaws.com/models.huggingface.co/bert"
CLOUDFRONT_DISTRIB_PREFIX = "https://cdn.huggingface.co"

_staging_mode = os.environ.get("HUGGINGFACE_CO_STAGING", "NO").upper() in ENV_VARS_TRUE_VALUES
_default_endpoint = "https://hub-ci.huggingface.co" if _staging_mode else "https://huggingface.co"

HUGGINGFACE_CO_RESOLVE_ENDPOINT = _default_endpoint
if os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None) is not None:
    warnings.warn(
        "Using the environment variable `HUGGINGFACE_CO_RESOLVE_ENDPOINT` is deprecated and will be removed in "
        "Transformers v5. Use `HF_ENDPOINT` instead.",
        FutureWarning,
    )
    HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HUGGINGFACE_CO_RESOLVE_ENDPOINT", None)
HUGGINGFACE_CO_RESOLVE_ENDPOINT = os.environ.get("HF_ENDPOINT", HUGGINGFACE_CO_RESOLVE_ENDPOINT)
HUGGINGFACE_CO_PREFIX = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/{model_id}/resolve/{revision}/{filename}"
HUGGINGFACE_CO_EXAMPLES_TELEMETRY = HUGGINGFACE_CO_RESOLVE_ENDPOINT + "/api/telemetry/examples"

# Return value when trying to load a file from cache but the file does not exist in the distant repo.
_CACHED_NO_EXIST = object()


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def define_sagemaker_information():
    try:
        instance_data = requests.get(os.environ["ECS_CONTAINER_METADATA_URI"]).json()
        dlc_container_used = instance_data["Image"]
        dlc_tag = instance_data["Image"].split(":")[1]
    except Exception:
        dlc_container_used = None
        dlc_tag = None

    sagemaker_params = json.loads(os.getenv("SM_FRAMEWORK_PARAMS", "{}"))
    runs_distributed_training = True if "sagemaker_distributed_dataparallel_enabled" in sagemaker_params else False
    account_id = os.getenv("TRAINING_JOB_ARN").split(":")[4] if "TRAINING_JOB_ARN" in os.environ else None

    sagemaker_object = {
        "sm_framework": os.getenv("SM_FRAMEWORK_MODULE", None),
        "sm_region": os.getenv("AWS_REGION", None),
        "sm_number_gpu": os.getenv("SM_NUM_GPUS", 0),
        "sm_number_cpu": os.getenv("SM_NUM_CPUS", 0),
        "sm_distributed_training": runs_distributed_training,
        "sm_deep_learning_container": dlc_container_used,
        "sm_deep_learning_container_tag": dlc_tag,
        "sm_account_id": account_id,
    }
    return sagemaker_object


def http_user_agent(user_agent: Union[Dict, str, None] = None) -> str:
    """
    Formats a user-agent string with basic info about a request.
    """
    ua = f"transformers/{__version__}; python/{sys.version.split()[0]}; session_id/{SESSION_ID}"
    if is_torch_available():
        ua += f"; torch/{_torch_version}"
    if is_tf_available():
        ua += f"; tensorflow/{_tf_version}"
    if DISABLE_TELEMETRY:
        return ua + "; telemetry/off"
    if is_training_run_on_sagemaker():
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in define_sagemaker_information().items())
    # CI will set this value to True
    if os.environ.get("TRANSFORMERS_IS_CI", "").upper() in ENV_VARS_TRUE_VALUES:
        ua += "; is_ci/true"
    if isinstance(user_agent, dict):
        ua += "; " + "; ".join(f"{k}/{v}" for k, v in user_agent.items())
    elif isinstance(user_agent, str):
        ua += "; " + user_agent
    return ua


def extract_commit_hash(resolved_file: Optional[str], commit_hash: Optional[str]):
    """
    Extracts the commit hash from a resolved filename toward a cache file.
    """
    if resolved_file is None or commit_hash is not None:
        return commit_hash
    resolved_file = str(Path(resolved_file).as_posix())
    search = re.search(r"snapshots/([^/]+)/", resolved_file)
    if search is None:
        return None
    commit_hash = search.groups()[0]
    raise NotImplementedError("`huggingface_hub.file_download.REGEX_COMMIT_HASH` is not available!")


def try_to_load_from_cache(
    repo_id: str,
    filename: str,
    cache_dir: Union[str, Path, None] = None,
    revision: Optional[str] = None,
    repo_type: Optional[str] = None,
) -> Optional[str]:
    """
    Explores the cache to return the latest cached file for a given revision if found.

    This function will not raise any exception if the file in not cached.

    Args:
        cache_dir (`str` or `os.PathLike`):
            The folder where the cached files lie.
        repo_id (`str`):
            The ID of the repo on huggingface.co.
        filename (`str`):
            The filename to look for inside `repo_id`.
        revision (`str`, *optional*):
            The specific model version to use. Will default to `"main"` if it's not provided and no `commit_hash` is
            provided either.
        repo_type (`str`, *optional*):
            The type of the repo.

    Returns:
        `Optional[str]` or `_CACHED_NO_EXIST`:
            Will return `None` if the file was not cached. Otherwise:
            - The exact path to the cached file if it's found in the cache
            - A special value `_CACHED_NO_EXIST` if the file does not exist at the given commit hash and this fact was
              cached.
    """
    if revision is None:
        revision = "main"

    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE

    object_id = repo_id.replace("/", "--")
    if repo_type is None:
        repo_type = "model"
    repo_cache = os.path.join(cache_dir, f"{repo_type}s--{object_id}")
    if not os.path.isdir(repo_cache):
        # No cache for this model
        return None
    for subfolder in ["refs", "snapshots"]:
        if not os.path.isdir(os.path.join(repo_cache, subfolder)):
            return None

    # Resolve refs (for instance to convert main to the associated commit sha)
    cached_refs = os.listdir(os.path.join(repo_cache, "refs"))
    if revision in cached_refs:
        with open(os.path.join(repo_cache, "refs", revision)) as f:
            revision = f.read()

    if os.path.isfile(os.path.join(repo_cache, ".no_exist", revision, filename)):
        return _CACHED_NO_EXIST

    cached_shas = os.listdir(os.path.join(repo_cache, "snapshots"))
    if revision not in cached_shas:
        # No cache for this revision and we won't try to return a random revision
        return None

    cached_file = os.path.join(repo_cache, "snapshots", revision, filename)
    return cached_file if os.path.isfile(cached_file) else None


def cached_file(
    path_or_repo_id: Union[str, os.PathLike],
    filename: str,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    resume_download: bool = False,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
    subfolder: str = "",
    repo_type: Optional[str] = None,
    user_agent: Optional[Union[str, Dict[str, str]]] = None,
    _raise_exceptions_for_missing_entries: bool = True,
    _raise_exceptions_for_connection_errors: bool = True,
    _commit_hash: Optional[str] = None,
):
    """
    Tries to locate a file in a local folder and repo, downloads and cache it if necessary.

    Args:
        path_or_repo_id (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a model repo on huggingface.co.
            - a path to a *directory* potentially containing the file.
        filename (`str`):
            The name of the file to locate in `path_or_repo`.
        cache_dir (`str` or `os.PathLike`, *optional*):
            Path to a directory in which a downloaded pretrained model configuration should be cached if the standard
            cache should not be used.
        force_download (`bool`, *optional*, defaults to `False`):
            Whether or not to force to (re-)download the configuration files and override the cached versions if they
            exist.
        resume_download (`bool`, *optional*, defaults to `False`):
            Whether or not to delete incompletely received file. Attempts to resume the download if such a file exists.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
        use_auth_token (`str` or *bool*, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `huggingface-cli login` (stored in `~/.huggingface`).
        revision (`str`, *optional*, defaults to `"main"`):
            The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
            git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
            identifier allowed by git.
        local_files_only (`bool`, *optional*, defaults to `False`):
            If `True`, will only try to load the tokenizer configuration from local files.
        subfolder (`str`, *optional*, defaults to `""`):
            In case the relevant files are located inside a subfolder of the model repo on huggingface.co, you can
            specify the folder name here.
        repo_type (`str`, *optional*):
            Specify the repo type (useful when downloading from a space for instance).

    <Tip>

    Passing `use_auth_token=True` is required when you want to use a private model.

    </Tip>

    Returns:
        `Optional[str]`: Returns the resolved file (to the cache folder if downloaded from a repo).

    Examples:

    ```python
    # Download a model weight from the Hub and cache it.
    model_weights_file = cached_file("bert-base-uncased", "pytorch_model.bin")
    ```"""
    # Private arguments
    #     _raise_exceptions_for_missing_entries: if False, do not raise an exception for missing entries but return
    #         None.
    #     _raise_exceptions_for_connection_errors: if False, do not raise an exception for connection errors but return
    #         None.
    #     _commit_hash: passed when we are chaining several calls to various files (e.g. when loading a tokenizer or
    #         a pipeline). If files are cached for this commit hash, avoid calls to head and get from the cache.
    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True
    if subfolder is None:
        subfolder = ""

    path_or_repo_id = str(path_or_repo_id)
    full_filename = os.path.join(subfolder, filename)
    if os.path.isdir(path_or_repo_id):
        resolved_file = os.path.join(os.path.join(path_or_repo_id, subfolder), filename)
        if not os.path.isfile(resolved_file):
            if _raise_exceptions_for_missing_entries:
                raise EnvironmentError(
                    f"{path_or_repo_id} does not appear to have a file named {full_filename}. Checkout "
                    f"'https://huggingface.co/{path_or_repo_id}/{revision}' for available files."
                )
            else:
                return None
        return resolved_file

    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if _commit_hash is not None and not force_download:
        # If the file is cached under that commit hash, we return it directly.
        resolved_file = try_to_load_from_cache(
            path_or_repo_id, full_filename, cache_dir=cache_dir, revision=_commit_hash, repo_type=repo_type
        )
        if resolved_file is not None:
            if resolved_file is not _CACHED_NO_EXIST:
                return resolved_file
            elif not _raise_exceptions_for_missing_entries:
                return None
            else:
                raise EnvironmentError(f"Could not locate {full_filename} inside {path_or_repo_id}.")

    user_agent = http_user_agent(user_agent)
    raise NotImplementedError("`huggingface_hub.hf_hub_download` is not available!")


def download_url(url, proxies=None):
    """
    Downloads a given url in a temporary file. This function is not safe to use in multiple processes. Its only use is
    for deprecated behavior allowing to download config/models with a single url instead of using the Hub.

    Args:
        url (`str`): The url of the file to download.
        proxies (`Dict[str, str]`, *optional*):
            A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
            'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.

    Returns:
        `str`: The location of the temporary file where the url was downloaded.
    """
    warnings.warn(
        f"Using `from_pretrained` with the url of a file (here {url}) is deprecated and won't be possible anymore in"
        " v5 of Transformers. You should host your file on the Hub (hf.co) instead and use the repository ID. Note"
        " that this is not compatible with the caching system (your file will be downloaded at each execution) or"
        " multiple processes (each process will download the file in a different temporary file)."
    )
    tmp_file = tempfile.mkstemp()[1]
    raise NotImplementedError("`huggingface_hub.file_download` is not available!")


def has_file(
    path_or_repo: Union[str, os.PathLike],
    filename: str,
    revision: Optional[str] = None,
    proxies: Optional[Dict[str, str]] = None,
    use_auth_token: Optional[Union[bool, str]] = None,
):
    """
    Checks if a repo contains a given file without downloading it. Works for remote repos and local folders.

    <Tip warning={false}>

    This function will raise an error if the repository `path_or_repo` is not valid or if `revision` does not exist for
    this repo, but will return False for regular connection errors.

    </Tip>
    """
    if os.path.isdir(path_or_repo):
        return os.path.isfile(os.path.join(path_or_repo, filename))

    raise NotImplementedError("Only work with local path for now!")


class PushToHubMixin:
    """
    A Mixin containing the functionality to push a model or tokenizer to the hub.
    """

    def _create_repo(
        self,
        repo_id: str,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        repo_url: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    def _get_files_timestamps(self, working_dir: Union[str, os.PathLike]):
        """
        Returns the list of files with their last modification timestamp.
        """
        return {f: os.path.getmtime(os.path.join(working_dir, f)) for f in os.listdir(working_dir)}

    def _upload_modified_files(
        self,
        working_dir: Union[str, os.PathLike],
        repo_id: str,
        files_timestamps: Dict[str, float],
        commit_message: Optional[str] = None,
        token: Optional[Union[bool, str]] = None,
        create_pr: bool = False,
    ):
        raise NotImplementedError

    def push_to_hub(
        self,
        repo_id: str,
        use_temp_dir: Optional[bool] = None,
        commit_message: Optional[str] = None,
        private: Optional[bool] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        max_shard_size: Optional[Union[int, str]] = "10GB",
        create_pr: bool = False,
        safe_serialization: bool = False,
        **deprecated_kwargs,
    ) -> str:
        raise NotImplementedError


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if organization is None:
        raise NotImplementedError("`huggingface_hub.whoami` is not available!")
    else:
        return f"{organization}/{model_id}"


def send_example_telemetry(example_name, *example_args, framework="pytorch"):
    """
    Sends telemetry that helps tracking the examples use.

    Args:
        example_name (`str`): The name of the example.
        *example_args (dataclasses or `argparse.ArgumentParser`): The arguments to the script. This function will only
            try to extract the model and dataset name from those. Nothing else is tracked.
        framework (`str`, *optional*, defaults to `"pytorch"`): The framework for the example.
    """
    if is_offline_mode():
        return

    data = {"example": example_name, "framework": framework}
    for args in example_args:
        args_as_dict = {k: v for k, v in args.__dict__.items() if not k.startswith("_") and v is not None}
        if "model_name_or_path" in args_as_dict:
            model_name = args_as_dict["model_name_or_path"]
            # Filter out local paths
            if not os.path.isdir(model_name):
                data["model_name"] = args_as_dict["model_name_or_path"]
        if "dataset_name" in args_as_dict:
            data["dataset_name"] = args_as_dict["dataset_name"]
        elif "task_name" in args_as_dict:
            # Extract script name from the example_name
            script_name = example_name.replace("tf_", "").replace("flax_", "").replace("run_", "")
            script_name = script_name.replace("_no_trainer", "")
            data["dataset_name"] = f"{script_name}-{args_as_dict['task_name']}"

    headers = {"user-agent": http_user_agent(data)}
    try:
        r = requests.head(HUGGINGFACE_CO_EXAMPLES_TELEMETRY, headers=headers)
        r.raise_for_status()
    except Exception:
        # We don't want to error in case of connection errors of any kind.
        pass


def convert_file_size_to_int(size: Union[int, str]):
    """
    Converts a size expressed as a string with digits an unit (like `"5MB"`) to an integer (in bytes).

    Args:
        size (`int` or `str`): The size to convert. Will be directly returned if an `int`.

    Example:
    ```py
    >>> convert_file_size_to_int("1MiB")
    1048576
    ```
    """
    if isinstance(size, int):
        return size
    if size.upper().endswith("GIB"):
        return int(size[:-3]) * (2**30)
    if size.upper().endswith("MIB"):
        return int(size[:-3]) * (2**20)
    if size.upper().endswith("KIB"):
        return int(size[:-3]) * (2**10)
    if size.upper().endswith("GB"):
        int_size = int(size[:-2]) * (10**9)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("MB"):
        int_size = int(size[:-2]) * (10**6)
        return int_size // 8 if size.endswith("b") else int_size
    if size.upper().endswith("KB"):
        int_size = int(size[:-2]) * (10**3)
        return int_size // 8 if size.endswith("b") else int_size
    raise ValueError("`size` is not in a valid format. Use an integer followed by the unit, e.g., '5GB'.")


def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    local_files_only=False,
    use_auth_token=None,
    user_agent=None,
    revision=None,
    subfolder="",
    _commit_hash=None,
):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.

    For the description of each arg, see [`PreTrainedModel.from_pretrained`]. `index_filename` is the full path to the
    index (downloaded and cached if `pretrained_model_name_or_path` is a model ID on the Hub).
    """
    import json

    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()

    # First, let's deal with local folder.
    if os.path.isdir(pretrained_model_name_or_path):
        shard_filenames = [os.path.join(pretrained_model_name_or_path, subfolder, f) for f in shard_filenames]
        return shard_filenames, sharded_metadata

    # At this stage pretrained_model_name_or_path is a model identifier on the Hub
    cached_filenames = []
    # Check if the model is already cached or not. We only try the last checkpoint, this should cover most cases of
    # downloaded (if interrupted).
    last_shard = try_to_load_from_cache(
        pretrained_model_name_or_path, shard_filenames[-1], cache_dir=cache_dir, revision=_commit_hash
    )
    show_progress_bar = last_shard is None or force_download
    for shard_filename in tqdm(shard_filenames, desc="Downloading shards", disable=not show_progress_bar):
        try:
            # Load from URL
            cached_filename = cached_file(
                pretrained_model_name_or_path,
                shard_filename,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=_commit_hash,
            )
        # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
        # we don't have to catch them here.
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {shard_filename} which is "
                "required according to the checkpoint index."
            )
        except HTTPError:
            raise EnvironmentError(
                f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load {shard_filename}. You should try"
                " again after checking your internet connection."
            )

        cached_filenames.append(cached_filename)

    return cached_filenames, sharded_metadata
