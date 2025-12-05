# Copyright 2020 The HuggingFace Team. All rights reserved.
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
Integrations with other Python libraries.
"""

import importlib.metadata
import importlib.util
import os
from typing import TYPE_CHECKING

import packaging.version

if os.getenv("WANDB_MODE") == "offline":
    print("⚙️  Running in WANDB offline mode")

from ..utils import logging

logger = logging.get_logger(__name__)

# comet_ml requires to be imported before any ML frameworks
_MIN_COMET_VERSION = "3.43.2"
try:
    _comet_version = importlib.metadata.version("comet_ml")
    _is_comet_installed = True

    _is_comet_recent_enough = packaging.version.parse(_comet_version) >= packaging.version.parse(_MIN_COMET_VERSION)

    # Check if the Comet API Key is set
    import comet_ml

    if comet_ml.config.get_config("comet.api_key") is not None:
        _is_comet_configured = True
    else:
        _is_comet_configured = False
except (importlib.metadata.PackageNotFoundError, ImportError, ValueError, TypeError, AttributeError, KeyError):
    _comet_version = None
    _is_comet_installed = False
    _is_comet_recent_enough = False
    _is_comet_configured = False

_has_neptune = importlib.util.find_spec("neptune") is not None or importlib.util.find_spec("neptune-client") is not None
if TYPE_CHECKING and _has_neptune:
    try:
        _neptune_version = importlib.metadata.version("neptune")
        logger.info(f"Neptune version {_neptune_version} available.")
    except importlib.metadata.PackageNotFoundError:
        try:
            _neptune_version = importlib.metadata.version("neptune-client")
            logger.info(f"Neptune-client version {_neptune_version} available.")
        except importlib.metadata.PackageNotFoundError:
            _has_neptune = False

from transformers.utils import ENV_VARS_TRUE_VALUES  # noqa: E402


# Integration functions:
def is_wandb_available():
    # any value of WANDB_DISABLED disables wandb
    if os.getenv("WANDB_DISABLED", "").upper() in ENV_VARS_TRUE_VALUES:
        logger.warning(
            "Using the `WANDB_DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False
    if importlib.util.find_spec("wandb") is not None:
        import wandb

        # wandb might still be detected by find_spec after an uninstall (leftover files or metadata), but not actually
        # import correctly. To confirm it's fully installed and usable, we check for a key attribute like "run".
        return hasattr(wandb, "run")
    else:
        return False


def is_trackio_available():
    return importlib.util.find_spec("trackio") is not None


def is_clearml_available():
    return importlib.util.find_spec("clearml") is not None


def is_comet_available():
    if os.getenv("COMET_MODE", "").upper() == "DISABLED":
        logger.warning(
            "Using the `COMET_MODE=DISABLED` environment variable is deprecated and will be removed in v5. Use the "
            "--report_to flag to control the integrations used for logging result (for instance --report_to none)."
        )
        return False

    if _is_comet_installed is False:
        return False

    if _is_comet_recent_enough is False:
        logger.warning(
            "comet_ml version %s is installed, but version %s or higher is required. "
            "Please update comet_ml to the latest version to enable Comet logging with pip install 'comet-ml>=%s'.",
            _comet_version,
            _MIN_COMET_VERSION,
            _MIN_COMET_VERSION,
        )
        return False

    if _is_comet_configured is False:
        logger.warning(
            "comet_ml is installed but the Comet API Key is not configured. "
            "Please set the `COMET_API_KEY` environment variable to enable Comet logging. "
            "Check out the documentation for other ways of configuring it: "
            "https://www.comet.com/docs/v2/guides/experiment-management/configure-sdk/#set-the-api-key"
        )
        return False

    return True


def is_tensorboard_available():
    return importlib.util.find_spec("tensorboard") is not None or importlib.util.find_spec("tensorboardX") is not None


def is_optuna_available():
    return importlib.util.find_spec("optuna") is not None


def is_ray_available():
    return importlib.util.find_spec("ray") is not None


def is_ray_tune_available():
    if not is_ray_available():
        return False
    return importlib.util.find_spec("ray.tune") is not None


def is_sigopt_available():
    return importlib.util.find_spec("sigopt") is not None


def is_azureml_available():
    if importlib.util.find_spec("azureml") is None:
        return False
    if importlib.util.find_spec("azureml.core") is None:
        return False
    return importlib.util.find_spec("azureml.core.run") is not None


def is_mlflow_available():
    if os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return importlib.util.find_spec("mlflow") is not None


def is_dagshub_available():
    return None not in [importlib.util.find_spec("dagshub"), importlib.util.find_spec("mlflow")]


def is_neptune_available():
    return _has_neptune


def is_codecarbon_available():
    return importlib.util.find_spec("codecarbon") is not None


def is_flytekit_available():
    return importlib.util.find_spec("flytekit") is not None


def is_flyte_deck_standard_available():
    if not is_flytekit_available():
        return False
    return importlib.util.find_spec("flytekitplugins.deck") is not None


def is_dvclive_available():
    return importlib.util.find_spec("dvclive") is not None


def is_swanlab_available():
    return importlib.util.find_spec("swanlab") is not None


def get_available_reporting_integrations():
    integrations = []
    if is_azureml_available() and not is_mlflow_available():
        integrations.append("azure_ml")
    if is_comet_available():
        integrations.append("comet_ml")
    if is_dagshub_available():
        integrations.append("dagshub")
    if is_dvclive_available():
        integrations.append("dvclive")
    if is_mlflow_available():
        integrations.append("mlflow")
    if is_neptune_available():
        integrations.append("neptune")
    if is_tensorboard_available():
        integrations.append("tensorboard")
    if is_wandb_available():
        integrations.append("wandb")
    if is_codecarbon_available():
        integrations.append("codecarbon")
    if is_clearml_available():
        integrations.append("clearml")
    if is_swanlab_available():
        integrations.append("swanlab")
    if is_trackio_available():
        integrations.append("trackio")
    return integrations


def rewrite_logs(d):
    new_d = {}
    eval_prefix = "eval_"
    eval_prefix_len = len(eval_prefix)
    test_prefix = "test_"
    test_prefix_len = len(test_prefix)
    for k, v in d.items():
        if k.startswith(eval_prefix):
            new_d["eval/" + k[eval_prefix_len:]] = v
        elif k.startswith(test_prefix):
            new_d["test/" + k[test_prefix_len:]] = v
        else:
            new_d["train/" + k] = v
    return new_d


INTEGRATION_TO_CALLBACK = {}


def get_reporting_integration_callbacks(report_to):
    if report_to is None:
        return []

    if isinstance(report_to, str):
        if "none" == report_to:
            return []
        elif "all" == report_to:
            report_to = get_available_reporting_integrations()
        else:
            report_to = [report_to]

    for integration in report_to:
        if integration not in INTEGRATION_TO_CALLBACK:
            raise ValueError(
                f"{integration} is not supported, only {', '.join(INTEGRATION_TO_CALLBACK.keys())} are supported."
            )

    return [INTEGRATION_TO_CALLBACK[integration] for integration in report_to]
