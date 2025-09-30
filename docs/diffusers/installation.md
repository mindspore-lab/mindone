<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Installation

Diffusers uses the `pyproject.toml` file for building and packaging, as introduced in [PEP 517](https://peps.python.org/pep-0517/). View this [configuration file](https://github.com/mindspore-lab/mindone/blob/master/pyproject.toml) file for more details on the specific build configuration of this project.

Diffusers is tested on Python 3.9+, MindSpore 2.6.0. Follow the installation instructions below for the deep learning library you are using:

- [MindSpore](https://www.mindspore.cn/install) installation instructions

Install Diffusers with one of the following methods.

<hfoptions id="install">
<hfoption id="pip">

MindSpore only supports Python 3.9 - 3.11 on Ascend.

```bash
pip install mindspore==2.7.0 -i https://repo.mindspore.cn/pypi/simple --trusted-host repo.mindspore.cn --extra-index-url https://repo.huaweicloud.com/repository/pypi/simple
```

!!! tip

    We define both required and optional dependencies in the `pyproject.toml` file. If you need to install optional dependencies, you can specify them. For example, if you want to install the 'training' optional dependency, you would run:

    ```bash
    pip install mindone[training]
    ```

Install Diffusers from source with the command below.

```bash
pip install git+https://github.com/mindspore-lab/mindone
```

</hfoption>
</hfoptions>

## Editable install

An editable install is recommended for development workflows or if you're using the `main` version of the source code. A special link is created between the cloned repository and the Python library paths. This avoids reinstalling a package after every change.

Clone the repository and install Diffusers with the following commands.

<hfoptions id="editable">
<hfoption id="MindSpore">

```bash
git clone https://github.com/mindspore-lab/mindone.git
cd mindone
```

!!! warning

    ⚠️ [PEP 660 – Editable installs for pyproject.toml based builds](https://peps.python.org/pep-0660/) defines how to build projects that only use `pyproject.toml`. Build tools must implement PEP 660 for editable installs to work. You need a front-end (such as [pip ≥ 21.3](https://pip.pypa.io/en/stable/news/#v21-3)) and a backend. The statuses of some popular backends are:

    - [Setuptools implements PEP 660 as of version 64.](https://github.com/pypa/setuptools/blob/v64.0.0/CHANGES.rst#breaking-changes)
    - [Flit implements PEP 660 as of version 3.4.](https://flit.readthedocs.io/en/latest/history.html?highlight=660#version-3-4)
    - [Poetry implements PEP 660 as of version 1.0.8.](https://github.com/python-poetry/poetry-core/releases/tag/1.0.8)

    (from: [https://stackoverflow.com/a/69711730](https://stackoverflow.com/a/69711730))

```bash
pip install -e .
```

!!! warning

    You must keep the `mindone` folder if you want to keep using the library with the editable install.

Update your cloned repository to the latest version of Diffusers with the command below.

```bash
cd ~/mindone/
git pull
```

## Cache

Model weights and files are downloaded from the Hub to a cache, which is usually your home directory. Change the cache location with the [HF_HOME](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhome) or [HF_HUB_CACHE](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubcache) environment variables or configuring the `cache_dir` parameter in methods like [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained).

<hfoptions id="cache">
<hfoption id="env variable">

```bash
export HF_HOME="/path/to/your/cache"
export HF_HUB_CACHE="/path/to/your/hub/cache"
```

</hfoption>
<hfoption id="from_pretrained">

```py
from mindone.diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    cache_dir="/path/to/your/cache"
)
```

</hfoption>
</hfoptions>

Cached files allow you to use Diffusers offline. Set the [HF_HUB_OFFLINE](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhuboffline) environment variable to `1` to prevent Diffusers from connecting to the internet.

```shell
export HF_HUB_OFFLINE=1
```

For more details about managing and cleaning the cache, take a look at the [Understand caching](https://huggingface.co/docs/huggingface_hub/guides/manage-cache) guide.

## Telemetry logging

Diffusers gathers telemetry information during [`from_pretrained`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/overview/#mindone.diffusers.DiffusionPipeline.from_pretrained) requests.
The data gathered includes the Diffusers and MindSpore version, the requested model or pipeline class,
and the path to a pretrained checkpoint if it is hosted on the Hub.

This usage data helps us debug issues and prioritize new features.
Telemetry is only sent when loading models and pipelines from the Hub,
and it is not collected if you're loading local files.

Opt-out and disable telemetry collection with the [HF_HUB_DISABLE_TELEMETRY](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisabletelemetry) environment variable.

<hfoptions id="telemetry">
<hfoption id="Linux/macOS">

```bash
export HF_HUB_DISABLE_TELEMETRY=1
```

</hfoption>
<hfoption id="Windows">

```bash
set HF_HUB_DISABLE_TELEMETRY=1
```

</hfoption>
</hfoptions>
