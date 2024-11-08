<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Configuration

Schedulers from [`SchedulerMixin`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/schedulers/overview/#mindone.diffusers.SchedulerMixin) and models from [`ModelMixin`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/models/overview/#mindone.diffusers.ModelMixin) inherit from [`ConfigMixin`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/configuration/#mindone.diffusers.configuration_utils.ConfigMixin) which stores all the parameters that are passed to their respective `__init__` methods in a JSON-configuration file.

!!! tip

    To use private or [gated](https://huggingface.co/docs/hub/models-gated#gated-models) models, log-in with `huggingface-cli login`.

::: mindone.diffusers.configuration_utils.ConfigMixin
    options:
      members:
        - load_config
        - from_config
        - save_config
        - to_json_file
        - to_json_string
