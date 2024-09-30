<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Outputs

!!! warning

    Default value of `return_dict` is changed to False and the outputs will be used as tuples, for `GRAPH_MODE` does not allow to construct an instance of it.

All model outputs are subclasses of [`~utils.BaseOutput`], data structures containing all the information returned by the model. The outputs can also be used as tuples or dictionaries.

For example:

```python
from mindone.diffusers import DDIMPipeline

pipeline = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32")
outputs = pipeline()
```

The `outputs` object is a [`ImagePipelineOutput`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/ddpm/#mindone.diffusers.pipelines.pipeline_utils.ImagePipelineOutput) which means it has an image attribute.

::: mindone.diffusers.pipelines.pipeline_utils.ImagePipelineOutput

You can access each attribute as you normally would or with a keyword lookup if you set `return_dict` to `True`, and if that attribute is not returned by the model, you will get `None`:

```python
outputs.images
outputs["images"]
```

When considering the `outputs` object as a tuple, it only considers the attributes that don't have `None` values.
For instance, retrieving an image by indexing into it returns the tuple `(outputs.images)`:

```python
outputs[:1]
```

!!! tip

    To check a specific pipeline or model output, refer to its corresponding API documentation.

::: mindone.diffusers.utils.BaseOutput
    options:
      members:
        - to_tuple

::: mindone.diffusers.pipelines.pipeline_utils.ImagePipelineOutput

::: mindone.diffusers.pipelines.pipeline_utils.AudioPipelineOutput
