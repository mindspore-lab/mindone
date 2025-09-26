<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Reproducible pipelines

Diffusion models are inherently random which is what allows it to generate different outputs every time it is run. But there are certain times when you want to generate the same output every time, like when you're testing, replicating results, and even [improving image quality](#deterministic-batch-generation). While you can't expect to get identical results across platforms, you can expect reproducible results across releases and platforms within a certain tolerance range (though even this may vary).

This guide will show you how to control randomness for deterministic generation on a Ascend.

## Control randomness

During inference, pipelines rely heavily on random sampling operations which include creating the
Gaussian noise tensors to denoise and adding noise to the scheduling step.

Take a look at the tensor values in the [`DDIMPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/ddim/#mindone.diffusers.DDIMPipeline) after two inference steps.

```python
from mindone.diffusers import DDIMPipeline
import numpy as np

ddim = DDIMPipeline.from_pretrained( "google/ddpm-cifar10-32", use_safetensors=True)
image = ddim(num_inference_steps=2, output_type="np")[0]
print(np.abs(image).sum())
```

Running the code above prints one value, but if you run it again you get a different value.

Each time the pipeline is run, [numpy.random.Generator.standard_normal](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.standard_normal.html) uses a different random seed to create the Gaussian noise tensors. This leads to a different result each time it is run and enables the diffusion pipeline to generate a different random image each time.

But if you need to reliably generate the same image, Diffusers has a [`randn_tensor`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/utilities/#mindone.diffusers.utils.mindspore_utils.randn_tensor) function for creating random noise using numpy, and then convert the array to tensor. The [`randn_tensor`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/utilities/#mindone.diffusers.utils.mindspore_utils.randn_tensor) function is used everywhere inside the pipeline. Now you can call [numpy.random.Generator](https://numpy.org/doc/stable/reference/random/generator.html) which automatically creates a `Generator` that can be passed to the pipeline.

```python
import numpy as np
from mindone.diffusers import DDIMPipeline

ddim = DDIMPipeline.from_pretrained("google/ddpm-cifar10-32", use_safetensors=True)
generator = np.random.Generator(np.random.PCG64(0))
image = ddim(num_inference_steps=2, output_type="np", generator=generator)[0]
print(np.abs(image).sum())
```

Finally, more complex pipelines such as [`UnCLIPPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/unclip/#mindone.diffusers.UnCLIPPipeline), are often extremely
susceptible to precision error propagation. You'll need to use
exactly the same hardware and MindSpore version for full reproducibility.
