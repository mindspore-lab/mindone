<!--Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# ModularPipelineBlocks

[`ModularPipelineBlocks`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_blocks#mindone.diffusers.ModularPipelineBlocks) is the basic block for building a [`ModularPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline#mindone.diffusers.ModularPipeline). It defines what components, inputs/outputs, and computation a block should perform for a specific step in a pipeline. A [`ModularPipelineBlocks`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_blocks#mindone.diffusers.ModularPipelineBlocks) connects with other blocks, using [state](https://mindspore-lab.github.io/mindone/latest/diffusers/modular_diffusers/modular_diffusers_states), to enable the modular construction of workflows.

A [`ModularPipelineBlocks`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_blocks#mindone.diffusers.ModularPipelineBlocks) on it's own can't be executed. It is a blueprint for what a step should do in a pipeline. To actually run and execute a pipeline, the [`ModularPipelineBlocks`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_blocks#mindone.diffusers.ModularPipelineBlocks) needs to be converted into a [`ModularPipeline`].

This guide will show you how to create a [`ModularPipelineBlocks`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_blocks#mindone.diffusers.ModularPipelineBlocks).

## Inputs and outputs

!!! tip
    Refer to the [States](https://mindspore-lab.github.io/mindone/latest/diffusers/modular_diffusers/modular_diffusers_states) guide if you aren't familiar with how state works in Modular Diffusers.

A [`ModularPipelineBlocks`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_blocks#mindone.diffusers.ModularPipelineBlocks) requires `inputs`, and `intermediate_outputs`.

- `inputs` are values provided by a user and retrieved from the [`PipelineState`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_state#mindone.diffusers.modular_pipelines.PipelineState). This is useful because some workflows resize an image, but the original image is still required. The [`PipelineState`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_state#mindone.diffusers.modular_pipelines.PipelineState) maintains the original image.

    Use `InputParam` to define `inputs`.

    ```py
    from mindone.diffusers.modular_pipelines import InputParam

    user_inputs = [
        InputParam(name="image", type_hint="PIL.Image", description="raw input image to process")
    ]
    ```

- `intermediate_inputs` are values typically created from a previous block but it can also be directly provided if no preceding block generates them. Unlike `inputs`, `intermediate_inputs` can be modified.

    Use `InputParam` to define `intermediate_inputs`.

    ```py
    user_intermediate_inputs = [
        InputParam(name="processed_image", type_hint="torch.Tensor", description="image that has been preprocessed and normalized"),
    ]
    ```

- `intermediate_outputs` are new values created by a block and added to the [`PipelineState`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_state#mindone.diffusers.modular_pipelines.PipelineState). The `intermediate_outputs` are available as `intermediate_inputs` for subsequent blocks or available as the final output from running the pipeline.

    Use `OutputParam` to define `intermediate_outputs`.

    ```py
    from mindone.diffusers.modular_pipelines import OutputParam

        user_intermediate_outputs = [
        OutputParam(name="image_latents", description="latents representing the image")
    ]
    ```

The intermediate inputs and outputs share data to connect blocks. They are accessible at any point, allowing you to track the workflow's progress.

## Computation logic

The computation a block performs is defined in the `__call__` method and it follows a specific structure.

1. Retrieve the [`BlockState`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_blocks#mindone.diffusers.modular_pipelines.BlockState) to get a local view of the `inputs` and `intermediate_inputs`.
2. Implement the computation logic on the `inputs` and `intermediate_inputs`.
3. Update [`PipelineState`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_states#mindone.diffusers.modular_pipelines.PipelineState) to push changes from the local [`BlockState`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_blocks#mindone.diffusers.modular_pipelines.BlockState) back to the global [`PipelineState`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_states#mindone.diffusers.modular_pipelines.PipelineState).
4. Return the components and state which becomes available to the next block.

```py
def __call__(self, components, state):
    # Get a local view of the state variables this block needs
    block_state = self.get_block_state(state)

    # Your computation logic here
    # block_state contains all your inputs and intermediate_inputs
    # Access them like: block_state.image, block_state.processed_image

    # Update the pipeline state with your updated block_states
    self.set_block_state(state, block_state)
    return components, state
```

### Components and configs

The components and pipeline-level configs a block needs are specified in [`ComponentSpec`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_components#mindone.diffusers.ComponentSpec) and [`ConfigSpec`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_components#mindone.diffusers.modular_pipelines.ConfigSpec).

- [`ComponentSpec`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_components#mindone.diffusers.ComponentSpec) contains the expected components used by a block. You need the `name` of the component and ideally a `type_hint` that specifies exactly what the component is.
- [`ConfigSpec`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/modular_diffusers/pipeline_components#mindone.diffusers.modular_pipelines.ConfigSpec) contains pipeline-level settings that control behavior across all blocks.

```py
from mindone.diffusers import ComponentSpec, ConfigSpec

expected_components = [
    ComponentSpec(name="unet", type_hint=UNet2DConditionModel),
    ComponentSpec(name="scheduler", type_hint=EulerDiscreteScheduler)
]

expected_config = [
    ConfigSpec("force_zeros_for_empty_prompt", True)
]
```

When the blocks are converted into a pipeline, the components become available to the block as the first argument in `__call__`.

```py
def __call__(self, components, state):
    # Access components using dot notation
    unet = components.unet
    vae = components.vae
    scheduler = components.scheduler
```
