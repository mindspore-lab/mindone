<!--Copyright 2024 Marigold authors and The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Marigold Computer Vision

# Marigold Computer Vision

**Marigold** is a diffusion-based [method](https://huggingface.co/papers/2312.02145) and a collection of [pipelines](../api/pipelines/marigold) designed for
dense computer vision tasks, including **monocular depth prediction**, **surface normals estimation**, and **intrinsic
image decomposition**.

This guide will walk you through using Marigold to generate fast and high-quality predictions for images and videos.

Each pipeline is tailored for a specific computer vision task, processing an input RGB image and generating a
corresponding prediction.
Currently, the following computer vision tasks are implemented:

| Pipeline                                                                                                                                          | Recommended Model Checkpoints                                                                                                                                                                           |                              Spaces (Interactive Apps)                               | Predicted Modalities                                                                                                                                                               |
|---------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------------------------------------------------------------------:|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [MarigoldDepthPipeline](../../../mindone/diffusers/pipelines/marigold/pipeline_marigold_depth.py)           | [prs-eth/marigold-depth-v1-1](https://huggingface.co/prs-eth/marigold-depth-v1-1)                                                                                                                       |          [Depth Estimation](https://huggingface.co/spaces/prs-eth/marigold)          | [Depth](https://en.wikipedia.org/wiki/Depth_map), [Disparity](https://en.wikipedia.org/wiki/Binocular_disparity)                                                                   |
| [MarigoldNormalsPipeline](../../../mindone/diffusers/pipelines/marigold/pipeline_marigold_normals.py)       | [prs-eth/marigold-normals-v1-1](https://huggingface.co/prs-eth/marigold-normals-v1-1)                                                                                                                   | [Surface Normals Estimation](https://huggingface.co/spaces/prs-eth/marigold-normals) | [Surface normals](https://en.wikipedia.org/wiki/Normal_mapping)                                                                                                                    |
| [MarigoldIntrinsicsPipeline](../../../mindone/diffusers/pipelines/marigold/pipeline_marigold_intrinsics.py) | [prs-eth/marigold-iid-appearance-v1-1](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1),<br>[prs-eth/marigold-iid-lighting-v1-1](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1) | [Intrinsic Image Decomposition](https://huggingface.co/spaces/prs-eth/marigold-iid)  | [Albedo](https://en.wikipedia.org/wiki/Albedo), [Materials](https://www.n.aiq3d.com/wiki/roughnessmetalnessao-map), [Lighting](https://en.wikipedia.org/wiki/Diffuse_reflection)   |

All original checkpoints are available under the [PRS-ETH](https://huggingface.co/prs-eth/) organization on Hugging Face.
They are designed for use with diffusers pipelines and the [original codebase](https://github.com/prs-eth/marigold), which can also be used to train
new model checkpoints.
The following is a summary of the recommended checkpoints, all of which produce reliable results with 1 to 4 steps.

| Checkpoint                                                                                          | Modality     | Comment                                                                                                                                                           |
|-----------------------------------------------------------------------------------------------------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [prs-eth/marigold-depth-v1-1](https://huggingface.co/prs-eth/marigold-depth-v1-1)                   | Depth        | Affine-invariant depth prediction assigns each pixel a value between 0 (near plane) and 1 (far plane), with both planes determined by the model during inference. |
| [prs-eth/marigold-normals-v0-1](https://huggingface.co/prs-eth/marigold-normals-v0-1)               | Normals      | The surface normals predictions are unit-length 3D vectors in the screen space camera, with values in the range from -1 to 1.                                     |
| [prs-eth/marigold-iid-appearance-v1-1](https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1) | Intrinsics   | InteriorVerse decomposition is comprised of Albedo and two BRDF material properties: Roughness and Metallicity.                                                   |
| [prs-eth/marigold-iid-lighting-v1-1](https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1)     | Intrinsics   | HyperSim decomposition of an image \\(I\\) is comprised of Albedo \\(A\\), Diffuse shading \\(S\\), and Non-diffuse residual \\(R\\): \\(I = A*S+R\\).            |

The examples below are mostly given for depth prediction, but they can be universally applied to other supported
modalities.
We showcase the predictions using the same input image of Albert Einstein generated by Midjourney.
This makes it easier to compare visualizations of the predictions across various modalities and checkpoints.

<div class="flex gap-4" style="justify-content: center; width: 100%;">
  <figure style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://marigoldmonodepth.github.io/images/einstein.jpg"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Example input image for all Marigold pipelines
    </figcaption>
  </figure>
</div>

## Depth Prediction

To get a depth prediction, load the `prs-eth/marigold-depth-v1-1` checkpoint into [`MarigoldDepthPipeline`],
put the image through the pipeline, and save the predictions:

```python
import mindone.diffusers
import mindspore as ms

pipe = mindone.diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-1", variant="fp16", mindspore_dtype=ms.float16
)

image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
depth = pipe(image)

vis = pipe.image_processor.visualize_depth(depth[0])
vis[0].save("einstein_depth.png")

depth_16bit = pipe.image_processor.export_depth_to_16bit_png(depth[0])
depth_16bit[0].save("einstein_depth_16bit.png")
```

The [`~pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_depth`] function applies one of
[matplotlib's colormaps](https://matplotlib.org/stable/users/explain/colors/colormaps.html) (`Spectral` by default) to map the predicted pixel values from a single-channel `[0, 1]`
depth range into an RGB image.
With the `Spectral` colormap, pixels with near depth are painted red, and far pixels are blue.
The 16-bit PNG file stores the single channel values mapped linearly from the `[0, 1]` range into `[0, 65535]`.
Below are the raw and the visualized predictions. The darker and closer areas (mustache) are easier to distinguish in
the visualization.

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/c03d39ad-c2b9-4f6a-bdad-61c447a9617f"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Predicted depth (16-bit PNG)
    </figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/83186dd2-2d29-488b-9d66-5d953eb06046"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Predicted depth visualization (Spectral)
    </figcaption>
  </div>
</div>

## Surface Normals Estimation

Load the `prs-eth/marigold-normals-v1-1` checkpoint into [`MarigoldNormalsPipeline`], put the image through the
pipeline, and save the predictions:

```python
import mindone.diffusers
import mindspore as ms

pipe = mindone.diffusers.MarigoldNormalsPipeline.from_pretrained(
    "prs-eth/marigold-normals-v1-1", variant="fp16", mindspore_dtype=ms.float16
)

image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
normals = pipe(image)

vis = pipe.image_processor.visualize_normals(normals[0])
vis[0].save("einstein_normals.png")
```

The [`~pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_normals`] maps the three-dimensional
prediction with pixel values in the range `[-1, 1]` into an RGB image.
The visualization function supports flipping surface normals axes to make the visualization compatible with other
choices of the frame of reference.
Conceptually, each pixel is painted according to the surface normal vector in the frame of reference, where `X` axis
points right, `Y` axis points up, and `Z` axis points at the viewer.
Below is the visualized prediction:

<div class="flex gap-4" style="justify-content: center; width: 100%;">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/528a43b7-4061-45b9-9677-2213e7ec23e7"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Predicted surface normals visualization
    </figcaption>
  </div>
</div>

In this example, the nose tip almost certainly has a point on the surface, in which the surface normal vector points straight at the viewer, meaning that its coordinates are `[0, 0, 1]`.
This vector maps to the RGB `[128, 128, 255]`, which corresponds to the violet-blue color.
Similarly, a surface normal on the cheek in the right part of the image has a large `X` component, which increases the red hue.
Points on the shoulders pointing up with a large `Y` promote green color.

## Intrinsic Image Decomposition

Marigold provides two models for Intrinsic Image Decomposition (IID): "Appearance" and "Lighting".
Each model produces Albedo maps, derived from InteriorVerse and Hypersim annotations, respectively.

- The "Appearance" model also estimates Material properties: Roughness and Metallicity.
- The "Lighting" model generates Diffuse Shading and Non-diffuse Residual.

Here is the sample code saving predictions made by the "Appearance" model:

```python
import mindone.diffusers
import mindspore

pipe = mindone.diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
    "prs-eth/marigold-iid-appearance-v1-1", variant="fp16", mindspore_dtype=mindspore.float16
)

image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

intrinsics = pipe(image)

vis = pipe.image_processor.visualize_intrinsics(intrinsics[0], pipe.target_properties)
vis[0]["albedo"].save("einstein_albedo.png")
vis[0]["roughness"].save("einstein_roughness.png")
vis[0]["metallicity"].save("einstein_metallicity.png")
```

Another example demonstrating the predictions made by the "Lighting" model:

```python
import mindone.diffusers
import mindspore

pipe = mindone.diffusers.MarigoldIntrinsicsPipeline.from_pretrained(
    "prs-eth/marigold-iid-lighting-v1-1", variant="fp16", mindspore_dtype=mindspore.float16
)

image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

intrinsics = pipe(image)

vis = pipe.image_processor.visualize_intrinsics(intrinsics[0], pipe.target_properties)
vis[0]["albedo"].save("einstein_albedo.png")
vis[0]["shading"].save("einstein_shading.png")
vis[0]["residual"].save("einstein_residual.png")
```

Both models share the same pipeline while supporting different decomposition types.
The exact decomposition parameterization (e.g., sRGB vs. linear space) is stored in the
`pipe.target_properties` dictionary, which is passed into the
[`~pipelines.marigold.marigold_image_processing.MarigoldImageProcessor.visualize_intrinsics`] function.


### Speeding up inference

The above quick start snippets are already optimized for quality and speed, loading the checkpoint, utilizing the
`fp16` variant of weights and computation, and performing the default number (4) of denoising diffusion steps.
The first step to accelerate inference, at the expense of prediction quality, is to reduce the denoising diffusion
steps to the minimum:

```diff
  import mindone.diffusers
  import mindspore

  pipe = mindone.diffusers.MarigoldDepthPipeline.from_pretrained(
      "prs-eth/marigold-depth-v1-1", variant="fp16", mindspore_dtype=mindspore.float16
  )

  image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

- depth = pipe(image)
+ depth = pipe(image, num_inference_steps=1)
```

With this change, the `pipe(image)` call completes in 180ms on Ascend Atlas 800T A2 machines in Graph mode.
Internally, the input image is first encoded using the Stable Diffusion VAE encoder, followed by a single denoising
step performed by the U-Net.
Finally, the prediction latent is decoded with the VAE decoder into pixel space.
In this setup, two out of three module calls are dedicated to converting between the pixel and latent spaces of the LDM.
Because Marigold's latent space is compatible with the base Stable Diffusion, it is possible to speed up the pipeline call by more than 3x by using a [lightweight replacement of the SD VAE](../api/models/autoencoder_tiny.md).
Note that using a lightweight VAE may slightly reduce the visual quality of the predictions.


```diff
  import mindone.diffusers
  import mindspore as ms

  pipe = mindone.diffusers.MarigoldDepthPipeline.from_pretrained(
      "prs-eth/marigold-depth-v1-1", variant="fp16", mindspore_dtype=ms.float16
  )

+ pipe.vae = mindone.diffusers.AutoencoderTiny.from_pretrained(
+     "madebyollin/taesd", mindspore_dtype=ms.float16
+ )

  image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")
  depth = pipe(image, num_inference_steps=1)
```

So far, we have optimized the number of diffusion steps and model components. Self-attention operations account for a
significant portion of computations.
Speeding them up can be achieved by using a more efficient attention processor:

```diff
  import mindone.diffusers
  import mindspore
+ from mindone.diffusers.models.attention_processor import AttnProcessor2_0

  pipe = mindone.diffusers.MarigoldDepthPipeline.from_pretrained(
      "prs-eth/marigold-depth-v1-1", variant="fp16", mindspore_dtype=mindspore.float16
  )

+ pipe.vae.set_attn_processor(AttnProcessor2_0())
+ pipe.unet.set_attn_processor(AttnProcessor2_0())

  image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

  depth = pipe(image, num_inference_steps=1)
```


## Maximizing Precision and Ensembling

Marigold pipelines have a built-in ensembling mechanism combining multiple predictions from different random latents.
This is a brute-force way of improving the precision of predictions, capitalizing on the generative nature of diffusion.
The ensembling path is activated automatically when the `ensemble_size` argument is set greater or equal than `3`.
When aiming for maximum precision, it makes sense to adjust `num_inference_steps` simultaneously with `ensemble_size`.
The recommended values vary across checkpoints but primarily depend on the scheduler type.
The effect of ensembling is particularly well-seen with surface normals:

```diff
import mindone.diffusers

pipe = mindone.diffusers.MarigoldNormalsPipeline.from_pretrained("prs-eth/marigold-normals-v1-1")
image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

- depth = pipe(image)
+ depth = pipe(image, num_inference_steps=10, ensemble_size=5)

vis = pipe.image_processor.visualize_normals(depth[0])
vis[0].save("einstein_normals.png")
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/528a43b7-4061-45b9-9677-2213e7ec23e7"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Surface normals, no ensembling
    </figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/2f2aa183-3fec-4837-9fd5-6f4f33fd23ac"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Surface normals, with ensembling
    </figcaption>
  </div>
</div>

As can be seen, all areas with fine-grained structurers, such as hair, got more conservative and on average more correct predictions.
Such a result is more suitable for precision-sensitive downstream tasks, such as 3D reconstruction.


## Frame-by-frame Video Processing with Temporal Consistency

Due to Marigold's generative nature, each prediction is unique and defined by the random noise sampled for the latent initialization.
This becomes an obvious drawback compared to traditional end-to-end dense regression networks, as exemplified in the following videos:

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/marigold/marigold_obama.gif"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">Input video</figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/7e1fe21b-402a-4058-ab5b-90216699df39"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">Marigold Depth applied to input video frames independently</figcaption>
  </div>
</div>

To address this issue, it is possible to pass `latents` argument to the pipelines, which defines the starting point of diffusion.
Empirically, we found that a convex combination of the very same starting point noise latent and the latent corresponding to the previous frame prediction give sufficiently smooth results, as implemented in the snippet below:

```python
import imageio
from PIL import Image
from tqdm import tqdm
import mindone.diffusers
from mindone.diffusers.models.attention_processor import AttnProcessor2_0
import mindspore as ms
import numpy as np

path_in = "https://huggingface.co/spaces/prs-eth/marigold-lcm/resolve/c7adb5427947d2680944f898cd91d386bf0d4924/files/video/obama.mp4"
path_out = "obama_depth.gif"

pipe = mindone.diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-1", variant="fp16", mindspore_dtype=ms.float16
)
pipe.vae = mindone.diffusers.AutoencoderTiny.from_pretrained(
    "madebyollin/taesd", mindspore_dtype=ms.float16
)
pipe.unet.set_attn_processor(AttnProcessor2_0())
pipe.set_progress_bar_config(disable=True)

with imageio.get_reader(path_in) as reader:
    size = reader.get_meta_data()['size']
    last_frame_latent = None
    latent_common = ms.Tensor(np.random.default_rng().standard_normal(
        (1, 4, 768 * size[1] // (8 * max(size)), 768 * size[0] // (8 * max(size)))
    ), dtype=ms.float16)

    out = []
    for frame_id, frame in tqdm(enumerate(reader), desc="Processing Video"):
        frame = Image.fromarray(frame)
        latents = latent_common
        if last_frame_latent is not None:
            latents = 0.9 * latents + 0.1 * last_frame_latent

        depth = pipe(
          frame, num_inference_steps=1, match_input_resolution=False, latents=latents, output_latent=True
        )
        last_frame_latent = depth[2]
        out.append(pipe.image_processor.visualize_depth(depth[0])[0])

    mindone.diffusers.utils.export_to_gif(out, path_out, fps=reader.get_meta_data()['fps'])
```

Here, the diffusion process starts from the given computed latent.
The pipeline sets `output_latent=True` to access `out.latent` and computes its contribution to the next frame's latent initialization.
The result is much more stable now:

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/7e1fe21b-402a-4058-ab5b-90216699df39"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">Marigold Depth applied to input video frames independently</figcaption>
  </div>
  <div style="flex: 1 1 50%; max-width: 50%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/43531a44-8216-43e9-b2c0-405a49c3a924"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">Marigold Depth with forced latents initialization</figcaption>
  </div>
</div>

## Marigold for ControlNet

A very common application for depth prediction with diffusion models comes in conjunction with ControlNet.
Depth crispness plays a crucial role in obtaining high-quality results from ControlNet.
As seen in comparisons with other methods above, Marigold excels at that task.
The snippet below demonstrates how to load an image, compute depth, and pass it into ControlNet in a compatible format:

```python
import mindspore as ms
import mindone.diffusers
import numpy as np

generator = np.random.Generator(np.random.PCG64(2024))
image = mindone.diffusers.utils.load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_depth_source.png"
)

pipe = mindone.diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-lcm-v1-1", mindspore_dtype=ms.float16, variant="fp16"
)

depth_image = pipe(image, generator=generator)[0]
depth_image = pipe.image_processor.visualize_depth(depth_image, color_map="binary")
depth_image[0].save("motorcycle_controlnet_depth.png")

controlnet = mindone.diffusers.ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0", mindspore_dtype=ms.float16, variant="fp16"
)
pipe = mindone.diffusers.StableDiffusionXLControlNetPipeline.from_pretrained(
    "SG161222/RealVisXL_V4.0", mindspore_dtype=ms.float16, variant="fp16", controlnet=controlnet
)
pipe.scheduler = mindone.diffusers.DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

controlnet_out = pipe(
    prompt="high quality photo of a sports bike, city",
    negative_prompt="",
    guidance_scale=6.5,
    num_inference_steps=25,
    image=depth_image,
    controlnet_conditioning_scale=0.7,
    control_guidance_end=0.7,
    generator=generator,
)[0]
controlnet_out[0].save("motorcycle_controlnet_out.png")
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div style="flex: 1 1 33%; max-width: 33%;">
    <img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/controlnet_depth_source.png"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Input image
    </figcaption>
  </div>
  <div style="flex: 1 1 33%; max-width: 33%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/a9022cf3-80f6-4019-840b-c79bb94a9829"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      Depth in the format compatible with ControlNet
    </figcaption>
  </div>
  <div style="flex: 1 1 33%; max-width: 33%;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/e6b7d7fc-c49a-4cb9-b517-7a5a18853a40"/>
    <figcaption class="mt-1 text-center text-sm text-gray-500">
      ControlNet generation, conditioned on depth and prompt: "high quality photo of a sports bike, city"
    </figcaption>
  </div>
</div>


## Quantitative Evaluation

To evaluate Marigold quantitatively in standard leaderboards and benchmarks (such as NYU, KITTI, and other datasets), follow the evaluation protocol outlined in the paper: load the full precision fp32 model and use appropriate values for `num_inference_steps` and `ensemble_size`.
Optionally seed randomness to ensure reproducibility. Maximizing `batch_size` will deliver maximum device utilization.

```python
import mindone.diffusers
import mindspore as ms
import numpy as np

seed = 2024

generator = np.random.Generator(np.random.PCG64(seed))
pipe = mindone.diffusers.MarigoldDepthPipeline.from_pretrained("prs-eth/marigold-depth-v1-1")

image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

depth = pipe(
    image,
    num_inference_steps=4,  # set according to the evaluation protocol from the paper
    ensemble_size=10,       # set according to the evaluation protocol from the paper
    generator=generator,
)

# evaluate metrics
```

## Using Predictive Uncertainty

The ensembling mechanism built into Marigold pipelines combines multiple predictions obtained from different random
latents.
As a side effect, it can be used to quantify epistemic (model) uncertainty; simply specify `ensemble_size` greater
or equal than 3 and set `output_uncertainty=True`.
The resulting uncertainty will be available in the `uncertainty` field of the output.
It can be visualized as follows:

```python
import mindnone.diffusers
import mindspore

pipe = mindone.diffusers.MarigoldDepthPipeline.from_pretrained(
    "prs-eth/marigold-depth-v1-1", variant="fp16", mindspore_dtype=mindspore.float16
)

image = mindone.diffusers.utils.load_image("https://marigoldmonodepth.github.io/images/einstein.jpg")

depth = pipe(
  image,
  ensemble_size=10,  # any number >= 3
  output_uncertainty=True,
)
# return (prediction, uncertainty, pred_latent)

uncertainty = pipe.image_processor.visualize_uncertainty(depth[1])
uncertainty[0].save("einstein_depth_uncertainty.png")
```


The interpretation of uncertainty is easy: higher values (white) correspond to pixels, where the model struggles to
make consistent predictions.
- The depth model exhibits the most uncertainty around discontinuities, where object depth changes abruptly.
- The surface normals model is least confident in fine-grained structures like hair and in dark regions such as the
collar area.
- Albedo uncertainty is represented as an RGB image, as it captures uncertainty independently for each color channel,
unlike depth and surface normals. It is also higher in shaded regions and at discontinuities.

## Conclusion

We hope Marigold proves valuable for your downstream tasks, whether as part of a broader generative workflow or for
perception-based applications like 3D reconstruction.
