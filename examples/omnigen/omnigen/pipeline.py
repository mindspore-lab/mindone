from typing import List, Union

import numpy as np
from omnigen import OmniGen, OmniGenProcessor, OmniGenScheduler
from PIL import Image
from transformers.models.phi3.configuration_phi3 import Phi3Config

import mindspore as ms
from mindspore import Tensor, ops
from mindspore.nn.utils import no_init_parameters

from mindone.diffusers import AutoencoderKL
from mindone.diffusers._peft import PeftModel
from mindone.transformers.models.phi3.modeling_phi3 import Phi3LongRoPEScaledRotaryEmbedding
from mindone.utils.amp import auto_mixed_precision

from .utils import load_ckpt_params

EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> from OmniGen import OmniGenPipeline
        >>> pipe = OmniGenPipeline.from_pretrained(base_model)
        >>> prompt = "A woman holds a bouquet of flowers and faces the camera"
        >>> image = pipe(
        ...     prompt,
        ...     guidance_scale=2.5,
        ...     num_inference_steps=50,
        ... ).images[0]
        >>> image.save("t2i.png")
        ```
"""


class OmniGenPipeline:
    def __init__(
        self,
        vae: AutoencoderKL,
        model: OmniGen,
        processor: OmniGenProcessor,
    ):
        self.vae = vae
        self.model = model
        self.processor = processor
        self.model.set_train(False)
        self.vae.set_train(False)

    @classmethod
    def from_pretrained(cls, model_path, vae_path: str = None):
        config = Phi3Config.from_pretrained(model_path)
        with no_init_parameters():
            model = OmniGen(config)
        load_ckpt_params(model, "models/omnigen.ckpt")
        model = auto_mixed_precision(
            model, amp_level="O2", dtype=ms.bfloat16, custom_fp32_cells=[Phi3LongRoPEScaledRotaryEmbedding]
        )

        # Load processor
        processor = OmniGenProcessor.from_pretrained(model_path)

        # Load VAE
        vae = AutoencoderKL.from_pretrained("{}/vae".format(model_path))

        return cls(vae, model, processor)

    def merge_lora(self, lora_path: str):
        model = PeftModel.from_pretrained(self.model, lora_path)
        model.base_model.merge_and_unload()
        self.model = model.base_model

    def vae_encode(self, x, dtype):
        if self.vae.config.shift_factor is not None:
            x = self.vae.diag_gauss_dist.sample(self.vae.encode(x)[0])
            x = (x - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        else:
            x = self.vae.diag_gauss_dist.sample(self.vae.encode(x)[0])
            x = x * self.vae.config.scaling_factor
        x = x.astype(dtype)
        return x

    def __call__(
        self,
        prompt: Union[str, List[str]],
        input_images: Union[List[str], List[List[str]]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 3,
        use_img_guidance: bool = True,
        img_guidance_scale: float = 1.6,
        max_input_image_size: int = 1024,
        separate_cfg_infer: bool = True,
        use_kv_cache: bool = True,
        use_input_image_size_as_output: bool = False,
        dtype: ms.dtype = ms.float32,
        seed: int = None,
        output_type: str = "pil",
    ):
        # self.model.to_float(dtype)
        # Input validation
        if use_input_image_size_as_output:
            assert (
                isinstance(prompt, str) and len(input_images) == 1
            ), "For matching output size to input, provide single image only"
        else:
            assert height % 16 == 0 and width % 16 == 0, "Height and width must be multiples of 16"

        # Handle inputs
        if input_images is None:
            use_img_guidance = False
        if isinstance(prompt, str):
            prompt = [prompt]
            input_images = [input_images] if input_images is not None else None

        # Process inputs
        if max_input_image_size != self.processor.max_image_size:
            self.processor = OmniGenProcessor(self.processor.text_tokenizer, max_image_size=max_input_image_size)

        input_data = self.processor(
            prompt,
            input_images,
            height=height,
            width=width,
            use_img_cfg=use_img_guidance,
            separate_cfg_input=separate_cfg_infer,
            use_input_image_size_as_output=use_input_image_size_as_output,
        )
        num_prompt = len(prompt)
        num_cfg = 2 if use_img_guidance else 1

        if use_input_image_size_as_output:
            if separate_cfg_infer:
                height, width = input_data["input_pixel_values"][0][0].shape[-2:]
            else:
                height, width = input_data["input_pixel_values"][0].shape[-2:]
        latent_size_h, latent_size_w = height // 8, width // 8

        # Initialize random latents
        if seed is not None:
            np.random.seed(seed)
        latents = ops.randn((num_prompt, 4, latent_size_h, latent_size_w), dtype=dtype)
        latents = ops.cat([latents] * (1 + num_cfg))
        # latents = Tensor(latents, dtype=dtype)

        # Process input images
        input_img_latents = []
        if separate_cfg_infer:
            for temp_pixel_values in input_data["input_pixel_values"]:
                temp_input_latents = []
                for img in temp_pixel_values:
                    img = self.vae_encode(Tensor(img), dtype)
                    temp_input_latents.append(img)
                input_img_latents.append(temp_input_latents)
        else:
            for img in input_data["input_pixel_values"]:
                img = self.vae_encode(Tensor(img), dtype)
                input_img_latents.append(img)

        # Prepare model inputs
        model_kwargs = dict(
            input_ids=input_data["input_ids"],
            input_img_latents=input_img_latents,
            input_image_sizes=input_data["input_image_sizes"],
            attention_mask=input_data["attention_mask"],
            position_ids=input_data["position_ids"],
            cfg_scale=guidance_scale,
            img_cfg_scale=img_guidance_scale,
            use_img_cfg=use_img_guidance,
            use_kv_cache=use_kv_cache,
        )

        # Choose generation function
        if separate_cfg_infer:
            func = self.model.forward_with_separate_cfg
        else:
            func = self.model.forward_with_cfg

        # Generate image
        scheduler = OmniGenScheduler(num_steps=num_inference_steps)
        samples = scheduler(latents, func, model_kwargs, use_kv_cache=use_kv_cache)

        samples = samples.chunk(chunks=1 + num_cfg, axis=0)[0]
        # Decode latents
        samples = samples.to(ms.float32)
        if self.vae.config.shift_factor is not None:
            samples = samples / self.vae.config.scaling_factor + self.vae.config.shift_factor
        else:
            samples = samples / self.vae.config.scaling_factor

        samples = self.vae.decode(samples)[0]
        samples = (samples * 0.5 + 0.5).clamp(0, 1)

        # Convert to output format
        if output_type == "pt":
            output_images = samples
        else:
            samples = (samples * 255).astype(ms.uint8)
            samples = samples.permute(0, 2, 3, 1)
            samples = samples.asnumpy()
            output_images = []
            for sample in samples:
                output_images.append(Image.fromarray(sample))

        return output_images
