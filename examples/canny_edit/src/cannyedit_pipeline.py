from typing import List

import numpy as np
from PIL import Image
from src.sampling import denoise_cannyedit, get_image_tensor, get_noise, get_schedule, prepare, unpack
from src.sampling_removal import denoise_cannyedit_removal
from src.util import (
    Annotator,
    load_ae,
    load_checkpoint,
    load_clip,
    load_controlnet,
    load_flow_model,
    load_flow_model_quantized,
    load_t5,
)

import mindspore as ms


def prepare_conditional_inputs(base_input, suffix):
    """
    Helper function to process and restructure the input dictionary.
    """
    result = {}
    for key in ["txt", "txt_ids", "vec"]:
        result[f"{key}{suffix}"] = base_input[key]
    base_input.pop("img")  # Remove the key from the original dictionary
    base_input.pop("img_ids")  # Remove the key from the original dictionary
    return result


class CannyEditPipeline:
    def __init__(self, model_type, offload: bool = False):
        self.offload = offload
        self.model_type = model_type

        self.ae = load_ae(
            model_type,
        )

        self.clip = load_clip()
        self.t5 = load_t5(max_length=512)

        if "fp8" in model_type:
            self.model = load_flow_model_quantized(
                model_type,
            )
        else:
            self.model = load_flow_model(
                model_type,
            )

        self.image_encoder_path = "openai/clip-vit-large-patch14"
        self.hf_lora_collection = "XLabs-AI/flux-lora-collection"
        self.lora_types_to_names = {
            "realism": "lora.safetensors",
        }
        self.controlnet_loaded = False
        self.ip_loaded = False
        self.paint_loaded = False

    def set_controlnet(self, control_type: str, local_path: str = None, repo_id: str = None, name: str = None):
        self.controlnet = load_controlnet(
            self.model_type,
        ).to_float(ms.bfloat16)
        checkpoint = load_checkpoint(local_path, repo_id, name)
        self.controlnet.load_state_dict(checkpoint, strict=False)
        self.annotator = Annotator(
            control_type,
        )
        self.controlnet_loaded = True
        self.control_type = control_type

    def __call__(
        self,
        prompt_source: str,
        prompt_local1: str,
        prompt_target: str,
        prompt_local_addition: List[str],
        controlnet_image: Image = None,
        local_mask=None,
        local_mask_addition=[],
        width: int = 512,
        height: int = 512,
        guidance: float = 4,
        num_steps: int = 50,
        seed: int = 123456789,
        true_gs: float = 3,
        control_weight: float = 0.9,
        control_weight2: float = 0.5,
        neg_prompt: str = "",
        neg_prompt2: str = "",
        timestep_to_start_cfg: int = 0,
        generate_save_path=None,
        inversion_save_path=None,
        stage=None,
    ):
        width = 16 * (width // 16)
        height = 16 * (height // 16)

        # change: process the source image
        if self.controlnet_loaded:
            source_image = controlnet_image.copy()
            controlnet_cond = self.annotator(controlnet_image, width, height)
            controlnet_cond = ms.Tensor.from_numpy((np.array(controlnet_cond) / 127.5) - 1)
            controlnet_cond = controlnet_cond.permute(2, 0, 1).unsqueeze(0).to(ms.bfloat16)

        # change:add parameters
        return self.construct(
            prompt_source,
            prompt_local1,
            prompt_target,
            prompt_local_addition,
            local_mask,
            local_mask_addition,
            width,
            height,
            guidance,
            num_steps,
            seed,
            source_image,
            controlnet_cond,
            timestep_to_start_cfg=timestep_to_start_cfg,
            true_gs=true_gs,
            control_weight=control_weight,
            control_weight2=control_weight2,
            neg_prompt=neg_prompt,
            neg_prompt2=neg_prompt2,
            generate_save_path=generate_save_path,
            inversion_save_path=inversion_save_path,
            stage=stage,
        )

    def construct(
        self,
        prompt_source,
        prompt_local1,
        prompt_target,
        prompt_local_addition,
        local_mask,
        local_mask_addition,
        width,
        height,
        guidance,
        num_steps,
        seed,
        source_image=None,
        controlnet_cond=None,
        timestep_to_start_cfg=0,
        true_gs=3.5,
        control_weight=0.9,
        control_weight2=0.5,
        neg_prompt="",
        neg_prompt2="",
        generate_save_path=None,
        inversion_save_path=None,
        stage=None,
    ):
        x = get_noise(1, height, width, dtype=ms.bfloat16, seed=seed)

        source_image_latent = self.ae.encode(get_image_tensor(source_image, height, width, dtype=ms.float32)).to(
            ms.bfloat16
        )

        timesteps = get_schedule(
            num_steps,
            (width // 8) * (height // 8) // (16 * 16),
            shift=True,
        )

        ms.manual_seed(seed)
        with ms._no_grad():
            if self.offload:
                self.t5, self.clip = self.t5, self.clip
                self.offload_model_to_cpu(self.t5, self.clip)

            inp_cond_im = prepare(t5=self.t5, clip=self.clip, img=source_image_latent, prompt=prompt_source)

            # Prepare inputs with different prompts
            inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt="a real-world image of " + prompt_source)
            neg_inp_cond = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt)
            # removal_add
            neg_inp_cond2 = prepare(t5=self.t5, clip=self.clip, img=x, prompt=neg_prompt2)
            # Process inp_cond2, local prompt 1
            inp_cond2 = prepare(t5=self.t5, clip=self.clip, img=x, prompt="a real-world image of " + prompt_local1)
            inp_cond2 = prepare_conditional_inputs(inp_cond2, "2")
            # Process inp_cond3, target prompt
            inp_cond3 = prepare(t5=self.t5, clip=self.clip, img=x, prompt="a real-world image of " + prompt_target)
            inp_cond3 = prepare_conditional_inputs(inp_cond3, "3")
            # Process additional local prompts
            inp_cond_addition = {}
            inp_cond_addition["txt_addition"] = []
            inp_cond_addition["txt_ids_addition"] = []
            inp_cond_addition["vec_addition"] = []
            for pp in prompt_local_addition:
                inp_cond4 = prepare(t5=self.t5, clip=self.clip, img=x, prompt="a real-world image of " + str(pp))
                inp_cond_addition["txt_addition"].append(inp_cond4["txt"])
                inp_cond_addition["txt_ids_addition"].append(inp_cond4["txt_ids"])
                inp_cond_addition["vec_addition"].append(inp_cond4["vec"])

            source_image_latent_rg = inp_cond_im["img"]

            if self.offload:
                self.offload_model_to_cpu(self.t5, self.clip)

            if self.controlnet_loaded:
                if stage == "stage_generate":
                    x = denoise_cannyedit(
                        self.model,
                        **inp_cond,
                        **inp_cond2,
                        **inp_cond3,
                        **inp_cond_addition,
                        local_mask=local_mask,
                        local_mask_addition=local_mask_addition,
                        source_image_latent=source_image_latent,
                        source_image_latent_rg=source_image_latent_rg,
                        controlnet=self.controlnet,
                        timesteps=timesteps,
                        guidance=guidance,
                        controlnet_cond=controlnet_cond,
                        timestep_to_start_cfg=timestep_to_start_cfg,
                        neg_txt=neg_inp_cond["txt"],
                        neg_txt_ids=neg_inp_cond["txt_ids"],
                        neg_vec=neg_inp_cond["vec"],
                        true_gs=true_gs,
                        controlnet_gs=control_weight,
                        controlnet_gs2=control_weight2,
                        seed=seed,
                        generate_save_path=generate_save_path,
                        inversion_save_path=inversion_save_path,
                        stage=stage,
                    )
                # removal_add
                elif stage == "stage_removal":
                    x = denoise_cannyedit_removal(
                        self.model,
                        **inp_cond,
                        **inp_cond2,
                        **inp_cond3,
                        **inp_cond_addition,
                        local_mask=local_mask,
                        local_mask_addition=local_mask_addition,
                        source_image_latent=source_image_latent,
                        source_image_latent_rg=source_image_latent_rg,
                        controlnet=self.controlnet,
                        timesteps=timesteps,
                        guidance=guidance,
                        controlnet_cond=controlnet_cond,
                        timestep_to_start_cfg=timestep_to_start_cfg,
                        neg_txt=neg_inp_cond["txt"] * 0.5 + neg_inp_cond2["txt"] * 0.5,
                        neg_txt_ids=neg_inp_cond["txt_ids"],
                        neg_vec=neg_inp_cond["vec"] * 0.5 + neg_inp_cond2["vec"] * 0.5,
                        true_gs=true_gs,
                        controlnet_gs=control_weight,
                        controlnet_gs2=control_weight2,
                        seed=seed,
                        generate_save_path=generate_save_path,
                        inversion_save_path=inversion_save_path,
                        stage=stage,
                    )

            if self.offload:
                self.offload_model_to_cpu(self.model)

            x = unpack(x.float(), height, width)
            x = self.ae.decode(x)

            self.offload_model_to_cpu(self.ae.decoder)

        x1 = x.clamp(-1, 1)
        # x1 = rearrange(x1[-1], "c h w -> h w c")
        x1 = x1[-1].permute(1, 2, 0)
        output_img = Image.fromarray((127.5 * (x1 + 1.0)).asnumpy().astype(np.uint8))
        return output_img

    def offload_model_to_cpu(self, *models):
        if not self.offload:
            return
        raise NotImplementedError("Offload is not implemented")
