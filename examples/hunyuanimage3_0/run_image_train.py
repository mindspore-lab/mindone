"""
Hunyuan-Image model fine-tuning script using LoRA.

This script with default values fine-tunes a pretrained model from Hunyuan-Image,
on the `lambdalabs/pokemon-blip-captions` dataset for pokemon-style image generation.

Usage (multi-NPUs in mindspore 2.7.0):
```
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NPUS=8
MASTER_PORT=9000
OUTPUT_DIR=outputs/lora
msrun --worker_num=${NPUS} --local_worker_num=${NPUS} --master_port=${MASTER_PORT} --log_dir="logs/train" --join=True \
python run_image_train.py \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --save_strategy no \
    --bf16
```
"""

import io
import logging
import math
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import numpy as np
from datasets import load_dataset
from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM
from PIL import Image
from transformers import HfArgumentParser

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore import mint, nn, ops

from mindone.diffusers.training_utils import pynative_no_grad
from mindone.peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from mindone.safetensors.mindspore import load_file, save_file
from mindone.trainers import create_optimizer
from mindone.transformers.mindspore_adapter import MindSporeArguments, init_environment
from mindone.transformers.modeling_outputs import CausalLMOutputWithPast
from mindone.transformers.optimization import get_scheduler
from mindone.transformers.trainer import Trainer
from mindone.transformers.training_args import TrainingArguments
from mindone.utils.amp import auto_mixed_precision

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100


@dataclass
class MyArguments(MindSporeArguments, TrainingArguments):
    amp_opt_level: str = field(default="O3")
    attn_impl: str = field(
        default="flash_attention_2", metadata={"help": "Attention implementation. sdpa is not supported yet"}
    )
    dataset_path: str = field(default="lambdalabs/pokemon-blip-captions")
    deepspeed: str = field(default="scripts/zero3.json")
    device_target: str = field(default="Ascend")
    do_eval: bool = field(default=False)
    enable_flash_attention: bool = field(default=True)
    is_distribute: bool = field(default=False)
    lora_rank: int = field(default=4, metadata={"help": "The dimension of the LoRA update matrices."})
    lora_alpha: int = field(default=16, metadata={"help": "The scaling factor alpha of the LoRA."})
    mode: int = field(default=ms.PYNATIVE_MODE, metadata={"help": "Graph(not supported)/Pynative"})
    model_path: str = field(default="HunyuanImage-3")
    moe_impl: str = field(default="eager", metadata={"help": "MoE implementation."})
    output_dir: str = field(default="./outputs/")
    save_strategy: str = field(default="no", metadata={"help": "Save strategy, no, steps or epoch."})
    seed: int = field(default=42)
    max_device_memory: str = field(
        default="59GB", metadata={"help": "30GB for 910, 59GB for Ascend Atlas 800T A2 machines"}
    )
    per_device_train_batch_size: int = field(default=1, metadata={"help": "batch size per device for training"})
    num_train_epochs: int = field(default=1, metadata={"help": "number of training epochs"})


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    height: int = field(default=512)
    width: int = field(default=512)
    max_length: int = field(default=2048, metadata={"help": "Fixed token length for training."})
    num_inference_steps: int = field(default=2, metadata={"help": "Inference steps when denoising in training."})
    guidance_scale: float = field(default=5.0)
    guidance_rescale: float = field(default=0.0)


@dataclass
class CausalMMOutputWithPast(CausalLMOutputWithPast):
    diffusion_prediction: Optional[ms.Tensor] = None


def slice_to_list(s):
    step = s.step if s.step is not None else 1
    return [s.start, s.stop, step]


def list_to_slice(lst):
    start, stop, step = lst
    return slice(start, stop, step)


def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False


def main():
    parser = HfArgumentParser((MyArguments, DataArguments))
    args, data_args = parser.parse_args_into_dataclasses()

    init_environment(args)

    dist.init_process_group()
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)

    args.rank_size = world_size
    args.rank = local_rank
    args.zero_stage = 3

    # 1. Load materials
    # 1.1 Load pretrained model from pipe
    dtype = ms.bfloat16 if args.bf16 else (ms.float16 if args.fp16 else ms.float32)
    kwargs = dict(
        attn_implementation=args.attn_impl,
        dtype=dtype,
        moe_impl=args.moe_impl,
    )
    with nn.no_init_parameters():
        parent_model = HunyuanImage3ForCausalMM.from_pretrained(args.model_path, **kwargs)
    parent_model.load_tokenizer(args.model_path)
    parent_model.vae = auto_mixed_precision(
        parent_model.vae, amp_level="O2", dtype=ms.bfloat16, custom_fp32_cells=[nn.GroupNorm]
    )
    ms.amp.auto_mixed_precision(parent_model, amp_level="auto", dtype=ms.bfloat16)

    # 1.2 Update data_args from model.config
    data_args.config = parent_model.config
    data_args.generation_config = parent_model.generation_config
    data_args.vae_config = parent_model.vae.config
    data_args.vae_config.has_ffactor_temporal = hasattr(parent_model.vae, "ffactor_temporal")
    data_args.ms_dtype = dtype
    data_args.seed = args.seed

    # 1.3 the dataset
    dataset = load_dataset("parquet", data_dir=args.dataset_path, split="train")
    dataset = dataset.shuffle(seed=args.seed)

    total_size = len(dataset)
    train_size = int(total_size * 0.8)

    train_indices = list(range(train_size))
    eval_indices = list(range(train_size, total_size))

    def process_function(examples):
        # prepare the inputs
        prompt = examples["text"]
        image_size = f"{data_args.width}x{data_args.height}"
        model_inputs = parent_model.prepare_model_inputs(
            prompt=prompt,
            mode="gen_image",
            seed=args.seed,
            image_size=image_size,
            max_length=data_args.max_length,
            add_pad=True,
        )
        # input_ids, position_ids, past_key_values, custom_pos_emb, tokenizer_output, batch_gen_image_info

        # convert slice dtype to dict: slice(15, 4111, None) -> [15, 4111, 1]
        def replace_slices(obj):
            if isinstance(obj, slice):
                return slice_to_list(obj)
            elif isinstance(obj, list):
                return [replace_slices(item) for item in obj]
            else:
                return obj

        joint_image_slices = replace_slices(model_inputs["tokenizer_output"].joint_image_slices)
        gen_image_slices = replace_slices(model_inputs["tokenizer_output"].gen_image_slices)

        # prepare the labels: convert the image to latent space
        image = (
            Image.open(io.BytesIO(examples["image"]["bytes"]))
            .convert("RGB")
            .resize((data_args.width, data_args.height), Image.Resampling.BILINEAR)  # original (1280, 1280)
        )

        pixel_values = ms.Tensor(np.array(image, dtype=np.float32)) / 255.0  # (H, W, C) = (512, 512, 3)
        pixel_values = pixel_values.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)

        # write to dataset
        examples["input_ids"] = model_inputs["input_ids"]
        examples["custom_pos_emb"] = model_inputs["custom_pos_emb"]
        examples["joint_image_slices"] = joint_image_slices
        examples["gen_image_slices"] = gen_image_slices
        examples["image_mask"] = model_inputs["image_mask"]
        examples["gen_timestep_scatter_index"] = model_inputs["gen_timestep_scatter_index"]

        examples["labels"] = pixel_values

        if not args.do_eval:
            examples.pop("text")  # remove text from examples
            examples.pop("image")  # remove image from examples

        return examples

    tokenized_datasets = dataset.map(process_function, batched=False)
    train_dataset = tokenized_datasets.select(train_indices)
    eval_dataset = tokenized_datasets.select(eval_indices)

    dataset_len = len(train_dataset)
    num_update_steps_per_epoch = max(1, dataset_len // args.gradient_accumulation_steps)
    num_training_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)

    # 2. Prepare for LoRA
    # 2.1. Determine the target model
    model = parent_model.model  # -> HunyuanImage3Model
    model.config.use_cache = False
    model.gradient_checkpointing = True
    model.training = True
    freeze_params(model)
    freeze_params(parent_model)
    freeze_params(parent_model.vae)
    freeze_params(parent_model.vision_model)

    parent_components = SimpleNamespace(
        build_image_info=parent_model.image_processor.build_image_info,
        prepare_model_inputs=parent_model.prepare_model_inputs,
        prepare_latents=parent_model.pipeline.prepare_latents,
        prepare_mask_for_gen=parent_model._prepare_attention_mask_for_generation,
        prepare_inputs_for_gen=parent_model.prepare_inputs_for_generation,
        instantiate_timestep_tokens=parent_model.instantiate_timestep_tokens,
        instantiate_vae_image_tokens=parent_model.instantiate_vae_image_tokens,
        time_embed=parent_model.time_embed,
        patch_embed=parent_model.patch_embed,
        timestep_emb=parent_model.timestep_emb,
        ragged_final_layer=parent_model.ragged_final_layer,
        update_kwargs_for_gen=parent_model._update_model_kwargs_for_generation,
        vae_decode=parent_model.vae.decode,
        scheduler=parent_model.scheduler,
        postprocess=parent_model.pipeline.image_processor.postprocess,
    )

    # 2.2. Prepare the LoRA config
    # all attn linear layers
    vision_modules = []
    vae_modules = []
    transformer_attn_modules = []
    for i in range(data_args.config.num_hidden_layers):
        # temporarily close due to OOM risk
        # transformer_attn_modules.append(f"layers.{i}.self_attn.qkv_proj")
        # transformer_attn_modules.append(f"layers.{i}.self_attn.o_proj")
        for j in range(data_args.config.num_experts):
            # transformer_attn_modules.append(f"layers.{i}.mlp.experts.{j}.gate_and_up_proj")
            transformer_attn_modules.append(f"layers.{i}.mlp.experts.{j}.down_proj")

    target_modules = vision_modules + vae_modules + transformer_attn_modules
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. [optional] Prepare the evalutaion metric
    compute_metrics = None

    # 4. Training setups: lr scheduler, optimizer, trainer, etc.
    # lr scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler_type,
        base_lr=args.learning_rate,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps,
        scheduler_specific_kwargs=args.lr_scheduler_kwargs,
    )
    # optimizer
    optimizer_kwargs = {
        "name": "adamw",
        "betas": (args.adam_beta1, args.adam_beta2),
        "eps": args.adam_epsilon,
        "lr": lr_scheduler,
    }
    optimizer = create_optimizer(model.get_base_model().trainable_params(), **optimizer_kwargs)

    # trainer
    trainer = Trainer(
        model=TrainStepForHunyuanImage(
            model.get_base_model(),  # use base model for parsing construct() arguments
            parent_components,
            data_args,
        ),
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, lr_scheduler),  # for LoRA
    )  # do not support passing compute_loss yet

    # train the model and save the LoRA weights
    def save_lora_model(model, output_dir):
        if args.zero_stage == 3:
            all_gather_op = ops.AllGather()

            transformer_lora_layers_to_save_new = {}
            transformer_lora_layers_to_save = get_peft_model_state_dict(model)

            for name, param in transformer_lora_layers_to_save.items():
                if name.startswith("base_model.model."):
                    name = name.replace("base_model.model.", "")
                data = ms.Tensor(all_gather_op(param).asnumpy())
                transformer_lora_layers_to_save_new[name] = data

            if args.rank == 0:
                output_path = os.path.join(output_dir, "adapter_model.safetensors")
                os.makedirs(output_dir, exist_ok=True)
                save_file(transformer_lora_layers_to_save_new, output_path)

        else:
            model.save_pretrained(output_dir, safe_serialization=True)

        print(f"Lora model has been saved in {output_dir}.")

    if trainer.args.num_train_epochs > 0:
        trainer.train()
        trainer.save_state()
        ms.runtime.synchronize()
        save_lora_model(model, os.path.join(args.output_dir, "lora"))

    # 5. Inference and evaluation
    if args.do_eval:  # FIXME: bf16 not supported yet
        print("Start fusing lora weights into pipe and doing eval.")

        # delete the trained model
        def clear_workspace(model):
            import gc

            del model
            gc.collect()
            ms.hal.empty_cache()

        # loading function
        def load_lora_model(input_dir):
            # get the saved weights
            lora_state_dict = load_file(input_dir)
            # add to the dict of model
            set_peft_model_state_dict(parent_model, lora_state_dict, adapter_name="default")
            parent_model.merge_and_unload()
            parent_model.set_train(False)

        clear_workspace(model)
        load_lora_model(os.path.join(args.output_dir, "lora/adapter_model.safetensors"))

        # inference function
        def inference(txt):
            image = parent_model.generate_image(
                prompt=txt,
                image_size=f"{data_args.width}x{data_args.height}",
                num_inference_steps=8,
                seed=args.seed,
            )
            return image

        def calculate_pixel_error(img1, img2):
            arr1 = np.array(img1, dtype=np.float32)
            arr2 = np.array(img2, dtype=np.float32)
            return np.mean(np.abs(arr1 - arr2))

        for idx, example in enumerate(eval_dataset):
            infer_image = inference(example["text"])
            infer_image_save_path = os.path.join(args.output_dir, f"infer_{idx}.png")
            infer_image.save(infer_image_save_path)

            width, height = data_args.width, data_args.height
            ref_image = Image.open(io.BytesIO(example["image"]["bytes"])).convert("RGB").resize((width, height))
            ref_image_save_path = os.path.join(args.output_dir, f"ref_{idx}.png")
            ref_image.save(ref_image_save_path)
            error = calculate_pixel_error(infer_image, ref_image)

            log_entry = f"Generation: #{idx} in {infer_image_save_path} with pixel errors {error:.2}\n"
            with open(os.path.join(args.output_dir, "results.txt"), "a") as f:
                print(log_entry.strip(), file=f)


class ClassifierFreeGuidance:
    def __init__(
        self,
        use_original_formulation: bool = False,
        start: float = 0.0,
        stop: float = 1.0,
    ):
        super().__init__()
        self.use_original_formulation = use_original_formulation

    def __call__(
        self,
        pred_cond: ms.Tensor,
        pred_uncond: Optional[ms.Tensor],
        guidance_scale: float,
        step: int,
    ) -> ms.Tensor:
        shift = pred_cond - pred_uncond
        pred = pred_cond if self.use_original_formulation else pred_uncond
        pred = pred + guidance_scale * shift

        return pred


class TrainStepForHunyuanImage(nn.Cell):
    BatchRaggedImages = Union[ms.Tensor, List[Union[ms.Tensor, List[ms.Tensor]]]]
    BatchRaggedTensor = Union[ms.Tensor, List[ms.Tensor]]

    def __init__(self, base_model, parent_components, data_args):
        super().__init__()
        self.base_model = base_model
        self.build_image_info = parent_components.build_image_info
        self.instantiate_timestep_tokens = parent_components.instantiate_timestep_tokens
        self.instantiate_vae_image_tokens = parent_components.instantiate_vae_image_tokens
        self.time_embed = parent_components.time_embed
        self.patch_embed = parent_components.patch_embed
        self.timestep_emb = parent_components.timestep_emb
        self.ragged_final_layer = parent_components.ragged_final_layer

        self.prepare_model_inputs = parent_components.prepare_model_inputs
        self.prepare_latents = parent_components.prepare_latents
        self.prepare_mask_for_gen = parent_components.prepare_mask_for_gen
        self.prepare_inputs_for_gen = parent_components.prepare_inputs_for_gen
        self.update_kwargs_for_gen = parent_components.update_kwargs_for_gen
        self.vae_decode = parent_components.vae_decode
        self.scheduler = parent_components.scheduler
        self.postprocess = parent_components.postprocess

        self.cfg_operator = ClassifierFreeGuidance()

        self.args = data_args

    @staticmethod
    def retrieve_timesteps(
        scheduler,
        num_inference_steps: Optional[int] = None,
    ):
        """
        Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
        custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

        Args:
            scheduler (`SchedulerMixin`):
                The scheduler to get timesteps from.
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
                must be `None`.

        Returns:
            `Tuple[ms.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
            second element is the number of inference steps.
        """

        scheduler.set_timesteps(num_inference_steps)
        timesteps = scheduler.timesteps
        return timesteps

    @staticmethod
    def _prepare_attention_mask_for_generation(
        inputs_tensor: ms.Tensor,
        tokenizer_output: OrderedDict,
    ) -> ms.Tensor:
        # create `4d` bool attention mask (b, 1, seqlen, seqlen) using this implementation to bypass the 2d requirement
        # in the `transformers.generation_utils.GenerationMixin.generate`.
        # This implementation can handle sequences with text and image modalities, where text tokens use causal
        # attention and image tokens use full attention.
        bsz, seq_len = inputs_tensor.shape
        batch_image_slices = [
            tokenizer_output.joint_image_slices[i] + tokenizer_output.gen_image_slices[i] for i in range(bsz)
        ]
        attention_mask = mint.ones((seq_len, seq_len), dtype=ms.bool_).tril(diagonal=0).repeat(bsz, 1, 1)
        for i in range(bsz):
            for j, image_slice in enumerate(batch_image_slices[i]):
                attention_mask[i, image_slice, image_slice] = True
        attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    @staticmethod
    def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
        r"""
        Rescales `noise_cfg` tensor based on `guidance_rescale` to improve image quality and fix overexposure. Based on
        Section 3.4 from [Common Diffusion Noise Schedules and Sample Steps are
        Flawed](https://arxiv.org/pdf/2305.08891.pdf).

        Args:
            noise_cfg (`ms.Tensor`):
                The predicted noise tensor for the guided diffusion process.
            noise_pred_text (`ms.Tensor`):
                The predicted noise tensor for the text-guided diffusion process.
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                A rescale factor applied to the noise predictions.
        Returns:
            noise_cfg (`ms.Tensor`): The rescaled noise prediction tensor.
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    @staticmethod
    def load_slices(tokenizer_output):
        """
        Convert dict-type data into the expected slice-type one
        """

        def restore_slices(obj):
            if len(obj) == 3:
                return list_to_slice(obj)
            else:
                return [restore_slices(item) for item in obj]

        return restore_slices(tokenizer_output)

    @staticmethod
    def real_batched_index_select(t, dim, idx):
        """index_select for batched index and batched t"""
        assert t.ndim >= 2 and idx.ndim >= 2, f"{t.ndim=} {idx.ndim=}"
        assert len(t) == len(idx), f"{len(t)=} != {len(idx)=}"
        return mint.stack([mint.index_select(t[i], dim - 1, idx[i]) for i in range(len(t))])

    def get_pos_emb(self, custom_pos_emb, position_ids):
        cos, sin = custom_pos_emb
        cos = self.real_batched_index_select(cos, dim=1, idx=position_ids)
        sin = self.real_batched_index_select(sin, dim=1, idx=position_ids)
        return cos, sin

    # The generation method is modified from HunyuanImage3ForCausalMM construct method
    def generation(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        custom_pos_emb: Optional[Tuple[ms.Tensor]] = None,
        mode: str = "gen_text",
        first_step: Optional[bool] = None,
        # for gen image
        images: Optional[BatchRaggedImages] = None,
        image_mask: Optional[ms.Tensor] = None,
        timestep: Optional[BatchRaggedTensor] = None,
        gen_timestep_scatter_index: Optional[ms.Tensor] = None,
        # for cond image
        cond_vae_images: Optional[BatchRaggedImages] = None,
        cond_timestep: Optional[BatchRaggedTensor] = None,
        cond_vae_image_mask: Optional[ms.Tensor] = None,
        cond_vit_images: Optional[BatchRaggedImages] = None,
        cond_vit_image_mask: Optional[ms.Tensor] = None,
        vit_kwargs: Optional[Dict[str, Any]] = None,
        cond_timestep_scatter_index: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, CausalMMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.args.config.use_return_dict

        custom_pos_emb = self.get_pos_emb(custom_pos_emb, position_ids)

        inputs_embeds = self.base_model.wte(input_ids)
        bsz, seq_len, n_embd = inputs_embeds.shape  # 2, 2048, 8196

        # Instantiate placeholder tokens: <timestep>, <img> for the gen image
        with pynative_no_grad():
            if mode == "gen_text":
                raise NotImplementedError("Not supported yet")
            else:
                if first_step:
                    inputs_embeds, token_h, token_w = self.instantiate_vae_image_tokens(
                        inputs_embeds, images, timestep, image_mask
                    )
                    inputs_embeds = self.instantiate_timestep_tokens(
                        inputs_embeds, timestep, gen_timestep_scatter_index
                    )
                else:
                    t_emb = self.time_embed(timestep)
                    image_emb, token_h, token_w = self.patch_embed(images, t_emb)
                    timestep_emb = self.timestep_emb(timestep).reshape(bsz, -1, n_embd)
                    inputs_embeds = mint.cat([timestep_emb, image_emb], dim=1)

        # Should only run once with kv-cache enabled.
        if cond_vae_images is not None or cond_vit_images is not None:
            raise NotImplementedError("Not supported yet")

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            custom_pos_emb=custom_pos_emb,
            mode=mode,
            first_step=first_step,
            gen_timestep_scatter_index=gen_timestep_scatter_index,
        )
        hidden_states = outputs[0]
        logits = None
        diffusion_prediction = self.ragged_final_layer(
            hidden_states, image_mask, timestep, token_h, token_w, first_step
        )

        if not return_dict:
            output = (logits,) + outputs[1:] + (diffusion_prediction,)
            return output

        output = CausalMMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            diffusion_prediction=diffusion_prediction,
        )

        return output

    # The construct method is modified from HunyuanImage3Text2ImagePipeline for finetune calculations
    def construct(
        self,
        input_ids,
        custom_pos_emb,
        joint_image_slices,
        gen_image_slices,
        image_mask,
        gen_timestep_scatter_index,
        labels,
        *args,
    ):
        # prepare inputs
        mode = "gen_image"

        input_ids = input_ids[0]

        image_size = f"{self.args.width}x{self.args.height}"
        batch_size = len([input_ids])

        batch_gen_image_info = [self.build_image_info(image_size) for _ in range(batch_size)]
        batch_input_pos = mint.arange(0, input_ids.shape[1], dtype=ms.int64)[None].expand(
            (batch_size * 2, -1)  # (batch_size * cfg_factor[mode], -1)
        )

        tokenizer_output = {}
        tokenizer_output["joint_image_slices"] = self.load_slices(joint_image_slices[0])
        tokenizer_output["gen_image_slices"] = self.load_slices(gen_image_slices[0])
        wrapped_tokenizer_output = SimpleNamespace(**tokenizer_output)

        model_inputs = dict(
            position_ids=batch_input_pos,
            past_key_values=None,
            custom_pos_emb=custom_pos_emb[0],
            mode=mode,
            image_mask=image_mask[0],
            gen_timestep_scatter_index=gen_timestep_scatter_index[0],
            # for inner usage
            tokenizer_output=wrapped_tokenizer_output,
            batch_gen_image_info=batch_gen_image_info,
            generator=np.random.Generator(np.random.PCG64(self.args.seed)),
            # generation config
            eos_token_id=[127957, 128000],  # stop_token_id[bot_task]
        )

        # Prepare parameters
        guidance_scale = self.args.guidance_scale
        guidance_rescale = self.args.guidance_rescale

        do_classifier_free_guidance = guidance_scale > 1.0
        cfg_factor = 1 + do_classifier_free_guidance

        num_inference_steps = self.args.num_inference_steps

        # Prapre timesteps
        timesteps = self.retrieve_timesteps(self.scheduler, num_inference_steps=num_inference_steps)

        # Prepare latent variables
        batch_gen_image_info = model_inputs["batch_gen_image_info"]
        latents = self.prepare_latents(
            batch_size=len(batch_gen_image_info),
            latent_channel=self.args.config.vae["latent_channels"],
            image_size=[batch_gen_image_info[0].image_height, batch_gen_image_info[0].image_width],
            dtype=self.args.ms_dtype,
            generator=model_inputs["generator"],
        )

        # Prepare extra step kwargs.
        _scheduler_step_extra_kwargs = {"generator": model_inputs["generator"]}

        # Prepare model kwargs
        model_kwargs = model_inputs
        attention_mask = self.prepare_mask_for_gen(
            input_ids,
            self.args.generation_config,
            model_kwargs=model_kwargs,
        )
        model_kwargs["attention_mask"] = attention_mask

        # Sampling loop
        self._num_timesteps = len(timesteps)

        # Denoising loop
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = ops.cat([latents] * cfg_factor)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            t_expand = t.repeat(latent_model_input.shape[0])

            model_inputs = self.prepare_inputs_for_gen(
                input_ids,
                images=latent_model_input,
                timestep=t_expand,
                **model_kwargs,
            )

            model_output = self.generation(**model_inputs, first_step=(i == 0))
            pred = model_output["diffusion_prediction"]
            pred = pred.to(dtype=ms.float32)  # (2, 3, 512, 512)

            # perform guidance
            if do_classifier_free_guidance:
                pred_cond, pred_uncond = pred.chunk(2)
                pred = self.cfg_operator(pred_cond, pred_uncond, guidance_scale, step=i)

                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    pred = self.rescale_noise_cfg(pred, pred_cond, guidance_rescale=guidance_scale)  # (1, 3, 512, 512)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(pred, t, latents, **_scheduler_step_extra_kwargs, return_dict=False)[0]

            if i != len(timesteps) - 1:
                model_kwargs = self.update_kwargs_for_gen(  # noqa
                    model_output,
                    model_kwargs,
                )
                if input_ids.shape[1] != model_kwargs["position_ids"].shape[1]:
                    input_ids = mint.gather(input_ids, 1, index=model_kwargs["position_ids"])

        if hasattr(self.args.vae_config, "scaling_factor") and self.args.vae_config.scaling_factor:
            latents = latents / self.args.vae_config.scaling_factor
        if hasattr(self.args.vae_config, "shift_factor") and self.args.vae_config.shift_factor:
            latents = latents + self.args.vae_config.shift_factor  # (1, 3, 512, 512)

        with pynative_no_grad():
            image = self.vae_decode(latents, return_dict=False)[0][:, :, 0]  # [B, C, H, W]
            do_denormalize = [True] * batch_size
            image = self.postprocess(image, output_type="ms", do_denormalize=do_denormalize)  # (1, 3, 512, 512)

        preds = image[0]  # [C, H, W] = (3, 512, 512)
        loss = ms.mint.mean(
            ((preds - labels[0]) ** 2).reshape(3, -1),
            dim=1,
        ).mean()

        return (loss,)


if __name__ == "__main__":
    main()
