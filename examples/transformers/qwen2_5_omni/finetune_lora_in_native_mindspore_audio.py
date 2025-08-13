"""
Qwen2.5-Omni model fine-tuning script using LoRA.

This script with default values fine-tunes a pretrained Talker model from  Qwen2.5-Omni-3B/Qwen2.5-Omni-7B,
on the `linxy/LaTex_OCR` dataset ,

reference lora config: https://github.com/modelscope/ms-swift/pull/3613
reference data processing: https://github.com/QwenLM/Qwen2.5-VL/blob/main/qwen-vl-finetune/qwenvl/data/data_qwen.py

Usage:
```
DEVICE_ID=0 python finetune_lora_in_native_mindspore_audio.py \
    --model_path Qwen/Qwen2.5-Omni-3B \
    --enable_flash_attention \
    --lora_rank 8 \
    --lora_alpha 16 \
    --dataset_path mozilla-foundation/common_voice_11_0 \
    --output_dir ./outputs/lora \
    --num_train_epochs 1

```
"""

import argparse
import ast
import os
import time

import numpy as np
from datasets import load_dataset
from qwen_omni_utils import process_mm_info

import mindspore as ms
from mindspore import nn

from mindone.diffusers._peft import LoraConfig, get_peft_model
from mindone.diffusers.training_utils import cast_training_params
from mindone.transformers import Qwen2_5OmniForConditionalGeneration
from mindone.transformers.mindspore_adapter import HF2MSDataset, TrainOneStepWrapper
from mindone.transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

IGNORE_INDEX = -100


def freeze_params(m: nn.Cell):
    for p in m.get_parameters():
        p.requires_grad = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-Omni-3B", help="pretrained model name")
    parser.add_argument("--dataset_path", type=str, default="mozilla-foundation/common_voice_11_0", help="dataset path.")
    parser.add_argument("--output_dir", type=str, default="./outputs/lora", help="output directory for checkpoints")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="batch size per device for training")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="number of training epochs")
    parser.add_argument("--save_steps", type=int, default=500, help="save checkpoint every X steps")
    parser.add_argument(
        "--do_eval", action="store_true", default=False, help="whether to run evaluation after training"
    )
    parser.add_argument("--max_length", type=int, default=4096, help="fixed token length for training")
    parser.add_argument("--enable_flash_attention", action="store_true", default=False, help="enable flash attention")
    parser.add_argument(
        "--zero_stage", type=int, default=0, choices=[0, 1, 2], help="stage of ZeRO optimizer parallelism"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=False, help="whether or not to enable mix precision with float16"
    )
    parser.add_argument(
        "--bf16", action="store_true", default=False, help="whether or not to enable mix precision with bfloat16"
    )
    parser.add_argument(
        "--is_distribute", type=ast.literal_eval, default=False, help="whether or not to run distribute"
    )
    parser.add_argument("--resume", type=str, default=None, help="lora path to resume training from a checkpoint")
    parser.add_argument("--lora_rank", type=int, default=8, help="The dimension of the LoRA update matrices")
    parser.add_argument("--lora_alpha", type=int, default=16, help="The scaling factor alpha of the LoRA")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="learning rate for training")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
        "capable of perceiving auditory and visual inputs, as well as generating text and speech.",
    )
    parser.add_argument("--prompt", type=str, default="Please convert the audio to text")
    args = parser.parse_args()

    # 0. set mindspore context
    ms.set_context(mode=ms.PYNATIVE_MODE, jit_config={"jit_level": "O0"})  # not support graph mode yet
    rank = 0
    rank_size = 1
    if args.is_distribute:
        from mindspore.communication import get_group_size, get_rank, init

        init()
        rank = get_rank()
        rank_size = get_group_size()
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(
            parallel_mode=ms.ParallelMode.DATA_PARALLEL,
            gradients_mean=True,
            device_num=get_group_size(),
        )

    # 1. create dataset
    dataset = load_dataset(args.dataset_path, name="yue", trust_remote_code=True)
    dataset["train"] = dataset["train"].shuffle(seed=42).select(range(1000))
    dataset["test"] = dataset["test"].select(range(1000))

    system_prompt = args.system_prompt
    prompt = args.prompt
    processor = Qwen2_5OmniProcessor.from_pretrained(args.model_path)

    def process_function(examples):
        answer = examples["sentence"]
        # audio = examples["audio"]
        audio_path = examples["path"]
        conversations = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]
        input_padding = "max_length" if args.per_device_train_batch_size > 1 else True
        # for batched training, use homogeneous token length, padding side is left
        inputs = processor.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="np",
            # kwargs to be passed to `Qwen2_5OmniProcessor`
            padding=input_padding,
            max_length=args.max_length,
            use_audio_in_video=False,
        )  # input_ids, attention_mask, pixel_values, image_grid_thw

        # Prepare the labels, keep response part as labels
        if input_padding == "max_length":
            input_full_ids = processor.apply_chat_template(
                conversations,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=False,
                return_tensors="np",
                padding=True,
            )[0] # only ids
        else:
            input_full_ids = inputs["input_ids"][0]
        prompt_ids = processor.apply_chat_template(
            conversations[:-1],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
            return_tensors="np",
            padding=True,
        )[0] # only ids
        labels = np.ones_like(inputs["input_ids"]) * IGNORE_INDEX
        response_start_id = inputs["input_ids"].shape[1] - len(input_full_ids) + len(prompt_ids)
        labels[..., response_start_id:] = inputs["input_ids"][..., response_start_id:]

        for k, v in inputs.items():
            if v.shape[0] == 1:
                examples[k] = v[0]  # remove batch dimension
            else:
                examples[k] = v  # pixel_values
        examples["labels"] = labels[0]
        if not args.do_eval:
            examples.pop("text")  # remove text from examples
            examples.pop("image")  # remove image from examples

        return examples

    tokenized_datasets = dataset.map(process_function, batched=False)
    small_train_dataset = tokenized_datasets["train"]
    small_eval_dataset = tokenized_datasets["test"]

    def ms_data_collator(features, batch_info):
        batch = {}
        for k, v in features[0].items():
            batch[k] = (
                np.stack([f[k] for f in features]) if isinstance(v, np.ndarray) else np.array([f[k] for f in features])
            )
        return batch

    batch_size = args.per_device_train_batch_size * rank_size
    num_epochs = args.num_train_epochs
    train_dataloader = ms.dataset.GeneratorDataset(
        HF2MSDataset(small_train_dataset),
        column_names="item",
        num_shards=rank_size,
        shard_id=rank,
        python_multiprocessing=False,
        shuffle=True,
    )
    train_dataloader = train_dataloader.batch(batch_size=batch_size, per_batch_map=ms_data_collator)
    train_dataloader = train_dataloader.repeat(1)
    train_dataloader = train_dataloader.create_dict_iterator(num_epochs=num_epochs, output_numpy=True)

    # 2. create train network
    # 2.1. load pretrained model
    parent_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        attn_implementation="flash_attention_2" if args.enable_flash_attention else "eager",
        mindspore_dtype=ms.bfloat16 if args.bf16 else (ms.float16 if args.fp16 else None),
    )  # TODO: only load thinker state dicts
    model = parent_model.thinker
    model.config.use_cache = False
    freeze_params(model)

    # 2.2. prepare the LoRA config
    # all attn linear layers in audio_tower and text model
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["k_proj", "v_proj", "q_proj", "out_proj", "o_proj"],
    )
    model.is_gradient_checkpointing = False
    model = get_peft_model(model, lora_config)
    if args.fp16 or args.bf16:
        cast_training_params(model, dtype=ms.float32)
    model.print_trainable_parameters()
    # print(model.target_module_names)

    # 2.3. optimizer
    if args.zero_stage == 0:
        optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=args.learning_rate)
    elif args.zero_stage == 1:
        from mindone.transformers.mindspore_adapter import AdamWeightDecayZeRO1

        optimizer = AdamWeightDecayZeRO1(model.trainable_params(), learning_rate=args.learning_rate)
    elif args.zero_stage == 2:
        from mindone.transformers.mindspore_adapter import AdamWeightDecayZeRO2

        optimizer = AdamWeightDecayZeRO2(model.trainable_params(), learning_rate=args.learning_rate)
    else:
        raise ValueError

    # 2.4. trainer
    class ReturnLoss(nn.Cell):
        def __init__(self, model):
            super(ReturnLoss, self).__init__(auto_prefix=False)
            self.model = model

        def construct(self, *args, **kwargs):
            outputs = self.model(*args, **kwargs)
            loss = outputs[0]
            return loss

    train_model = TrainOneStepWrapper(ReturnLoss(model), optimizer)

    # 3. training
    train_model.set_train()
    total_time = 0.0
    for step, batch in enumerate(train_dataloader):
        start_time = time.time()

        batch = batch["item"]

        # convert dict to tuple
        tuple_inputs = (
            ms.Tensor(batch["input_ids"], ms.int32),
            ms.Tensor(batch["input_features"],  parent_model.dtype),
            None,
            None,
            None,
            None,
            ms.Tensor(batch["attention_mask"], ms.int32),
            ms.Tensor(batch["feature_attention_mask"], ms.int32),
            ms.Tensor(batch["audio_feature_lengths"], ms.int32),
            None,
            None,
            None,
            None,
            ms.Tensor(batch["labels"], ms.int32),
        )

        loss, _, overflow = train_model(*tuple_inputs)

        if step > 1:
            step_time = time.time() - start_time
            total_time += step_time
            print(
                f"step: {step}, loss: {loss}, step time: {step_time:.5f}s, avg speed: {total_time/(step+1):.5f}s/step"
            )
        else:
            print(f"step: {step}, loss: {loss}")

        if (step + 1) % args.save_steps == 0:
            output_dir = os.path.join(args.output_dir, f"checkpoint-{step}")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model.save_pretrained(output_dir)
            print(f"LoRA model saved to {output_dir}")

    # save final model
    output_dir = os.path.join(args.output_dir, "checkpoint-final")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save_pretrained(output_dir)
    print(f"Final LoRA model saved to {output_dir}")

    # 4. inference and evaluation
    if args.do_eval:  # FIXME: bf16 not supported yet
        model.merge_and_unload()  # merge LoRA weights into the base model
        parent_model.thinker = model.get_base_model()  # replace thinker with LoRA-enhanced model
        parent_model.set_train(False)

        # inference function
        def inference(medium_path, prompt, medium_type="audio", use_audio_in_video=True):
            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            medium = None
            if medium_type == "audio":
                medium = {
                    "type": medium_type,
                    "audio": medium_path,
                }
            if medium is not None:
                messages[1]["content"].append(medium)

            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            audios, images, videos = process_mm_info(messages, use_audio_in_video=use_audio_in_video)
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="np",
                padding=True,
                use_audio_in_video=use_audio_in_video,
            )

            # convert input to Tensor
            for key, value in inputs.items():  # by default input numpy array or list
                inputs[key] = ms.Tensor(value)
                if inputs[key].dtype == ms.int64:
                    inputs[key] = inputs[key].to(ms.int32)
                else:
                    inputs[key] = inputs[key].to(parent_model.dtype)

            text_ids = parent_model.generate(**inputs, use_audio_in_video=use_audio_in_video, return_audio=False)
            text_ids = [output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, text_ids)]
            text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return text

        correct = 0
        file_path = os.path.join(args.output_dir, "inference_lora_eval.txt")
        for idx, example in enumerate(small_eval_dataset):
            medium = example["path"]
            answer = example["sentence"]
            response = inference(medium, prompt, medium_type="audio", use_audio_in_video=False)
            print(f"Response #{idx}: {response}\n")

            with open(file_path, "a") as f:
                f.write(f"Response #{idx}: {response}\n")
                if response != answer:
                    f.write(f"WRONG! GT #{idx}: {answer}\n")
                else:
                    correct += 1
        with open(file_path, "a") as f:
            f.write(f"Correctness: {correct}/{len(small_eval_dataset)} = {correct/len(small_eval_dataset):.2%}\n")
            print(
                f"Test Set Correctness: {correct}/{len(small_eval_dataset)} = {correct/len(small_eval_dataset):.2%}\n"
            )


if __name__ == "__main__":
    main()
