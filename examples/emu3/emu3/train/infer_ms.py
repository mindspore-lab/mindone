"""
This script is to load mindspore checkpoint and run image generation or vqa or qa after SFT in parallel mode

Usage:
cd examples/emu3

DEVICE_ID=0 python emu3/train/infer_ms.py \
--model_path outputs/Emu3-T2I-SFT \
--tokenizer_path BAAI/Emu3-Stage1 \
--vision_tokenizer_path BAAI/Emu3-VisionTokenizer \
--ckpt_dir outputs/Emu3-T2I-SFT/rank_all_8/ckpt/emu3-e5.ckpt \
--prompt "A girl."
--task img-gen \
--output_path output

DEVICE_ID=0 python emu3/train/infer_ms.py \
--model_path outputs/Emu3-VQA-SFT \
--tokenizer_path BAAI/Emu3-Stage1 \
--vision_tokenizer_path BAAI/Emu3-VisionTokenizer \
--ckpt_dir outputs/Emu3-VQA-SFT/rank_all_8/ckpt/emu3-e5.ckpt \
--image img_prompt.jpg \
--prompt "Describe this image."
--task vqa \
--output_path output
"""
import argparse
import json
import logging
import os
import time

from emu3.mllm import Emu3Config, Emu3ForCausalLM, Emu3Tokenizer
from emu3.mllm.processing_emu3 import Emu3Processor
from emu3.tokenizer import Emu3VisionVQImageProcessor, Emu3VisionVQModel
from PIL import Image
from transformers.generation.configuration_utils import GenerationConfig

import mindspore as ms
from mindspore import Tensor, nn, ops

from mindone.utils import set_logger
from mindone.utils.amp import auto_mixed_precision
from mindone.utils.config import str2bool
from mindone.utils.seed import set_random_seed

logger = logging.getLogger(__name__)


def load_net(model, model_file):
    print(f"Loading weights from local pretrained directory: {model_file}")
    state_dict = ms.load_checkpoint(model_file)
    state_dict_tmp = {}
    for k, v in state_dict.items():
        if k.startswith("network."):
            k = k.replace("network.", "")
        k = k.replace("_backbone.", "")
        state_dict_tmp[k] = v
    state_dict = state_dict_tmp

    # Instantiate the model
    param_not_load, ckpt_not_load = ms.load_param_into_net(model, state_dict)
    print(f"Loaded checkpoint: param_not_load {param_not_load}, ckpt_not_load {ckpt_not_load}")
    if param_not_load or ckpt_not_load:
        print(
            f"Exist ckpt params not loaded: {ckpt_not_load} (total: {len(ckpt_not_load)}),\n"
            f"or net params not loaded: {param_not_load} (total: {len(param_not_load)})"
        )
    ckpt_name = os.path.basename(model_file)
    if ckpt_name != "train_resume.ckpt":
        epoch_num = int(ckpt_name.strip()[6:-5])
    else:
        epoch_num = state_dict["epoch_num"].item()

    return epoch_num


def evaluate(args):
    save_dir = args.output_path
    rank_id = 0
    ms.set_context(
        mode=args.mode,  # only support PYNATIVE using DynamicCache
        device_target=args.device_target,
        pynative_synchronize=args.debug,
        jit_config={"jit_level": args.jit_level},
        device_id=int(os.getenv("DEVICE_ID")),
    )
    set_random_seed(args.seed)
    logger = set_logger(name="", output_dir=args.output_path, rank=rank_id, log_level=eval(args.log_level))

    # 1. Load Models and Processor
    # model path
    TOKENIZER_HUB = args.tokenizer_path
    VQ_HUB = args.vision_tokenizer_path
    EMU_DTYPE = ms.float16 if args.dtype == "fp16" else ms.bfloat16
    VQ_DTYPE = ms.bfloat16
    start_time = time.time()

    # prepare model and processor
    cfg_path = os.path.join(args.model_path, "config.json")
    with open(cfg_path, "r") as f:
        # returns JSON object as a dictionary
        config_dict = json.load(f)
    model_config = Emu3Config(**config_dict)
    model_config.attn_implementation = "flash_attention_2"  # optional: "eager"
    model = Emu3ForCausalLM(model_config).to(EMU_DTYPE)
    model.set_train(False)

    # load pretrained checkpoint
    logger.info(f"Loading ckpt in {args.model_path}.")

    epoch_num = load_net(model, args.ckpt_dir)
    logger.info(f"Loaded checkpoint at Epoch #{epoch_num}")

    image_path = os.path.join(save_dir, f"e{epoch_num}")
    os.makedirs(image_path, exist_ok=True)

    print("Start to load tokenizer...")
    tokenizer = Emu3Tokenizer.from_pretrained(TOKENIZER_HUB, padding_side="left")
    image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(
        VQ_HUB, use_safetensors=True, mindspore_dtype=VQ_DTYPE
    ).set_train(False)
    image_tokenizer = auto_mixed_precision(
        image_tokenizer, amp_level="O2", dtype=VQ_DTYPE, custom_fp32_cells=[nn.BatchNorm3d]
    )
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)
    print("Loaded all models, time elapsed: %.4fs" % (time.time() - start_time))

    # 2. Prepare Input
    start_time = time.time()

    text = [args.prompt]
    image = None
    if args.image is not None:
        image = Image.open(args.image)
        image = [image]

    if args.task == "img-gen":
        POSITIVE_PROMPT = "masterpiece, film grained, best quality."
        NEGATIVE_PROMPT = "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, \
            fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry."

        classifier_free_guidance = 3.0
        prompt = [args.prompt]
        prompt = [p + POSITIVE_PROMPT for p in prompt]

        kwargs = dict(
            mode="G",
            ratio=["1:1"],
            image_area=model.config.image_area,  # 512*512
            return_tensors="np",
            padding="longest",
        )
        inputs = processor(text=prompt, **kwargs)
        neg_inputs = processor(text=[NEGATIVE_PROMPT] * len(prompt), **kwargs)
        # prepare hyper parameters
        GENERATION_CONFIG = GenerationConfig(
            use_cache=True,
            bos_token_id=model.config.bos_token_id,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            max_new_tokens=40960,
            do_sample=True,
            top_k=2048,
        )

        h = Tensor(inputs.image_size[:, 0])
        w = Tensor(inputs.image_size[:, 1])
        constrained_fn = processor.build_prefix_constrained_fn(h, w)

        from mindone.transformers.generation.logits_process import (
            LogitsProcessorList,
            PrefixConstrainedLogitsProcessor,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
        )

        logits_processor = LogitsProcessorList(
            [
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    classifier_free_guidance,
                    model,
                    unconditional_ids=Tensor(neg_inputs.input_ids, dtype=ms.int32),
                ),
                PrefixConstrainedLogitsProcessor(
                    constrained_fn,
                    num_beams=1,
                ),
            ]
        )
    else:
        inputs = processor(
            text=text,
            image=image,
            mode="U",
            padding_image=True,
            padding="longest",
            return_tensors="np",
        )
        # prepare hyper parameters
        GENERATION_CONFIG = GenerationConfig(
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=1024,
        )
        logits_processor = None

    print("Prepared inputs, time elapsed: %.4fs" % (time.time() - start_time))

    # 3. Generate Next Tokens, Decode Tokens

    # generate
    start_time = time.time()
    outputs = model.generate(
        Tensor(inputs.input_ids, dtype=ms.int32),
        GENERATION_CONFIG,
        logits_processor=logits_processor,
        attention_mask=Tensor(inputs.attention_mask),
    )

    print(f"generated_ids length / #steps: {len(outputs[0])}")
    elapsed = time.time() - start_time
    print("Average speed %.4fs/step" % (elapsed / len(outputs[0])))
    print("Finish generation, time elapsed: %.4fs" % (time.time() - start_time))

    if args.task == "img-gen":
        # since input_ids are deleted in generate() output
        # need to add input_ids back ahead, which contains visual boi/eoi tokens and meta data for image detokenization
        outputs = ops.cat((Tensor(inputs.input_ids, dtype=outputs.dtype), outputs), axis=1)
        start_time = time.time()
        for idx_i, out in enumerate(outputs):
            mm_list = processor.decode(out)
            for idx_j, im in enumerate(mm_list):
                if not isinstance(im, Image.Image):
                    continue
                im.save(os.path.join(image_path, f"result_{idx_i}_{idx_j}.png"))
                print(f"Saved result_{idx_i}_{idx_j}.png in {image_path}")
    else:
        # detokenization
        start_time = time.time()
        answers = processor.batch_decode(outputs, skip_special_tokens=True)
        for ans in answers:
            print(ans)

    print("\nFinished detokenization, time elapsed: %.4fs" % (time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model path with config.json")
    parser.add_argument(
        "--ckpt_dir", type=str, default=None, help="model ckpt diretory, e.g. outputs/rank_group_8/ckpt/emu3-e50.ckpt"
    )
    parser.add_argument("--tokenizer_path", type=str, default=None, help="tokenizer folder, e.g. BAAI/Emu3-Stage1")
    parser.add_argument(
        "--vision_tokenizer_path",
        type=str,
        default=None,
        help="vision tokenizer folder, e.g. BAAI/Emu3-VisionTokenizer",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output",
        help="output dir to save the log and generated image (if applicable)",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="vqa",
        help="inference task, either vqa, img-gen",
    )
    parser.add_argument("--prompt", default=None, help="Text prompt")
    parser.add_argument("--image", default=None, help="Image path")
    parser.add_argument("--data_json", default=None)
    parser.add_argument("--device_target", type=str, default="Ascend", help="Ascend or GPU")
    parser.add_argument("--mode", type=int, default=1, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=1)")
    parser.add_argument("--debug", default=False, type=str2bool, help="enable pynative_synchronize")
    parser.add_argument("--seed", type=int, default=42, help="Inference seed")
    # parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
    parser.add_argument(
        "--jit_level",
        default="O0",
        type=str,
        choices=["O0", "O1", "O2"],
        help="Used to control the compilation optimization level. Supports [“O0”, “O1”, “O2”]."
        "O0: Except for optimizations that may affect functionality, all other optimizations are turned off, adopt KernelByKernel execution mode."
        "O1: Using commonly used optimizations and automatic operator fusion optimizations, adopt KernelByKernel execution mode."
        "O2: Ultimate performance optimization, adopt Sink execution mode.",
    )
    parser.add_argument("--batch_size", default=1, type=int, help="infer batch size")
    parser.add_argument(
        "--dtype",
        default="fp16",  # if amp level O0/1, must pass fp32
        type=str,
        choices=["bf16", "fp16", "fp32"],
        help="what computation data type to use in Emu3 model. Default is `fp16`, which corresponds to ms.float16",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="logging.INFO",
        help="log level, options: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
