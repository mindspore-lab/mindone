from time import time

import numpy as np
from src import LLaDAModelLM
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

import mindspore as ms
import mindspore.ops as ops
from mindspore import Tensor, mint

ms.set_context(mode=ms.PYNATIVE_MODE)


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    MindSpore uses float32 for better inference speed
    """
    if temperature == 0:
        return logits
    logits = logits.to(ms.float32)
    noise = ops.rand_like(logits, dtype=ms.float32)
    gumbel_noise = (-ops.log(noise)) ** temperature
    return ops.exp(logits) / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(axis=1, keepdims=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = ops.zeros((mask_num.shape[0], steps), dtype=ms.int64) + base

    for i in range(mask_num.shape[0]):
        num_transfer_tokens[i, : remainder[0, i]] += 1

    return num_transfer_tokens


def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    # x = ops.full((1, prompt.shape[1] + gen_length), mask_id, dtype=ms.int64)
    # x[:, : prompt.shape[1]] = prompt.copy()
    # avoid inplace operation
    x = ops.cat([prompt.copy(), ops.full((1, gen_length), mask_id, dtype=prompt.dtype)], axis=1)

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (
            x[:, prompt.shape[1] + num_block * block_length : prompt.shape[1] + (num_block + 1) * block_length :]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        with tqdm(total=steps, desc=f"Generating Block {num_block}") as pbar:
            for i in range(steps):
                step_start = time()
                mask_index = x == mask_id
                if cfg_scale > 0.0:
                    un_x = x.copy()
                    un_x[prompt_index] = mask_id
                    x_ = ops.cat([x, un_x], axis=0)
                    logits = model(x_, return_dict=False)[0]
                    logits, un_logits = ops.chunk(logits, 2, axis=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, return_dict=False)[0]

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = ops.argmax(logits_with_noise, dim=-1)  # b, l

                if remasking == "low_confidence":
                    p = ops.softmax(logits.to(ms.float32), axis=-1)
                    x0_p = ops.squeeze(mint.gather(p, dim=-1, index=ops.unsqueeze(x0, -1)), -1)  # b, l
                elif remasking == "random":
                    x0_p = ops.rand((x0.shape[0], x0.shape[1]))
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -np.inf

                x0 = ops.where(mask_index, x0, x)
                confidence = ops.where(mask_index, x0_p, -np.inf)

                transfer_index = ops.zeros_like(x0, dtype=ms.bool_)
                for j in range(confidence.shape[0]):
                    _, select_index = ops.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]
                pbar.set_postfix(iteration_time=f"{time() - step_start:.3f}")
                pbar.update()

    return x


def run_profiler(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
):
    x = ops.cat([prompt.copy(), ops.full((1, gen_length), mask_id, dtype=prompt.dtype)], axis=1)

    prompt_index = x != mask_id

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    output_path = "./profiler_data"
    profiler = ms.Profiler(
        start_profile=False, output_path=output_path, use_stack=True, profiler_level=ms.profiler.ProfilerLevel.Level1
    )
    for num_block in range(num_blocks):
        for i in range(steps):
            if i == 1:
                profiler.start()
            if cfg_scale > 0.0:
                un_x = x.copy()
                un_x[prompt_index] = mask_id
                x_ = ops.cat([x, un_x], axis=0)
                logits = model(x_, return_dict=False)[0]
                logits, un_logits = ops.chunk(logits, 2, axis=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                logits = model(x, return_dict=False)[0]
            if i == 1:
                break
        break
    profiler.stop()
    profiler.analyse()

    print(f"Profiler analysis finished. See results in {output_path}")


def main():
    tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    config = AutoConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    # config.flash_attention = True
    model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", mindspore_dtype=ms.bfloat16, config=config)

    prompt = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    input_ids = (
        Tensor(input_ids) if (len(input_ids.shape) == 2 and input_ids.shape[0] == 1) else Tensor(input_ids).unsqueeze(0)
    )  # (1, L)
    infer_start = time()
    out = generate(
        model,
        input_ids,
        steps=128,
        gen_length=128,
        block_length=32,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
    )
    print(f"Inference time: {time() - infer_start:.3f}s")
    print(tokenizer.batch_decode(out[:, input_ids.shape[1] :], skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
