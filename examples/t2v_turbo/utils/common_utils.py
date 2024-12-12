import ast
import gc

import numpy as np
from lvdm.modules.attention import BasicTransformerBlock

import mindspore as ms
from mindspore import mint, nn, ops

from mindone.diffusers.models.attention_processor import AttnProcessor


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(t, -1)
    new_shape = (b,) + ((1,) * (len(x_shape) - 1))
    return out.reshape(new_shape)


def is_attn(name):
    return "attn1" or "attn2" == name.split(".")[-1]


def set_processors(attentions):
    for attn in attentions:
        attn.set_processor(AttnProcessor())


def set_torch_2_attn(unet):
    optim_count = 0

    for name, module in unet.cells_and_names():
        if is_attn(name):
            if isinstance(module, nn.CellList):
                for m in module:
                    if isinstance(m, BasicTransformerBlock):
                        set_processors([m.attn1, m.attn2])
                        optim_count += 1
    if optim_count > 0:
        print(f"{optim_count} Attention layers using Scaled Dot Product Attention.")


# From LatentConsistencyModel.get_guidance_scale_embedding
def guidance_scale_embedding(w, embedding_dim=512, dtype=ms.float32):
    """
    See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

    Args:
        timesteps (`ms.Tensor`):
            generate embedding vectors at these timesteps
        embedding_dim (`int`, *optional*, defaults to 512):
            dimension of the embeddings to generate
        dtype:
            data type of the generated embeddings

    Returns:
        `ms.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
    """
    assert len(w.shape) == 1
    w = w * 1000.0

    half_dim = embedding_dim // 2
    emb = mint.log(ms.Tensor(10000.0)) / (half_dim - 1)
    emb = mint.exp(mint.arange(half_dim, dtype=dtype) * -emb)
    emb = w.to(dtype)[:, None] * emb[None, :]
    emb = mint.cat([mint.sin(emb), mint.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1))
    assert emb.shape == (w.shape[0], embedding_dim)
    return emb


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    for i in range(dims_to_append):
        x = ops.expand_dims(x, -1)
    return x


# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    scaled_timestep = timestep_scaling * timestep
    c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
    c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5
    return c_skip, c_out


# Compare LCMScheduler.step, Step 4
def get_predicted_original_sample(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_x_0 = (sample - sigmas * model_output) / alphas
    elif prediction_type == "sample":
        pred_x_0 = model_output
    elif prediction_type == "v_prediction":
        pred_x_0 = alphas * sample - sigmas * model_output
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_x_0


# Based on step 4 in DDIMScheduler.step
def get_predicted_noise(model_output, timesteps, sample, prediction_type, alphas, sigmas):
    alphas = extract_into_tensor(alphas, timesteps, sample.shape)
    sigmas = extract_into_tensor(sigmas, timesteps, sample.shape)
    if prediction_type == "epsilon":
        pred_epsilon = model_output
    elif prediction_type == "sample":
        pred_epsilon = (sample - alphas * model_output) / sigmas
    elif prediction_type == "v_prediction":
        pred_epsilon = alphas * model_output + sigmas * sample
    else:
        raise ValueError(
            f"Prediction type {prediction_type} is not supported; currently, `epsilon`, `sample`, and `v_prediction`"
            f" are supported."
        )

    return pred_epsilon


def param_optim(model, condition, extra_params=None, is_lora=False, negation=None):
    extra_params = extra_params if len(extra_params.keys()) > 0 else None
    return {
        "model": model,
        "condition": condition,
        "extra_params": extra_params,
        "is_lora": is_lora,
        "negation": negation,
    }


def create_optim_params(name="param", params=None, lr=5e-6, extra_params=None):
    params = {"params": params, "lr": lr}
    if extra_params is not None:
        for k, v in extra_params.items():
            params[k] = v

    return params


def create_optimizer_params(model_list, lr):
    optimizer_params = []

    for optim in model_list:
        model, condition, extra_params, is_lora, negation = optim.values()
        # Check if we are doing LoRA training.
        if is_lora and condition and isinstance(model, list):
            params = create_optim_params(params=model, extra_params=extra_params)
            optimizer_params.append(params)
            continue

        if is_lora and condition and not isinstance(model, list):
            for n, p in model.parameters_and_names():
                if "lora" in n:
                    params = create_optim_params(n, p, lr, extra_params)
                    optimizer_params.append(params)
            continue

        # If this is true, we can train it.
        if condition:
            for n, p in model.parameters_and_names():
                should_negate = "lora" in n and not is_lora
                if should_negate:
                    continue

                params = create_optim_params(n, p, lr, extra_params)
                optimizer_params.append(params)

    return optimizer_params


def handle_trainable_modules(model, trainable_modules=None, is_enabled=True, negation=None):
    acc = []
    unfrozen_params = 0

    if trainable_modules is not None:
        unlock_all = any([name == "all" for name in trainable_modules])
        if unlock_all:
            model.requires_grad_(True)
            unfrozen_params = len(list(model.parameters()))
        else:
            model.requires_grad_(False)
            for name, param in model.named_parameters():
                for tm in trainable_modules:
                    if all([tm in name, name not in acc, "lora" not in name]):
                        param.requires_grad_(is_enabled)
                        acc.append(name)
                        unfrozen_params += 1


def huber_loss(pred, target, huber_c=0.001):
    loss = ops.sqrt((pred.float() - target.float()) ** 2 + huber_c**2) - huber_c
    return loss.mean()


def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def log_validation_video(pipeline, args, trackers, save_fps):
    if args.seed is None:
        generator = None
    else:
        generator = np.random.Generator(np.random.PCG64(args.seed))

    validation_prompts = [
        "An astronaut riding a horse.",
        "Darth vader surfing in waves.",
        "Robot dancing in times square.",
        "Clown fish swimming through the coral reef.",
        "A child excitedly swings on a rusty swing set, laughter filling the air.",
        "With the style of van gogh, A young couple dances under the moonlight by the lake.",
        "A young woman with glasses is jogging in the park wearing a pink headband.",
        "Impressionist style, a yellow rubber duck floating on the wave on the sunset",
    ]

    video_logs = []

    for _, prompt in enumerate(validation_prompts):
        videos = pipeline(
            prompt=prompt,
            frames=args.n_frames,
            num_inference_steps=4,
            num_videos_per_prompt=2,
            generator=generator,
        )
        videos = (videos.clamp(-1.0, 1.0) + 1.0) / 2.0
        videos = (videos * 255).to(ms.uint8).permute(0, 2, 1, 3, 4).cpu().numpy()
        video_logs.append({"validation_prompt": prompt, "videos": videos})

    del pipeline
    gc.collect()


def tuple_type(s):
    if isinstance(s, tuple):
        return s
    value = ast.literal_eval(s)
    if isinstance(value, tuple):
        return value
    raise TypeError("Argument must be a tuple")


def load_model_checkpoint(model, ckpt):
    state_dict = ms.load_checkpoint(ckpt)
    param_not_load, _ = ms.load_param_into_net(model, state_dict, strict_load=False)
    if param_not_load:
        print("param_not_load: ", param_not_load)
    del state_dict
    gc.collect()

    print(">>> model checkpoint loaded.")
    return model
