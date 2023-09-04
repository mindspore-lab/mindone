import os
from datetime import datetime
from typing import List, Union

import numpy as np
import yaml
from gm.modules.diffusionmodules.sampler import EulerEDMSampler
from gm.util import auto_mixed_precision, instantiate_from_config, seed_everything
from omegaconf import ListConfig
from PIL import Image

import mindspore as ms
from mindspore import Tensor, context, nn, ops
from mindspore.communication.management import get_group_size, get_rank, init

SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}

VERSION2SPECS = {
    "SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": False,
    },
}


def set_default(args):
    seed_everything(args.seed)

    # Set Mindspore Context
    context.set_context(mode=args.ms_mode, device_target=args.device_target)
    if args.device_target == "Ascend":
        device_id = int(os.getenv("DEVICE_ID", 0))
        context.set_context(device_id=device_id)
    elif args.device_target == "GPU" and args.ms_enable_graph_kernel:
        context.set_context(enable_graph_kernel=True)
    # Set Parallel
    if args.is_parallel:
        init()
        args.rank, args.rank_size, parallel_mode = get_rank(), get_group_size(), context.ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(device_num=args.rank_size, parallel_mode=parallel_mode, gradients_mean=True)
    else:
        args.rank, args.rank_size = 0, 1

    # Directories and Save run settings
    time = _get_broadcast_datetime(rank_size=args.rank_size)
    args.save_path = os.path.join(
        args.save_path, f"{time[0]:04d}.{time[1]:02d}.{time[2]:02d}-{time[3]:02d}.{time[4]:02d}.{time[5]:02d}"
    )
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "weights"), exist_ok=True)
    if args.rank % args.rank_size == 0:
        with open(os.path.join(args.save_path, "cfg.yaml"), "w") as f:
            yaml.dump(vars(args), f, sort_keys=False)

    # Modelarts: Copy data/ckpt, from the s3 bucket to the computing node; Reset dataset dir.
    if args.enable_modelarts:
        from gm.util.modelarts import sync_data

        os.makedirs(args.data_dir, exist_ok=True)
        sync_data(args.data_url, args.data_dir)
        sync_data(args.save_path, args.train_url)
        if args.ckpt_url:
            sync_data(args.ckpt_url, args.ckpt_dir)  # pretrain ckpt

        args.data_path = os.path.join(args.data_dir, args.data_path)
        args.weight = args.ckpt_dir if args.ckpt_dir else ""
        args.ema_weight = os.path.join(args.ckpt_dir, args.ema_weight) if args.ema_weight else ""

    return args


def create_model(config, checkpoints=None, freeze=False, load_filter=False, amp_level="O0"):
    # create model
    model = load_model_from_config(config, checkpoints, amp_level=amp_level)
    if freeze:
        model.set_train(False)
        model.set_grad(False)
        for _, p in model.parameters_and_names():
            p.requires_grad = False

    if load_filter:
        # TODO: Add DeepFloydDataFiltering
        raise NotImplementedError

    return model, None


def get_grad_reducer(is_parallel, parameters):
    if is_parallel:
        mean = ms.context.get_auto_parallel_context("gradients_mean")
        degree = ms.context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(parameters, mean, degree)
    else:
        grad_reducer = ops.functional.identity
    return grad_reducer


def get_loss_scaler(ms_loss_scaler="static", scale_value=1024, scale_factor=2, scale_window=1000):
    if ms_loss_scaler == "dynamic":
        from mindspore.amp import DynamicLossScaler

        loss_scaler = DynamicLossScaler(scale_value=scale_value, scale_factor=scale_factor, scale_window=scale_window)
    elif ms_loss_scaler == "static":
        from mindspore.amp import StaticLossScaler

        loss_scaler = StaticLossScaler(scale_value=scale_value)
    elif ms_loss_scaler in ("none", "None"):
        from mindspore.amp import StaticLossScaler

        loss_scaler = StaticLossScaler(1.0)
    else:
        raise NotImplementedError(f"Not support ms_loss_scaler: {ms_loss_scaler}")

    return loss_scaler


def load_model_from_config(config, ckpts=None, verbose=True, amp_level="O0"):
    model = instantiate_from_config(config.model)
    ignore_lora_key = len(ckpts) == 1

    if ckpts is not None:
        print(f"Loading model from {ckpts}")
        if isinstance(ckpts, str):
            ckpts = [ckpts]

        sd_dict = {}
        for ckpt in ckpts:
            assert ckpt.endswith(".ckpt")
            _sd_dict = ms.load_checkpoint(ckpt)
            sd_dict.update(_sd_dict)

            if "global_step" in sd_dict:
                global_step = sd_dict["global_step"]
                print(f"loaded ckpt from global step {global_step}")
                print(f"Global Step: {sd_dict['global_step']}")

        m, u = ms.load_param_into_net(model, sd_dict, strict_load=False)

        if len(m) > 0 and verbose:
            m = m if not ignore_lora_key else [k for k in m if "lora_" not in k]
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

    auto_mixed_precision(model, amp_level=amp_level)
    model.set_train(False)
    return model


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([emb.input_key for emb in conditioner.embedders]))


def get_batch(keys, value_dict, N: Union[List, ListConfig], dtype=ms.float32):
    # Hardcoded demo setups; might undergo some changes in the future

    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=np.prod(N)).reshape(N).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=np.prod(N)).reshape(N).tolist()
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = Tensor(
                np.tile(
                    np.array([value_dict["orig_height"], value_dict["orig_width"]]),
                    N
                    + [
                        1,
                    ],
                ),
                dtype,
            )
        elif key == "crop_coords_top_left":
            batch["crop_coords_top_left"] = Tensor(
                np.tile(
                    np.array([value_dict["crop_coords_top"], value_dict["crop_coords_left"]]),
                    N
                    + [
                        1,
                    ],
                ),
                dtype,
            )
        elif key == "aesthetic_score":
            batch["aesthetic_score"] = Tensor(
                np.tile(
                    np.array([value_dict["aesthetic_score"]]),
                    N
                    + [
                        1,
                    ],
                ),
                dtype,
            )
            batch_uc["aesthetic_score"] = Tensor(
                np.tile(
                    np.array([value_dict["negative_aesthetic_score"]]),
                    N
                    + [
                        1,
                    ],
                ),
                dtype,
            )
        elif key == "target_size_as_tuple":
            batch["target_size_as_tuple"] = Tensor(
                np.tile(
                    np.array([value_dict["target_height"], value_dict["target_width"]]),
                    N
                    + [
                        1,
                    ],
                ),
                dtype,
            )
        else:
            batch[key] = value_dict[key]

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], Tensor):
            batch_uc[key] = batch[key].copy()
    return batch, batch_uc


def get_discretization(discretization, sigma_min=0.03, sigma_max=14.61, rho=3.0):
    if discretization == "LegacyDDPMDiscretization":
        discretization_config = {
            "target": "gm.modules.diffusionmodules.discretizer.LegacyDDPMDiscretization",
        }
    elif discretization == "EDMDiscretization":
        discretization_config = {
            "target": "gm.modules.diffusionmodules.discretizer.EDMDiscretization",
            "params": {
                "sigma_min": sigma_min,
                "sigma_max": sigma_max,
                "rho": rho,
            },
        }
    else:
        raise NotImplementedError

    return discretization_config


def get_guider(guider="VanillaCFG", cfg_scale=5.0):
    if guider == "IdentityGuider":
        guider_config = {"target": "gm.modules.diffusionmodules.guiders.IdentityGuider"}
    elif guider == "VanillaCFG":
        scale = min(max(cfg_scale, 0.0), 100.0)

        dyn_thresh_config = {"target": "gm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"}
        guider_config = {
            "target": "gm.modules.diffusionmodules.guiders.VanillaCFG",
            "params": {"scale": scale, "dyn_thresh_config": dyn_thresh_config},
        }
    else:
        raise NotImplementedError
    return guider_config


def get_sampler(
    sampler_name,
    steps,
    discretization_config,
    guider_config,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=999.0,
    s_noise=1.0,
):
    if sampler_name in ("EulerEDMSampler", "HeunEDMSampler"):
        s_churn = max(s_churn, 0.0)
        s_tmin = max(s_tmin, 0.0)
        s_tmax = max(s_tmax, 0.0)
        s_noise = max(s_noise, 0.0)

        if sampler_name == "EulerEDMSampler":
            sampler = EulerEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        elif sampler_name == "HeunEDMSampler":
            raise NotImplementedError
        else:
            raise ValueError

    elif sampler_name in ("EulerAncestralSampler", "DPMPP2SAncestralSampler"):
        raise NotImplementedError
    elif sampler_name in ("DPMPP2MSampler",):
        raise NotImplementedError
    elif sampler_name in ("LinearMultistepSampler",):
        raise NotImplementedError
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler


def init_sampling(
    steps=40,
    num_cols=None,
    sampler="EulerEDMSampler",
    guider="VanillaCFG",
    discretization="LegacyDDPMDiscretization",
    img2img_strength=1.0,
    specify_num_samples=True,
    stage2strength=None,
):
    assert sampler in [
        "EulerEDMSampler",
        "HeunEDMSampler",
        "EulerAncestralSampler",
        "DPMPP2SAncestralSampler",
        "DPMPP2MSampler",
        "LinearMultistepSampler",
    ]
    assert guider in ["VanillaCFG", "IdentityGuider"]
    assert discretization in [
        "LegacyDDPMDiscretization",
        "EDMDiscretization",
    ]

    steps = min(max(steps, 1), 1000)
    num_rows = 1
    if specify_num_samples:
        num_cols = num_cols if num_cols else 2
        num_cols = min(max(num_cols, 1), 10)
    else:
        num_cols = num_cols if num_cols else 1

    guider_config = get_guider(guider)
    discretization_config = get_discretization(discretization)
    sampler = get_sampler(sampler, steps, discretization_config, guider_config)

    if img2img_strength < 1.0:
        raise NotImplementedError
    if stage2strength is not None:
        raise NotImplementedError

    return sampler, num_rows, num_cols


def _get_broadcast_datetime(rank_size=1, root_rank=0):
    time = datetime.now()
    time_list = [time.year, time.month, time.day, time.hour, time.minute, time.second, time.microsecond]

    if rank_size <= 1:
        return time_list

    bd_cast = ops.Broadcast(root_rank=root_rank)
    # only broadcast in distribution mode
    x = bd_cast((Tensor(time_list, dtype=ms.int32),))
    x = x[0].asnumpy().tolist()

    return x


def _do_sample(
    model,
    sampler,
    value_dict,
    num_samples,
    H,
    W,
    C,
    F,
    force_uc_zero_embeddings: List = None,
    batch2model_input: List = None,
    return_latents=False,
    filter=None,
    amp_level="O0",
):
    print("Sampling")

    if force_uc_zero_embeddings is None:
        force_uc_zero_embeddings = []
    if batch2model_input is None:
        batch2model_input = []

    num_samples = [num_samples]
    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        value_dict,
        num_samples,
        dtype=ms.float32 if amp_level not in ("O2", "O3") else ms.float16,
    )
    for key in batch:
        if isinstance(batch[key], Tensor):
            print(key, batch[key].shape)
        elif isinstance(batch[key], list):
            print(key, [len(i) for i in batch[key]])
        else:
            print(key, batch[key])
    print("Get Condition Done.")

    print("Embedding Starting...")
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=force_uc_zero_embeddings,
    )
    print("Embedding Done.")

    for k in c:
        if not k == "crossattn":
            c[k], uc[k] = map(
                lambda y: y[k][: int(np.prod(num_samples))],
                (c, uc)
                # lambda y: y[k][: math.prod(num_samples)], (c, uc)
            )

    additional_model_inputs = {}
    for k in batch2model_input:
        additional_model_inputs[k] = batch[k]

    shape = (np.prod(num_samples), C, H // F, W // F)
    randn = np.random.randn(*shape)

    def denoiser(input, sigma, c):
        return model.denoiser(model.model, input, sigma, c, **additional_model_inputs)

    print("Sample latent Starting...")
    samples_z = sampler(denoiser, randn, cond=c, uc=uc)
    print("Sample latent Done.")

    print("Decode latent Starting...")
    samples_x = model.decode_first_stage(Tensor(samples_z, ms.float32)).asnumpy()
    print("Decode latent Done.")

    samples = np.clip((samples_x + 1.0) / 2.0, a_min=0.0, a_max=1.0)

    if filter is not None:
        print("Filter Starting...")
        samples = filter(samples)
        print("Filter Done.")

    if return_latents:
        return samples, samples_z
    return samples


def embed_watermark(img):
    # TODO: Add Water Mark
    return img


def perform_save_locally(save_path, samples):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    samples = embed_watermark(samples)

    for sample in samples:
        sample = 255.0 * sample.transpose(1, 2, 0)
        Image.fromarray(sample.astype(np.uint8)).save(os.path.join(save_path, f"{base_count:09}.png"))
        base_count += 1


def save_checkpoint(model, path, only_save_lora=False):
    ckpt, ckpt_lora = [], []
    for n, p in model.parameters_and_names():
        if "lora_" in n:
            ckpt_lora.append({"name": n, "data": p})
        else:
            ckpt.append({"name": n, "data": p})

    if not only_save_lora:
        ms.save_checkpoint(ckpt, path)
        print(f"save checkpoint to {path}")

    if len(ckpt_lora) > 0:
        path_lora = path[: -len(".ckpt")] + "_lora.ckpt"
        ms.save_checkpoint(ckpt_lora, path_lora)
        print(f"save lora checkpoint to {path_lora}")
