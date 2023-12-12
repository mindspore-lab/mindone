import os
from datetime import datetime
from typing import List, Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import numpy as np
import yaml
from gm.modules.diffusionmodules.discretizer import Img2ImgDiscretizationWrapper, Txt2NoisyDiscretizationWrapper
from gm.modules.diffusionmodules.sampler import (
    AncestralSampler,
    DPMPP2MSampler,
    DPMPP2SAncestralSampler,
    EulerAncestralSampler,
    EulerEDMSampler,
    HeunEDMSampler,
    LinearMultistepSampler,
    PanGuEulerEDMSampler,
)
from gm.util import auto_mixed_precision, get_obj_from_str, instantiate_from_config, seed_everything
from omegaconf import DictConfig, ListConfig
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
    "1.0_768": (768, 768),
    "1.0_512": (512, 512),
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
    "PanGu-SDXL-base-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
    },
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
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
    if args.max_device_memory is not None:
        context.set_context(max_device_memory=args.max_device_memory)
        context.set_context(memory_optimize_level="O1", ascend_config={"atomic_clean_policy": 1})

    # Set Parallel
    if args.is_parallel:
        init()
        args.rank, args.rank_size, parallel_mode = get_rank(), get_group_size(), context.ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(device_num=args.rank_size, parallel_mode=parallel_mode, gradients_mean=True)
    else:
        args.rank, args.rank_size = 0, 1

    # data sink step
    if args.data_sink:
        assert args.dataset_load_tokenizer
        args.log_interval = args.sink_size
        if not (args.save_ckpt_interval >= args.sink_size and args.save_ckpt_interval % args.sink_size == 0):
            args.save_ckpt_interval = args.sink_size * max(1, (args.save_ckpt_interval // args.sink_size))
        if not (args.infer_interval >= args.sink_size and args.infer_interval % args.sink_size == 0):
            args.infer_interval = args.sink_size * max(1, (args.infer_interval // args.sink_size))

    # split weights path
    args.weight = args.weight.split(",") if len(args.weight) > 0 else ""

    # Directories and Save run settings
    if args.save_path_with_time:
        # FIXME: Bug when running with rank_table on MindSpore 2.2.1; This is not a problem when running with OpenMPI
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


def create_model(
    config: DictConfig,
    checkpoints: Union[str, List[str]] = "",
    freeze: bool = False,
    load_filter: bool = False,
    param_fp16: bool = False,
    amp_level: Literal["O0", "O1", "O2", "O3"] = "O0",
):
    # create model
    model = load_model_from_config(config.model, checkpoints, amp_level=amp_level)

    if freeze:
        model.set_train(False)
        model.set_grad(False)
        for _, p in model.parameters_and_names():
            p.requires_grad = False

    if param_fp16:
        convert_modules = (model.conditioner, model.first_stage_model)
        if isinstance(model.model, nn.Cell):
            convert_modules += (model.model,)
        else:
            assert hasattr(model, "stage1") and isinstance(model.stage1, nn.Cell)
            convert_modules += (model.stage1, model.stage2)

        for module in convert_modules:
            k_num, c_num = 0, 0
            for _, p in module.parameters_and_names():
                # filter norm/embedding position_ids param
                if ("position_ids" in p.name) or ("norm" in p.name):
                    # print(f"param {p.name} keep {p.dtype}") # disable print
                    k_num += 1
                else:
                    c_num += 1
                    p.set_dtype(ms.float16)

            print(f"Convert `{type(module).__name__}` param to fp16, keep/modify num {k_num}/{c_num}.")

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


def get_learning_rate(optim_comfig, total_step):
    base_lr = optim_comfig.get("base_learning_rate", 1.0e-6)
    if "scheduler_config" in optim_comfig:
        scheduler_config = optim_comfig.get("scheduler_config")
        scheduler = instantiate_from_config(scheduler_config)
        lr = [base_lr * scheduler(step) for step in range(total_step)]
    else:
        print(f"scheduler_config not exist, train with base_lr {base_lr}")
        lr = base_lr

    return lr


def get_optimizer(optim_comfig, lr, params, filtering=True):
    optimizer_config = optim_comfig.get("optimizer_config", {"target": "mindspore.nn.SGD"})

    def decay_filter(x):
        return "norm" not in x.name.lower() and "bias" not in x.name.lower()

    # filtering weight
    if filtering:
        weight_decay = optimizer_config.get("params", dict()).get("weight_decay", 1e-6)
        decay_params = list(filter(decay_filter, params))
        other_params = list(filter(lambda x: not decay_filter(x), params))
        group_params = []
        if len(decay_params) > 0:
            group_params.append({"params": decay_params, "weight_decay": weight_decay})
        if len(other_params) > 0:
            group_params.append({"params": other_params, "weight_decay": 0.0})
        group_params.append({"order_params": params})
        params = group_params
        print(
            f"Enable optimizer group param, "
            f"decay params num: {len(decay_params)}, "
            f"no decay params num: {len(other_params)}, "
            f"full params num: {len(decay_params) + len(other_params)}"
        )

    # build optimizer
    optimizer = get_obj_from_str(optimizer_config["target"])(
        params, learning_rate=lr, **optimizer_config.get("params", dict())
    )

    return optimizer


def load_model_from_config(model_config, ckpts=None, verbose=True, amp_level="O0"):
    model = instantiate_from_config(model_config)

    from gm.models.diffusion import DiffusionEngineMultiGraph

    if not isinstance(model, DiffusionEngineMultiGraph):
        if ckpts:
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
                ignore_lora_key = len(ckpts) == 1
                m = m if not ignore_lora_key else [k for k in m if "lora_" not in k]
                print("missing keys:")
                print(m)
            if len(u) > 0 and verbose:
                print("unexpected keys:")
                print(u)
        else:
            print(f"Warning: Loading checkpoint from {ckpts} fail")

        model = auto_mixed_precision(model, amp_level=amp_level)
        model.set_train(False)
    else:
        model.load_pretrained(ckpts, verbose=verbose)

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
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        else:
            raise ValueError
    elif sampler_name in ("AncestralSampler"):
        sampler = AncestralSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name in ("EulerAncestralSampler"):
        sampler = EulerAncestralSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
            eta=0.001,
        )
    elif sampler_name in ("DPMPP2SAncestralSampler"):
        sampler = DPMPP2SAncestralSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )

    elif sampler_name in ("DPMPP2MSampler",):
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name in ("LinearMultistepSampler",):
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler


def init_sampling(
    steps=40,
    num_cols=None,
    sampler="EulerEDMSampler",
    guider="VanillaCFG",
    guidance_scale=5.0,
    discretization="LegacyDDPMDiscretization",
    img2img_strength=1.0,
    specify_num_samples=True,
    stage2strength=None,
    other_scale=None,
    enable_pangu=False,
):
    assert sampler in [
        "EulerEDMSampler",
        "PanGuEulerEDMSampler",
        "HeunEDMSampler",
        "EulerAncestralSampler",
        "DPMPP2SAncestralSampler",
        "DPMPP2MSampler",
        "LinearMultistepSampler",
        "AncestralSampler",
    ]
    assert guider in ["VanillaCFG", "PanGuVanillaCFG", "IdentityGuider"]
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

    if enable_pangu:
        guider_config = pangu_get_guider(guider, cfg_scale=guidance_scale, other_scale=other_scale)
        discretization_config = get_discretization(discretization)
        sampler = pangu_get_sampler(sampler, steps, discretization_config, guider_config)
    else:
        guider_config = get_guider(guider, cfg_scale=guidance_scale)
        discretization_config = get_discretization(discretization)
        sampler = get_sampler(sampler, steps, discretization_config, guider_config)

    if img2img_strength < 1.0:
        print(f"WARNING: Wrapping {sampler.__class__.__name__} with Img2ImgDiscretizationWrapper")
        sampler.discretization = Img2ImgDiscretizationWrapper(sampler.discretization, strength=img2img_strength)
    if stage2strength is not None:
        sampler.discretization = Txt2NoisyDiscretizationWrapper(
            sampler.discretization, strength=stage2strength, original_steps=steps
        )

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
        # FIXME: save checkpoint bug on mindspore 2.1.0
        if "._backbone" in n:
            _index = n.find("._backbone")
            n = n[:_index] + n[_index + len("._backbone") :]
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


def get_interactive_image(image) -> Image.Image:
    if image is not None:
        image = Image.open(image)
        if not image.mode == "RGB":
            image = image.convert("RGB")
    return image


def load_img(image):
    if isinstance(image, str):
        image = get_interactive_image(image)
    if image is None:
        return None
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
    image = image.resize((width, height))
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)  # (h, w, c) -> (1, c, h, w)
    image = image / 127.5 - 1.0  # norm to (-1, 1)
    return image


# =============
# PanGu Draw
# =============


def pangu_get_batch(keys, value_dict, N: Union[List, ListConfig], dtype=ms.float32):
    # Hardcoded demo setups; might undergo some changes in the future
    photography_prefix = "霙 "
    aesthetic_prefix = "義 "
    cartoon_prefix = "饅 "
    batch = {}
    batch_uc = {}
    other_batch = []

    num_prompts = len(value_dict.get("prompt", []))
    N_reshape = [n * num_prompts for n in N]

    for key in keys:
        if key == "txt":
            batch["txt"] = np.repeat([value_dict["prompt"]], repeats=np.prod(N)).reshape(N_reshape).tolist()
            batch_uc["txt"] = np.repeat([value_dict["negative_prompt"]], repeats=np.prod(N)).reshape(N_reshape).tolist()
        elif key == "original_size_as_tuple":
            batch["original_size_as_tuple"] = Tensor(
                np.tile(
                    np.array([value_dict["orig_height"], value_dict["orig_width"]]),
                    N_reshape
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
                    N_reshape
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
                    N_reshape
                    + [
                        1,
                    ],
                ),
                dtype,
            )
            batch_uc["aesthetic_score"] = Tensor(
                np.tile(
                    np.array([value_dict["negative_aesthetic_score"]]),
                    N_reshape
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
                    N_reshape
                    + [
                        1,
                    ],
                ),
                dtype,
            )
        elif key == "target_size_as_ind":
            batch["target_size_as_ind"] = Tensor(
                np.tile(
                    np.array([value_dict["target_size_as_ind"]]),
                    N_reshape
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

    if "aesthetic_scale" in value_dict and value_dict["aesthetic_scale"] > 0:
        batch_aes = dict()
        for key in batch.keys():
            if isinstance(batch[key], Tensor):
                batch_aes[key] = batch[key].copy()
        batch_aes["txt"] = (
            np.repeat(
                [aesthetic_prefix + prompt for prompt in value_dict["prompt"]],
                repeats=np.prod(N),
            )
            .reshape(N_reshape)
            .tolist()
        )
        other_batch.append(batch_aes)
    if "anime_scale" in value_dict and value_dict["anime_scale"] > 0:
        batch_anime = dict()
        for key in batch.keys():
            if isinstance(batch[key], Tensor):
                batch_anime[key] = batch[key].copy()
        batch_anime["txt"] = (
            np.repeat(
                [cartoon_prefix + prompt for prompt in value_dict["prompt"]],
                repeats=np.prod(N),
            )
            .reshape(N_reshape)
            .tolist()
        )
        other_batch.append(batch_anime)
    if "photography_scale" in value_dict and value_dict["photography_scale"] > 0:
        batch_photography = dict()
        for key in batch.keys():
            if isinstance(batch[key], Tensor):
                batch_photography[key] = batch[key].copy()
        batch_photography["txt"] = (
            np.repeat(
                [photography_prefix + prompt for prompt in value_dict["prompt"]],
                repeats=np.prod(N),
            )
            .reshape(N_reshape)
            .tolist()
        )
        other_batch.append(batch_photography)

    return batch, batch_uc, other_batch


def pangu_get_guider(guider="PanGuVanillaCFG", cfg_scale=7.5, other_scale=None):
    if guider == "IdentityGuider":
        guider_config = {"target": "gm.modules.diffusionmodules.guiders.IdentityGuider"}
    elif guider == "PanGuVanillaCFG":
        scale = min(max(cfg_scale, 0.0), 100.0)

        dyn_thresh_config = {"target": "gm.modules.diffusionmodules.sampling_utils.PanGuNoDynamicThresholding"}
        guider_config = {
            "target": "gm.modules.diffusionmodules.guiders.PanGuVanillaCFG",
            "params": {"scale": scale, "dyn_thresh_config": dyn_thresh_config, "other_scale": other_scale},
        }
    else:
        raise NotImplementedError
    return guider_config


def pangu_get_sampler(
    sampler_name,
    steps,
    discretization_config,
    guider_config,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=999.0,
    s_noise=1.0,
):
    if sampler_name in ("AncestralSampler"):
        sampler = AncestralSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name in ("EulerAncestralSampler"):
        sampler = EulerAncestralSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
            eta=0.001,
        )
    elif sampler_name in ("DPMPP2SAncestralSampler"):
        sampler = DPMPP2SAncestralSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )

    elif sampler_name in ("DPMPP2MSampler",):
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name in ("LinearMultistepSampler",):
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name in ("EulerEDMSampler", "HeunEDMSampler", "PanGuEulerEDMSampler"):
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
        elif sampler_name == "PanGuEulerEDMSampler":
            sampler = PanGuEulerEDMSampler(
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
            sampler = HeunEDMSampler(
                num_steps=steps,
                discretization_config=discretization_config,
                guider_config=guider_config,
                s_churn=s_churn,
                s_tmin=s_tmin,
                s_tmax=s_tmax,
                s_noise=s_noise,
                verbose=True,
            )
        else:
            raise ValueError
    else:
        raise ValueError(f"unknown sampler {sampler_name}!")

    return sampler
