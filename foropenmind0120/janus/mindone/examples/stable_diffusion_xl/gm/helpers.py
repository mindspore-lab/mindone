import logging
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
    LCMSampler,
    LinearMultistepSampler,
)
from gm.util import auto_mixed_precision, get_obj_from_str, instantiate_from_config, seed_everything
from omegaconf import DictConfig, ListConfig
from PIL import Image

import mindspore as ms
from mindspore import Parameter, Tensor, context, nn, ops
from mindspore.communication.management import get_group_size, get_rank, init


class BroadCast(nn.Cell):
    def __init__(self, root_rank):
        super().__init__()
        self.broadcast = ops.Broadcast(root_rank)

    def construct(self, x):
        return (self.broadcast((x,)))[0]


SD_XL_BASE_RATIOS = {
    # W/H ratio: (W, H)
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
    "SDXL-refiner-1.0": {
        "H": 1024,
        "W": 1024,
        "C": 4,
        "f": 8,
        "is_legacy": True,
    },
}


_ema_op = ops.MultitypeFuncGraph("grad_ema_op")


@_ema_op.register("Tensor", "Tensor", "Tensor")
def _ema_weights(factor, ema_weight, weight):
    return ops.assign(ema_weight, ema_weight * factor + weight * (1 - factor))


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

    try:
        if args.jit_level in ["O0", "O1", "O2"]:
            ms.set_context(jit_config={"jit_level": args.jit_level})
            print(f"set jit_level: {args.jit_level}.")
        else:
            print(
                f"WARNING: Unsupport jit_level: {args.jit_level}. The framework automatically selects the execution method"
            )
    except Exception:
        print(
            "WARNING: The current jit_level is not suitable because current MindSpore version or mode does not match,"
            "please ensure the MindSpore version >= ms2.3.0, and use GRAPH_MODE."
        )
    # Set Parallel
    if args.is_parallel:
        init()
        args.rank, args.rank_size, parallel_mode = get_rank(), get_group_size(), context.ParallelMode.DATA_PARALLEL
        if args.task != "cache":
            context.set_auto_parallel_context(
                device_num=args.rank_size, parallel_mode=parallel_mode, gradients_mean=True
            )
    else:
        args.rank, args.rank_size = 0, 1

    # data sink step
    if args.data_sink:
        if args.sink_queue_size > 1:
            os.environ["MS_DATASET_SINK_QUEUE"] = f"{args.sink_queue_size}"
            print(f"WARNING: Set env `MS_DATASET_SINK_QUEUE` to {args.sink_queue_size}.")
        else:
            if os.environ.get("MS_DATASET_SINK_QUEUE") is None:
                os.environ["MS_DATASET_SINK_QUEUE"] = "10"
                print("WARNING: Set env `MS_DATASET_SINK_QUEUE` to 10.")
            else:
                print(f"env `MS_DATASET_SINK_QUEUE`: {os.environ.get('MS_DATASET_SINK_QUEUE')}")

        assert args.dataset_load_tokenizer
        args.log_interval = args.sink_size
        if not (args.save_ckpt_interval >= args.sink_size and args.save_ckpt_interval % args.sink_size == 0):
            args.save_ckpt_interval = args.sink_size * max(1, (args.save_ckpt_interval // args.sink_size))
        if not (args.infer_interval >= args.sink_size and args.infer_interval % args.sink_size == 0):
            args.infer_interval = args.sink_size * max(1, (args.infer_interval // args.sink_size))

    # split weights path
    args.weight = args.weight.split(",") if args.weight is not None and len(args.weight) > 0 else ""

    # cache
    if "cache_latent" in args and "cache_text_embedding" in args:
        assert (
            args.cache_latent == args.cache_text_embedding
        ), "Please confirm that `args.cache_latent` and `args.cache_text_embedding` are consistent"

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
    if args.max_num_ckpt is not None and args.max_num_ckpt <= 0:
        raise ValueError("args.max_num_ckpt must be None or a positive integer!")

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


def get_all_reduce_config(model):
    trainable_params = []
    all_reduce_fusion_config = []

    i = -1

    if model.conditioner is not None and len(model.conditioner.trainable_params()) > 0:
        for p in model.conditioner.trainable_params():
            trainable_params.append(p)
            i += 1
        all_reduce_fusion_config.append(i)

    unet = model.model.diffusion_model
    for p in unet.time_embed.trainable_params():
        trainable_params.append(p)
        i += 1
    for p in unet.label_emb.trainable_params():
        trainable_params.append(p)
        i += 1
    all_reduce_fusion_config.append(i)

    for block in unet.input_blocks:
        for p in block.trainable_params():
            trainable_params.append(p)
            i += 1
        all_reduce_fusion_config.append(i)

    for p in unet.middle_block.trainable_params():
        trainable_params.append(p)
        i += 1
    all_reduce_fusion_config.append(i)

    for block in unet.output_blocks:
        for p in block.trainable_params():
            trainable_params.append(p)
            i += 1
        all_reduce_fusion_config.append(i)

    for p in unet.out.trainable_params():
        trainable_params.append(p)
        i += 1
    if hasattr(unet, "id_predictor"):
        for p in unet.id_predictor.trainable_params():
            trainable_params.append(p)
            i += 1
    all_reduce_fusion_config.append(i)

    trainable_params = tuple(trainable_params)

    num_trainable_params = (
        len(model.model.trainable_params()) + len(model.conditioner.trainable_params())
        if model.conditioner is not None
        else len(model.model.trainable_params())
    )
    assert len(trainable_params) == num_trainable_params

    return trainable_params, all_reduce_fusion_config


def create_model(
    config: DictConfig,
    checkpoints: Union[str, List[str]] = "",
    freeze: bool = False,
    load_filter: bool = False,
    param_fp16: bool = False,
    amp_level: Literal["O0", "O1", "O2", "O3"] = "O0",
    textual_inversion_ckpt: str = None,
    placeholder_token: str = None,
    num_vectors: int = None,
    load_first_stage_model: bool = True,
    load_conditioner: bool = True,
):
    from gm.models.diffusion import DiffusionEngine, DiffusionEngineControlNet, DiffusionEngineDreamBooth

    assert config.model["target"] in [
        "gm.models.diffusion.DiffusionEngine",
        "gm.models.diffusion.DiffusionEngineDreamBooth",
        "gm.models.diffusion.DiffusionEngineControlNet",
    ], f"Not supported for `class {config.model['target']}`"

    # create diffusion engine
    config.model["params"]["load_first_stage_model"] = load_first_stage_model
    config.model["params"]["load_conditioner"] = load_conditioner
    target_map = {
        "gm.models.diffusion.DiffusionEngine": DiffusionEngine,
        "gm.models.diffusion.DiffusionEngineDreamBooth": DiffusionEngineDreamBooth,
        "gm.models.diffusion.DiffusionEngineControlNet": DiffusionEngineControlNet,
    }
    sdxl = target_map[config.model["target"]](**config.model.get("params", dict()))

    # load pretrained
    sdxl.load_pretrained(checkpoints)

    # set auto-mix-precision
    sdxl = auto_mixed_precision(sdxl, amp_level=amp_level)
    sdxl.set_train(False)

    # set model parameter/weight dtype to fp16
    if param_fp16:
        print(
            "!!! WARNING: Converted the weight to `fp16`, that may lead to unstable training. You can turn it off by setting `--param_fp16=False`"
        )
        convert_sdxl_to_fp16(sdxl)

    if freeze:
        sdxl.set_train(False)
        sdxl.set_grad(False)
        for _, p in sdxl.parameters_and_names():
            p.requires_grad = False

    if load_filter:
        # TODO: Add DeepFloydDataFiltering
        raise NotImplementedError

    if textual_inversion_ckpt is not None:
        assert os.path.exists(textual_inversion_ckpt), f"{textual_inversion_ckpt} does not exist!"
        from gm.modules.textual_inversion.manager import TextualInversionManager

        manager = TextualInversionManager(sdxl, placeholder_token, num_vectors)
        manager.load_checkpoint_textual_inversion(textual_inversion_ckpt, verbose=True)
        return (sdxl, manager), None

    return sdxl, None


def convert_sdxl_to_fp16(sdxl):
    vae = sdxl.first_stage_model
    text_encoders = sdxl.conditioner
    unet = sdxl.model

    convert_to_fp16(vae)
    convert_to_fp16(text_encoders)
    convert_to_fp16(unet)


def convert_to_fp16(model, keep_norm_fp32=True):
    if model is not None:
        assert isinstance(model, nn.Cell)

        k_num, c_num = 0, 0
        for _, p in model.parameters_and_names():
            # filter norm/embedding position_ids param
            if keep_norm_fp32 and ("norm" in p.name):
                # print(f"param {p.name} keep {p.dtype}") # disable print
                k_num += 1
            elif "position_ids" in p.name:
                k_num += 1
            else:
                c_num += 1
                p.set_dtype(ms.float16)

        print(f"Convert `{type(model).__name__}` param to fp16, keep/modify num {k_num}/{c_num}.")

    return model


def get_grad_reducer(is_parallel, parameters):
    if is_parallel:
        mean = ms.context.get_auto_parallel_context("gradients_mean")
        degree = ms.context.get_auto_parallel_context("device_num")
        grad_reducer = nn.DistributedGradReducer(parameters, mean, degree)
    else:
        grad_reducer = nn.Identity()
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


def get_learning_rate(optim_config, total_step, scaler=1.0):
    base_lr = optim_config.get("base_learning_rate", 1.0e-6)
    scaled_lr = scaler * base_lr
    if "scheduler_config" in optim_config:
        scheduler_config = optim_config.get("scheduler_config")
        scheduler = instantiate_from_config(scheduler_config)
        if hasattr(scheduler, "lr_max_decay_steps") and scheduler.lr_max_decay_steps == -1:
            scheduler.lr_max_decay_steps = total_step
        lr = [scaled_lr * scheduler(step) for step in range(total_step)]
    else:
        print(f"scheduler_config not exist, train with base_lr {base_lr} and lr_scaler {scaler}")
        lr = scaled_lr

    return lr


def _scale_lr(group_params, lr, scaler):
    """scale lr of a particular group of params"""
    new_groups = list()
    for group in group_params:
        scale_params, unscale_params = list(), list()
        for params in group["params"]:
            name = params.name.lower()
            # keys below are adapted for ControlNet training
            if "zero_conv" in name or "input_hint_block" in name or "middle_block_out" in name:
                scale_params.append(params)
            else:
                unscale_params.append(params)

        new_groups.append(
            {
                "params": scale_params,
                "weight_decay": group["weight_decay"],
                "lr": lr * scaler,
            }
        )
        new_groups.append(
            {
                "params": unscale_params,
                "weight_decay": group["weight_decay"],
                "lr": lr,
            }
        )
    return new_groups


def get_optimizer(optim_config, lr, params, filtering=True, group_lr_scaler=None):
    optimizer_config = optim_config.get("optimizer_config", {"target": "mindspore.nn.SGD"})

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
        print(
            f"Enable optimizer group param, "
            f"decay params num: {len(decay_params)}, "
            f"no decay params num: {len(other_params)}, "
            f"full params num: {len(decay_params) + len(other_params)}"
        )
        if isinstance(group_lr_scaler, float) or isinstance(group_lr_scaler, int):
            group_params = _scale_lr(group_params, lr, group_lr_scaler)
        group_params.append({"order_params": params})
        params = group_params

    # build optimizer
    optimizer = get_obj_from_str(optimizer_config["target"])(
        params, learning_rate=lr, **optimizer_config.get("params", dict())
    )

    return optimizer


def load_checkpoint(model, ckpt, verbose=True, remove_prefix=None):
    sd_dict = ms.load_checkpoint(ckpt)

    if remove_prefix is not None:
        new_sd_dict = {}
        for k in sd_dict:
            if k.startswith(remove_prefix):
                new_k = k[len(remove_prefix) :]
            else:
                new_k = k[:]
            new_sd_dict[new_k] = sd_dict[k]
        sd_dict = new_sd_dict

    m, u = ms.load_param_into_net(model, sd_dict, strict_load=False)

    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)


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
        elif key == "clip_img":
            batch["clip_img"] = value_dict["clip_img"]
            batch_uc["clip_img"] = None
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


def get_discretization(discretization, sigma_min=0.002, sigma_max=80, rho=7.0):
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
    elif discretization == "DiffusersDDPMDiscretization":
        discretization_config = {
            "target": "gm.modules.diffusionmodules.discretizer.DiffusersDDPMDiscretization",
        }
    else:
        raise NotImplementedError

    return discretization_config


def get_guider(guider="VanillaCFG", cfg_scale=5.0, dyn_thresh_config=None):
    if guider == "IdentityGuider":
        guider_config = {"target": "gm.modules.diffusionmodules.guiders.IdentityGuider"}
    elif guider == "VanillaCFG":
        scale = min(max(cfg_scale, 0.0), 100.0)

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
    elif sampler_name == "AncestralSampler":
        sampler = AncestralSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "EulerAncestralSampler":
        sampler = EulerAncestralSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
            eta=0.001,
        )
    elif sampler_name == "DPMPP2SAncestralSampler":
        sampler = DPMPP2SAncestralSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )

    elif sampler_name == "DPMPP2MSampler":
        sampler = DPMPP2MSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LinearMultistepSampler":
        sampler = LinearMultistepSampler(
            num_steps=steps,
            discretization_config=discretization_config,
            guider_config=guider_config,
            verbose=True,
        )
    elif sampler_name == "LCMSampler":
        sampler = LCMSampler(
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
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    thresholding=False,
    dynamic_thresholding_ratio=0.995,
    sample_max_value=1.0,
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
        "AncestralSampler",
        "LCMSampler",
    ]
    assert guider in ["VanillaCFG", "IdentityGuider"]
    if isinstance(discretization, str):
        assert discretization in [
            "LegacyDDPMDiscretization",
            "EDMDiscretization",
            "DiffusersDDPMDiscretization",
        ]
        discretization_config = get_discretization(discretization, sigma_min, sigma_max, rho)
    elif isinstance(discretization, DictConfig):
        discretization_config = discretization
    else:
        raise TypeError("discretization must be str or DictConfig")

    steps = min(max(steps, 1), 1000)
    num_rows = 1
    if specify_num_samples:
        num_cols = num_cols if num_cols else 2
        num_cols = min(max(num_cols, 1), 10)
    else:
        num_cols = num_cols if num_cols else 1

    if thresholding:
        dyn_thresh_config = {
            "target": "gm.modules.diffusionmodules.sampling_utils.DynamicThresholding",
            "params": {
                "dynamic_thresholding_ratio": dynamic_thresholding_ratio,
                "sample_max_value": sample_max_value,
            },
        }
    else:
        dyn_thresh_config = {"target": "gm.modules.diffusionmodules.sampling_utils.NoDynamicThresholding"}

    guider_config = get_guider(guider, cfg_scale=guidance_scale, dyn_thresh_config=dyn_thresh_config)
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

    # only broadcast in distribution mode
    bd_cast = BroadCast(root_rank=root_rank)(Tensor(time_list, dtype=ms.int32))
    x = bd_cast.asnumpy().tolist()

    return x


def embed_watermark(img):
    # TODO: Add Water Mark
    return img


def concat_images(images: list, num_cols: int):
    images = np.concatenate(images, axis=0)
    _n, _c, _h, _w = images.shape
    row, col = (int(_n / num_cols), num_cols) if _n % num_cols == 0 else (_n, 1)

    images = images.reshape((row, col, _c, _h, _w)).transpose(2, 0, 3, 1, 4).reshape(1, _c, row * _h, col * _w)

    return images


def perform_save_locally(save_path, samples, num_cols=1):
    os.makedirs(os.path.join(save_path), exist_ok=True)
    base_count = len(os.listdir(os.path.join(save_path)))
    if isinstance(samples, np.ndarray):
        samples = [samples]
    samples = embed_watermark(samples)
    samples = concat_images(samples, num_cols=num_cols)

    for sample in samples:
        sample = 255.0 * sample.transpose(1, 2, 0)
        Image.fromarray(sample.astype(np.uint8)).save(os.path.join(save_path, f"{base_count:09}.png"))
        base_count += 1


def _build_lora_ckpt_path(ckpt_path, save_ema_ckpt=False, save_lora_ckpt=True):
    if save_lora_ckpt and not save_ema_ckpt:
        path = ckpt_path.replace(".ckpt", "_lora.ckpt")
    elif save_lora_ckpt and save_ema_ckpt:
        path = ckpt_path.replace(".ckpt", "_lora_ema.ckpt")
    elif not save_lora_ckpt and save_ema_ckpt:
        path = ckpt_path.replace(".ckpt", "_ema.ckpt")
    else:
        path = ckpt_path

    return path


def save_checkpoint(model, path, ckpt_queue, max_num_ckpt, only_save_lora=False):
    ckpt, ckpt_lora, ckpt_lora_ema, ckpt_ema = [], [], [], []
    for n, p in model.parameters_and_names():
        # FIXME: save checkpoint bug on mindspore 2.1.0
        if "._backbone" in n:
            _index = n.find("._backbone")
            n = n[:_index] + n[_index + len("._backbone") :]
        if "lora_" in n and "ema." not in n:
            ckpt_lora.append({"name": n, "data": p})
        elif "lora_" in n and "ema." in n:
            _index = n.find("ema.")
            n = n[:_index] + n[_index + len("ema.") :]
            ckpt_lora_ema.append({"name": n, "data": p})
        elif "lora_" not in n and "ema." in n:
            _index = n.find("ema.")
            n = n[:_index] + n[_index + len("ema.") :]
            ckpt_ema.append({"name": n, "data": p})
        else:
            ckpt.append({"name": n, "data": p})

    delete_checkpoint(ckpt_queue, max_num_ckpt, only_save_lora)

    if not only_save_lora:
        ms.save_checkpoint(ckpt, path)
        print(f"save checkpoint to {path}")
        if len(ckpt_ema) > 0:
            path_ema = _build_lora_ckpt_path(path, save_ema_ckpt=True, save_lora_ckpt=False)
            ms.save_checkpoint(ckpt_ema, path_ema)
            print(f"save ema checkpoint to {path_ema}")

    if len(ckpt_lora) > 0:
        path_lora = _build_lora_ckpt_path(path)
        ms.save_checkpoint(ckpt_lora, path_lora)
        print(f"save lora checkpoint to {path_lora}")
        if len(ckpt_lora_ema) > 0:
            path_lora_ema = _build_lora_ckpt_path(path, save_ema_ckpt=True)
            ms.save_checkpoint(ckpt_lora_ema, path_lora_ema)
            print(f"save ema lora checkpoint to {path_lora_ema}")


def delete_checkpoint(ckpt_queue, max_num_ckpt, only_save_lora):
    """
    Only keep the latest `max_num_ckpt` ckpts while training. If max_num_ckpt == 0, keep all ckpts.
    """
    if max_num_ckpt is not None and len(ckpt_queue) >= max_num_ckpt:
        del_ckpt = ckpt_queue.pop(0)
        del_ckpt_lora = _build_lora_ckpt_path(del_ckpt)
        del_ckpt_ema = _build_lora_ckpt_path(del_ckpt, save_ema_ckpt=True, save_lora_ckpt=False)
        del_ckpt_lora_ema = _build_lora_ckpt_path(del_ckpt, save_ema_ckpt=True)
        if only_save_lora:
            del_ckpts = [del_ckpt_lora, del_ckpt_lora_ema]
        else:
            del_ckpts = [del_ckpt, del_ckpt_lora, del_ckpt_ema, del_ckpt_lora_ema]

        for to_del in del_ckpts:
            if os.path.isfile(to_del):
                try:
                    os.remove(to_del)
                    logging.debug(
                        f"The ckpt file {to_del} is deleted, because the number of ckpt files exceeds the limit {max_num_ckpt}."
                    )
                except OSError as e:
                    logging.exception(e)
            else:
                logging.debug(
                    f"The ckpt file {to_del} to be deleted doesn't exist. If lora is not used for training, it's normal that lora ckpt doesn't exist."
                )


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


class EMA(nn.Cell):
    """
    Args:
        updates: number of ema updates, which can be restored from resumed training.
    """

    def __init__(self, network, ema_decay=0.9999, updates=0, trainable_only=True):
        super().__init__()
        # TODO: net.trainable_params() is more reasonable?
        if trainable_only:
            self.net_weight = ms.ParameterTuple(network.trainable_params())
        else:
            self.net_weight = ms.ParameterTuple(network.get_parameters())
        self.ema_weight = self.net_weight.clone(prefix="ema", init="same")
        self.ema_decay = ema_decay
        self.updates = Parameter(Tensor(updates, ms.float32), requires_grad=False)
        self.hyper_map = ops.HyperMap()

    def ema_update(self):
        """Update EMA parameters."""
        self.updates += 1
        d = self.ema_decay * (1 - ops.exp(-self.updates / 2000))
        # update trainable parameters
        success = self.hyper_map(ops.partial(_ema_op, d), self.ema_weight, self.net_weight)
        self.updates = ops.depend(self.updates, success)
        return self.updates
