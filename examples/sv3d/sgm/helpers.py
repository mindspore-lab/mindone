import copy
import logging
import os
from typing import List, Optional, Union

from omegaconf import OmegaConf

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

import numpy as np
from omegaconf import DictConfig, ListConfig
from sgm.util import auto_mixed_precision, instantiate_from_config

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

_logger = logging.getLogger("")  # get the root _logger


class BroadCast(nn.Cell):
    def __init__(self, root_rank):
        super().__init__()
        self.broadcast = ops.Broadcast(root_rank)

    def construct(self, x):
        return (self.broadcast((x,)))[0]


def create_model_sv3d(
    config: DictConfig,  # model cfg
    train_config: Optional[DictConfig] = None,
    checkpoints: Union[str, List[str]] = None,
    freeze: bool = False,
    load_filter: bool = False,
    param_fp16: bool = False,
    amp_level: Literal["O0", "O1", "O2", "O3"] = "O0",
    textual_inversion_ckpt: str = None,
    placeholder_token: str = None,
    num_vectors: int = None,
    load_first_stage_model: bool = True,
    load_conditioner: bool = True,
    config_arch_toload_vanilla_sv3d_ckpt: bool = False,
):
    if train_config:
        config.model.params = OmegaConf.merge(config.model.params, train_config.model)
        config.model.params.network_config.params["use_recompute"] = True

    # create model
    model = load_model_from_config(
        config.model,
        checkpoints,
        amp_level=amp_level,
        load_first_stage_model=load_first_stage_model,
        load_conditioner=load_conditioner,
        config_arch_toload_vanilla_sv3d_ckpt=config_arch_toload_vanilla_sv3d_ckpt,
    )

    if freeze:
        model.set_train(False)
        model.set_grad(False)
        for _, p in model.parameters_and_names():
            p.requires_grad = False

    if param_fp16:
        convert_modules = ()
        if load_conditioner:
            convert_modules += (model.conditioner,)
        if load_first_stage_model:
            convert_modules += (model.first_stage_model,)

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

            print(f"Convert '{type(module).__name__}' param to fp16, keep/modify num {k_num}/{c_num}.")

    if load_filter:
        # TODO: Add DeepFloydDataFiltering
        raise NotImplementedError

    if textual_inversion_ckpt is not None:
        assert os.path.exists(textual_inversion_ckpt), f"{textual_inversion_ckpt} does not exist!"
        from sgm.modules.textual_inversion.manager import TextualInversionManager

        manager = TextualInversionManager(model, placeholder_token, num_vectors)
        manager.load_checkpoint_textual_inversion(textual_inversion_ckpt, verbose=True)
        return (model, manager), None

    return model, None


def load_model_from_config(
    model_config,
    ckpts=None,
    verbose=True,
    amp_level="O0",
    load_first_stage_model=True,
    load_conditioner=True,
    config_arch_toload_vanilla_sv3d_ckpt=False,
):
    model_config["params"]["load_first_stage_model"] = load_first_stage_model
    model_config["params"]["load_conditioner"] = load_conditioner
    model_config["params"]["config_arch_toload_vanilla_sv3d_ckpt"] = config_arch_toload_vanilla_sv3d_ckpt
    model = instantiate_from_config(model_config)

    # from sgm.models.diffusion import DiffusionEngineMultiGraph

    if ckpts:
        _logger.info(f"Loading model from {ckpts} with the cfg file")
        # if not isinstance(model, DiffusionEngineMultiGraph):
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

        # FIXME: parameter auto-prefix name bug on mindspore 2.2.10
        sd_dict = {k.replace("._backbone", ""): v for k, v in sd_dict.items()}

        # filter first_stage_model and conditioner
        _keys = copy.deepcopy(list(sd_dict.keys()))
        for _k in _keys:
            if not load_first_stage_model and _k.startswith("first_stage_model."):
                sd_dict.pop(_k)
            if not load_conditioner and _k.startswith("conditioner."):
                sd_dict.pop(_k)

        # make svd-d19a808f ckpt to accomodate sv3d_u.yaml setup: [1280, 768] -> [1280, 256]
        if not config_arch_toload_vanilla_sv3d_ckpt:
            sd_dict["model.diffusion_model.label_emb.0.0.weight"] = Parameter(
                sd_dict["model.diffusion_model.label_emb.0.0.weight"][:, :256],
                name="model.diffusion_model.label_emb.0.0.weight",
            )

        m, u = ms.load_param_into_net(model, sd_dict, strict_load=False)

        # actually not necessary as the api already returns missing keys
        if len(m) > 0 and verbose:
            ignore_lora_key = len(ckpts) == 1
            m = m if not ignore_lora_key else [k for k in m if "lora_" not in k]
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)
        # else:
        #     print(f"fred load diffusion multigraph {isinstance(model, DiffusionEngineMultiGraph)}")
        #     model.load_pretrained(ckpts, verbose=verbose)
    else:
        _logger.warning("No checkpoints were provided.")

    # if not isinstance(model, DiffusionEngineMultiGraph):
    model = auto_mixed_precision(model, amp_level=amp_level)
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
