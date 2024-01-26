import logging
import os

import mindspore as ms

from mindone.utils.config import instantiate_from_config
from mindone.utils.params import load_param_into_net_with_filter

logger = logging.getLogger()


def merge_motion_lora_to_unet(unet, lora_ckpt_path, alpha=1.0):
    """
    Merge lora weights to motion modules of UNet cell. Make sure motion module checkpoint has been loaded before invoking this function.

    Args:
        unet: nn.Cell
        lora_ckpt_path: path to lora checkpoint
        alpha: the strength of LoRA, typically in range [0, 1]
    Returns:
        unet with updated weights

    Note: expect format
        Case 1: source from torch checkpoint
        lora pname:
            model.diffusion_model.input_blocks.1.2.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.to_out_lora.down.weight
            = {attn_layer}{lora_postfix}
            = {attn_layer}.processor.{to_q/k/v/out}_lora.{down/up}.weight
        mm attn dense weight name:
            model.diffusion_model.input_blocks.1.2.temporal_transformer.transformer_blocks.0.attention_blocks.1.to_out.0.weight
            = {attn_layer}.{to_q/k/v/out.0}.weight
        Case 2: source from ms finetuned
        lora pname:
            model.diffusion_model.output_blocks.1.2.temporal_transformer.transformer_blocks.0.attention_blocks.0.to_out.0.lora_down.weight
            = {attn_layer}.{to_q/k/v/out.0}.lora_{down/up}.weight
            remove .lora_{down/up}
    """
    lora_pdict = ms.load_checkpoint(lora_ckpt_path)
    unet_pdict = unet.parameters_dict()

    for lora_pname in lora_pdict:
        is_from_torch = "_lora." in lora_pname
        if ("lora.down." in lora_pname) or ("lora_down." in lora_pname):  # skip lora.up
            lora_down_pname = lora_pname
            if is_from_torch:
                lora_up_pname = lora_pname.replace("lora.down.", "lora.up.")
            else:
                lora_up_pname = lora_pname.replace("lora_down.", "lora_up.")

            # 1. locate the target attn dense layer weight (q/k/v/out) by param name
            if is_from_torch:
                attn_pname = (
                    lora_pname.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
                )
                attn_pname = attn_pname.replace("to_out.", "to_out.0.")
            else:
                attn_pname = lora_pname.replace("lora_down.", "").replace("lora_up.", "")

            # 2. merge lora up and down weight to target dense layer weight
            down_weight = lora_pdict[lora_down_pname]
            up_weight = lora_pdict[lora_up_pname]

            dense_weight = unet_pdict[attn_pname].value()
            merged_weight = dense_weight + alpha * ms.ops.matmul(up_weight, down_weight)

            unet_pdict[attn_pname].set_data(merged_weight)

    logger.info(f"Inspected LoRA rank: {down_weight.shape[0]}")

    return unet


def merge_motion_lora_to_mm_pdict(mm_param_dict, lora_ckpt_path, alpha=1.0):
    """
    Merge lora weights to montion module param dict. So that we don't need to load param dict to UNet twice.
    Args:
        mm_param_dict: motion module param dict
        lora_ckpt_path: path to lora checkpoint
        alpha: the strength of LoRA, typically in range [0, 1]
    Returns:
        updated motion module param dict
    """
    lora_pdict = ms.load_checkpoint(lora_ckpt_path)

    for lora_pname in lora_pdict:
        if "lora.down." in lora_pname:  # skip lora.up
            lora_down_pname = lora_pname
            lora_up_pname = lora_pname.replace("lora.down.", "lora.up.")

            # 1. locate the target attn dense layer weight (q/k/v/out) by param name
            attn_pname = (
                lora_pname.replace("processor.", "").replace("_lora", "").replace("down.", "").replace("up.", "")
            )
            attn_pname = attn_pname.replace("to_out.", "to_out.0.")

            # 2. merge lora up and down weight to target dense layer weight
            down_weight = lora_pdict[lora_down_pname]
            up_weight = lora_pdict[lora_up_pname]

            dense_weight = mm_param_dict[attn_pname].value()
            merged_weight = dense_weight + alpha * ms.ops.matmul(up_weight, down_weight)

            mm_param_dict[attn_pname].set_data(merged_weight)

    return mm_param_dict


def update_unet2d_params_for_unet3d(ckpt_param_dict, unet3d_type="adv2"):
    # after injecting temporal moduels to unet2d cell, param name of some layers are changed.
    # apply the change to ckpt param names as well to load all unet ckpt params to unet3d cell

    # map the name change from 2d to 3d, annotated from vimdiff compare,
    if unet3d_type == "adv2":
        prefix_mapping = {
            "model.diffusion_model.middle_block.2": "model.diffusion_model.middle_block.3",
            "model.diffusion_model.output_blocks.2.1": "model.diffusion_model.output_blocks.2.2",
            "model.diffusion_model.output_blocks.5.2": "model.diffusion_model.output_blocks.5.3",
            "model.diffusion_model.output_blocks.8.2": "model.diffusion_model.output_blocks.8.3",
            "model.diffusion_model.out.0": "model.diffusion_model.conv_norm_out",
            "model.diffusion_model.out.2.conv": "model.diffusion_model.out.1.conv",
        }
    elif unet3d_type == "adv1":
        prefix_mapping = {
            "model.diffusion_model.output_blocks.2.1": "model.diffusion_model.output_blocks.2.2",
            "model.diffusion_model.output_blocks.5.2": "model.diffusion_model.output_blocks.5.3",
            "model.diffusion_model.output_blocks.8.2": "model.diffusion_model.output_blocks.8.3",
            "model.diffusion_model.out.0": "model.diffusion_model.conv_norm_out",
            "model.diffusion_model.out.2.conv": "model.diffusion_model.out.1.conv",
        }

    pnames = list(ckpt_param_dict.keys())
    for pname in pnames:
        for prefix_2d, prefix_3d in prefix_mapping.items():
            if pname.startswith(prefix_2d):
                new_pname = pname.replace(prefix_2d, prefix_3d)
                ckpt_param_dict[new_pname] = ckpt_param_dict.pop(pname)

    return ckpt_param_dict


def load_motion_modules(
    unet, motion_module_path, motion_lora_config=None, add_ldm_prefix=True, ldm_prefix="model.diffusion_model."
):
    # load motion module weights if use mm
    logger.info("Loading motion module from {}".format(motion_module_path))
    mm_state_dict = ms.load_checkpoint(motion_module_path)

    def _clear_insertion_from_training(param_name):
        return param_name.replace("diffusion_model.diffusion_model.", "diffusion_model.").replace("._backbone.", ".")

    # add prefix (used in the whole sd model) to param if needed
    mm_pnames = list(mm_state_dict.keys())
    for pname in mm_pnames:
        if add_ldm_prefix:
            if not pname.startswith(ldm_prefix):
                new_pname = ldm_prefix + pname
                # remove duplicated "diffusion_model" caused by saving mm only during training
                new_pname = _clear_insertion_from_training(new_pname)
                mm_state_dict[new_pname] = mm_state_dict.pop(pname)

    params_not_load, ckpt_not_load = load_param_into_net_with_filter(
        unet,
        mm_state_dict,
        filter=mm_state_dict.keys(),
    )
    if len(ckpt_not_load) > 0:
        logger.warning(
            "The following params in mm ckpt are not loaded into net: {}\nTotal: {}".format(
                ckpt_not_load, len(ckpt_not_load)
            )
        )
    assert len(ckpt_not_load) == 0, "All params in motion module must be loaded"

    # motion lora
    if motion_lora_config is not None:
        if motion_lora_config["path"] not in ["", None]:
            _mlora_path, alpha = motion_lora_config["path"], motion_lora_config["alpha"]
            logger.info("Loading motion lora from {}".format(_mlora_path))
            unet = merge_motion_lora_to_unet(unet, _mlora_path, alpha)

    return unet


def build_model_from_config(config, ckpt: str, is_training=False, use_motion_module=True):
    def _load_model(_model, checkpoint, verbose=True, ignore_net_param_not_load_warning=False):
        if isinstance(checkpoint, str):
            if os.path.exists(checkpoint):
                param_dict = ms.load_checkpoint(checkpoint)
            else:
                raise FileNotFoundError(f"{checkpoint} doesn't exist")
        elif isinstance(checkpoint, dict):
            param_dict = checkpoint
        else:
            raise TypeError(f"unknown checkpoint type: {checkpoint}")

        if param_dict:
            if ignore_net_param_not_load_warning:
                filter = param_dict.keys()
            else:
                filter = None
            param_not_load, ckpt_not_load = load_param_into_net_with_filter(_model, param_dict, filter=filter)
            assert (
                len(ckpt_not_load) == 0
            ), f"All params in SD checkpoint must be loaded. but got these not loaded {ckpt_not_load}"
            if verbose:
                if len(param_not_load) > 0:
                    logger.info(
                        "Net params not loaded: {}".format([p for p in param_not_load if not p.startswith("adam")])
                    )

    model = instantiate_from_config(config.model)
    if ckpt != "":
        param_dict = ms.load_checkpoint(ckpt)

        # update param dict loading unet2d checkpoint to unet3d
        if use_motion_module:
            if config.model.params.unet_config.params.motion_module_mid_block:
                unet3d_type = "adv2"
            else:
                unet3d_type = "adv1"
            param_dict = update_unet2d_params_for_unet3d(param_dict, unet3d_type=unet3d_type)

        logger.info(f"Loading main model from {ckpt}")
        _load_model(model, param_dict, ignore_net_param_not_load_warning=True)
    else:
        logger.warning("No pretarined weights loaded. Input checkpoint path is empty.")

    if not is_training:
        model.set_train(False)
        for param in model.trainable_params():
            param.requires_grad = False

    return model
