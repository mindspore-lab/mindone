import sys

sys.path.append(".")

import argparse
import copy

from omegaconf import OmegaConf
from train_text_to_image import get_obj_from_str

import mindspore as ms
from mindspore import load_param_into_net
from mindspore.train.serialization import _update_param


def build_model_from_config(config):
    config = OmegaConf.load(config).model
    if "target" not in config:
        if config == "__is_first_stage__":
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    config_params = config.get("params", dict())

    # Force to stop loading checkpoint
    config.pretrained_ckpt = None

    return get_obj_from_str(config["target"])(**config_params)


def read_ckpt_params(args):
    param_dict = ms.load_checkpoint(args.pretrained_model_path)
    for pn in param_dict:
        print(pn)


def convert(args):
    sd_controlnet = build_model_from_config(args.model_config)
    print("Load SD checkpoint from ", args.pretrained_model_path, "to ", sd_controlnet.__class__.__name__)

    # load sd main
    pretrained_weights = ms.load_checkpoint(args.pretrained_model_path)
    param_not_load, ckpt_not_load = load_param_into_net(sd_controlnet, pretrained_weights)
    print("Net params not load: {}".format(param_not_load))
    print("Checkpoint parm not load: {}".format(param_not_load))

    # copy sd encoder weights to controlnet
    # prior: controlnet param names start with "model.diffusion_model.controlnet", e.g. model.diffusion_model.controlnet.input_blocks.0.0.conv.weight
    # removing "controlnet." from it gives the original param name in sd
    # except for input_hint_block for control image encoding.
    net_params = sd_controlnet.get_parameters()
    # net_param_names = [x.name for x in net_params]
    for net_param in net_params:
        if "controlnet." in net_param.name:
            sd_param_name = net_param.name.replace("controlnet.", "")
            if sd_param_name in pretrained_weights:
                new_param = copy.deepcopy(pretrained_weights[sd_param_name])
                _update_param(net_param, new_param, strict_load=False)
                print(f"Copied {sd_param_name} -> {net_param.name}")
            else:
                print(
                    f"WARNING: {sd_param_name} not in preatrined_weights! Ignore this warning if the param belongs to input hint block or zero moduels."
                )

    save_fn = args.pretrained_model_path.replace(".ckpt", "_controlnet_init.ckpt")
    ms.save_checkpoint(sd_controlnet, save_fn)

    print("Finish! Checkpoint saved in : ", save_fn)

    return save_fn


def validate(controlnet_init_ckpt):
    param_dict = ms.load_checkpoint(args.pretrained_model_path)
    for pn in param_dict:
        if "controlnet." in pn:
            sd_pn = pn.replace("controlnet.", "")
            if sd_pn in param_dict:
                assert param_dict[pn].asnumpy().sum() == param_dict[sd_pn].asnumpy().sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config", default="configs/v1-train-cldm.yaml", type=str, help="sd with controlnet model config path"
    )
    parser.add_argument(
        "--pretrained_model_path",
        "-p",
        default="models/sd_v1.5-d0ab7146.ckpt",
        type=str,
        help="Specify the pretrained model from this checkpoint",
    )
    # parser.add_argument("--output_dir", "-o", default="models", type=str, help="dir to save the converted checkpoint")

    args = parser.parse_args()
    convert(args)
