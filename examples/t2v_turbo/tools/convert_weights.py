import os, sys
import argparse
import torch
import pickle
from omegaconf import OmegaConf

import mindspore as ms
from mindspore import context


context.set_context(mode=1, device_target="CPU")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from utils.utils import instantiate_from_config


def _load_torch_ckpt(ckpt_file):
    source_data = torch.load(ckpt_file, map_location="cpu")
    if "state_dict" in source_data:
        source_data = source_data["state_dict"]
    return source_data


def _load_huggingface_safetensor(ckpt_file):
    from safetensors import safe_open

    db_state_dict = {}
    with safe_open(ckpt_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            db_state_dict[key] = f.get_tensor(key)
    return db_state_dict


LOAD_PYTORCH_FUNCS = {"others": _load_torch_ckpt, "safetensors": _load_huggingface_safetensor}


def load_torch_ckpt(ckpt_path):
    extension = ckpt_path.split(".")[-1]
    if extension not in LOAD_PYTORCH_FUNCS.keys():
        extension = "others"
    torch_params = LOAD_PYTORCH_FUNCS[extension](ckpt_path)
    return torch_params


def param_convert(ms_params, pt_params, ckpt_path, extra_dict=None):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight", "beta": "bias", "moving_mean": "running_mean", "moving_variance": "running_var"}

    if extra_dict:
        bn_ms2pt.update(extra_dict)

    new_params_list = []
    for ms_param in ms_params:

        param_name = ms_param.name
        for k, v in bn_ms2pt.items():
            param_name = param_name.replace(k, v)
        pt_param = param_name

        # 如找到参数对应且shape一致，加入到参数列表
        if pt_param in pt_params and pt_params[pt_param].shape == ms_param.data.shape:
            ms_value = pt_params[pt_param].cpu().detach().numpy()
            new_params_list.append({"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32)})
        elif pt_param in pt_params and "weight" in ms_param.name:
            ms_value = pt_params[pt_param].cpu().detach().numpy()
            new_params_list.append({"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32).unsqueeze(2)})
        else:
            print(ms_param.name, "not match in pt_params")

    ms.save_checkpoint(new_params_list, ckpt_path)


def convert_t2v_vc2(src_path, target_path):
    config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)

    extra_dict = {
        "attn.in_proj.weight": "attn.in_proj_weight",
        "attn.in_proj.bias": "attn.in_proj_bias",
        "token_embedding.embedding_table": "token_embedding.weight",
    }

    ms_params = pretrained_t2v.get_parameters()

    state_dict = load_torch_ckpt(src_path)
    param_convert(ms_params, state_dict, target_path, extra_dict)


def convert_lora(src_path, target_path):
    lora_weights = load_torch_ckpt(src_path)
    if isinstance(lora_weights, dict):
        weights = {k: ms.Tensor(v.cpu().detach().numpy()) for k, v in lora_weights.items()}
    elif isinstance(lora_weights, list):
        weights = [ms.Tensor(v.cpu().detach().numpy()) for v in lora_weights]
    else:
        raise Exception("Unknown LORA weights format!")

    with open(target_path, "wb") as f:
        pickle.dump(weights, f)


def convert_internvid2(src_path, target_path):

    from intern_vid2.demo_config import Config, eval_dict_leaf
    from intern_vid2.demo_utils import InternVideo2_Stage2

    def _param_convert(ms_params, pt_params, ckpt_path, extra_dict=None):
        # 参数名映射字典
        bn_ms2pt = {"gamma": "weight", "beta": "bias", "moving_mean": "running_mean", "moving_variance": "running_var"}

        if extra_dict:
            bn_ms2pt.update(extra_dict)

        new_params_list = []
        for ms_param in ms_params:

            param_name = ms_param.name
            for k, v in bn_ms2pt.items():
                if "norm" in param_name:
                    param_name = param_name.replace(k, v)
            pt_param = param_name

            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_param.data.shape:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32)})
            elif pt_param in pt_params and "weight" in ms_param.name:
                ms_value = pt_params[pt_param]
                new_params_list.append({"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32).unsqueeze(2)})
            else:
                print(ms_param.name, "not match in pt_params")

        ms.save_checkpoint(new_params_list, ckpt_path)

    n_frames = 4
    config = Config.from_file("intern_vid2/configs/internvideo2_stage2_config.py")
    config = eval_dict_leaf(config)
    config["inputs"]["video_input"]["num_frames"] = n_frames
    config["inputs"]["video_input"]["num_frames_test"] = n_frames
    config["model"]["vision_encoder"]["num_frames"] = n_frames

    model = InternVideo2_Stage2(config=config, tokenizer=None, is_pretrain=False)
    ms_params = model.get_parameters()

    state_dict = load_torch_ckpt(src_path)
    _param_convert(ms_params, state_dict, target_path)


def convert_hpsv2(src_path, target_path):
    from lvdm.modules.encoders.clip import CLIPModel, parse, support_list

    extra_dict = {
        "attn.in_proj.weight": "attn.in_proj_weight",
        "attn.in_proj.bias": "attn.in_proj_bias",
        "token_embedding.embedding_table": "token_embedding.weight",
    }

    config_path = support_list["open_clip_vit_h_14"]
    config = parse(config_path, src_path)
    model = CLIPModel(config)
    ms_params = model.get_parameters()

    state_dict = load_torch_ckpt(src_path)
    param_convert(ms_params, state_dict, target_path, extra_dict)


def convert_weights(model_folder):
    # convert videocrafter2_model and lora to mindspore checkpoints
    fn_vc2 = os.path.join(model_folder, "VideoCrafter2_model.ckpt")
    fn_lora = os.path.join(model_folder, "unet_lora.pt")

    # output path
    out_vc2 = os.path.join(model_folder, "VideoCrafter2_model_ms.ckpt")
    out_lora = os.path.join(model_folder, "unet_lora.ckpt")

    ## convert videocrafter2
    print(f"converting the weights of {fn_vc2} ...")
    convert_t2v_vc2(fn_vc2, out_vc2)
    print(f"converted to {out_vc2}.")

    ## convert lora
    print(f"converting the weights of {fn_lora} ...")
    convert_lora(fn_lora, out_lora)
    print("converted to {out_lora}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert the PixArt-Sigma checkpoint.")

    parser.add_argument("-s", "--source", required=True, help="file path of the checkpoint (.pth / .safetensors)")
    parser.add_argument(
        "-t",
        "--target",
        help="output file path. If it is None, then the converted file will be saved in the input directory.",
    )
    parser.add_argument("-t", "--type", choices=["vc2", "lora", "hps", "internvid"], required=True, help="checkpoint/weights type.")

    args = parser.parse_args()

    if args.target is None:
        filename, suffix = os.path.splitext(args.source)
        target_path = filename + ".ckpt"
    else:
        target_path = args.target

    if os.path.exists(target_path):
        print(f"Warnings: {target_path} will be overwritten!")

    if args.type == "vc2":
        convert_t2v_vc2(args.source, target_path)
    elif args.type == "lora":
        convert_lora(args.source, target_path)
    elif args.type == "hps":
        convert_hpsv2(args.source, target_path)
    elif args.type == "internvid":
        convert_internvid2(args.source, target_path)
    else:
        raise Exception("Unknown checkpoint type!")

    print(f"Converted weight saved to {target_path}")
