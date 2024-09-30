import os, sys
import pickle
import mindspore as ms
from mindspore import context
from omegaconf import OmegaConf

context.set_context(mode=1, device_target="CPU")
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from utils.utils import instantiate_from_config


def param_convert(ms_params, pt_params, ckpt_path, extra_dict=None):
    # 参数名映射字典
    bn_ms2pt = {"gamma": "weight",
                "beta": "bias",
                "moving_mean": "running_mean",
                "moving_variance": "running_var"}

    if extra_dict:
        bn_ms2pt.update(extra_dict)

    new_params_list = []
    for ms_param in ms_params:
        # 在参数列表中，只有包含bn和downsample.1的参数是BatchNorm算子的参数

        # if "conv_block.1." in ms_param.name:
        # if any(x in ms_param.name and "mlp_gamma" not in ms_param.name and "mlp_beta" not in ms_param.name for x in bn_ms2pt.keys()):
        if True:
            # ms_param_item = ms_param.name.split(".")
            # pt_param_item = ms_param_item[:-1] + [bn_ms2pt[ms_param_item[-1]]]
            # pt_param = ".".join(pt_param_item)

            param_name = ms_param.name
            for k, v in bn_ms2pt.items():
                param_name = param_name.replace(k, v)
            pt_param = param_name

            # 如找到参数对应且shape一致，加入到参数列表
            if pt_param in pt_params and pt_params[pt_param].shape == ms_param.data.shape:
                ms_value = pt_params[pt_param]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32)})
            elif pt_param in pt_params and "weight" in ms_param.name:
                ms_value = pt_params[pt_param]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32).unsqueeze(2)})
            else:
                print(ms_param.name, "not match in pt_params")
        # 其他参数
        else:
            # 如找到参数对应且shape一致，加入到参数列表
            if ms_param.name in pt_params and tuple(pt_params[ms_param.name].shape) == tuple(ms_param.data.shape):
                ms_value = pt_params[ms_param.name]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32)})
            # elif ms_param.name in pt_params and ("weight_u" in ms_param.name or "weight_v" in ms_param.name):
            elif ms_param.name in pt_params and "weight" in ms_param.name:
                ms_value = pt_params[ms_param.name]
                new_params_list.append(
                    {"name": ms_param.name, "data": ms.Tensor(ms_value, ms.float32).unsqueeze(2)})
            else:
                print(ms_param.name, "not match in pt_params")
    # 保存成MindSpore的checkpoint
    ms.save_checkpoint(new_params_list, ckpt_path)


def convert_t2v():
    config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)

    extra_dict = {
        "attn.in_proj.weight": "attn.in_proj_weight",
        "attn.in_proj.bias": "attn.in_proj_bias",
        "token_embedding.embedding_table": "token_embedding.weight"
    }

    ms_params = pretrained_t2v.get_parameters()

    with open("/home/mindone/katekong/src-repos/t2v-turbo/t2v_vc2_np.pkl", "rb") as f:
        state_dict = pickle.load(f)

    param_convert(ms_params, state_dict, "checkpoints/t2v_VC2.ckpt", extra_dict)


if __name__ == "__main__":
    convert_t2v()