import argparse

import numpy as np
from mg.models.tae.sd3_vae import SD3d5_CONFIG, SD3d5_VAE
from safetensors import safe_open

import mindspore as ms


def get_shape_from_str(shape):
    shape = shape.replace("(", "").replace(")", "").split(",")
    shape = [int(s) for s in shape if len(s) > 0]

    return shape


def get_pname_shape(ckpt_path):
    with safe_open(ckpt_path, framework="pt", device="cpu") as fp:
        for key in fp.keys():
            val = fp.get_tensor(key)
            shape = tuple(val.shape)
            dtype = val.dtype
            print(f"{key}#{shape}#{dtype}")


def load_torch_ckpt(ckpt_path):
    pt_state_dict = {}
    with safe_open(ckpt_path, framework="pt", device="cpu") as fp:
        for key in fp.keys():
            pt_state_dict[key] = fp.get_tensor(key)
            # print(key)
    return pt_state_dict


def plot_ms_vae2d5():
    tae = SD3d5_VAE(config=SD3d5_CONFIG)

    sd = tae.parameters_dict()
    pnames = list(sd.keys())
    for pname in pnames:
        shape = tuple(sd[pname].shape)
        print(f"{pname}#{shape}")


def convert_vae2d(source_fp, target_fp, target_model="vae2d"):
    # read param mapping files
    ms_pnames_file = "tools/ms_pnames_sd3.5_vae.txt" if target_model == "vae2d" else "tools/ms_pnames_tae_vae.txt"
    print("target ms pnames is annotated in ", ms_pnames_file)
    with open(ms_pnames_file) as file_ms:
        lines_ms = list(file_ms.readlines())
    with open("tools/pt_pnames_sd3.5_vae.txt") as file_pt:
        lines_pt = list(file_pt.readlines())

    # if "from_vae2d":
    # lines_ms = [line for line in lines_ms if line.startswith("spatial_vae")]
    # lines_pt = [line for line in lines_pt if line.startswith("spatial_vae")]

    assert len(lines_ms) == len(lines_pt)

    # convert and save
    sd_pt = load_torch_ckpt(source_fp)  # state dict
    num_params_pt = len(list(sd_pt.keys()))
    print("Total params in pt ckpt: ", num_params_pt)
    target_data = []
    for i in range(len(lines_pt)):
        name_pt, shape_pt = lines_pt[i].strip().split("#")
        shape_pt = get_shape_from_str(shape_pt)
        name_ms, shape_ms = lines_ms[i].strip().split("#")
        shape_ms = get_shape_from_str(shape_ms)
        assert np.prod(shape_pt) == np.prod(
            shape_ms
        ), f"Mismatch param: PT: {name_pt}, {shape_pt} vs MS: {name_ms}, {shape_ms}"

        # if "from_vae2d":
        #    name_pt = name_pt.replace("spatial_vae.module.", "")
        # param can be saved in bf16
        data = sd_pt[name_pt].cpu().detach().float().numpy().reshape(shape_ms)

        data = ms.Tensor(input_data=data.astype(np.float32), dtype=ms.float32)
        target_data.append({"name": name_ms, "data": data})  # ms.Tensor(data, dtype=ms.float32)})

    print("Total params converted: ", len(target_data))
    ms.save_checkpoint(target_data, target_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        "-s",
        type=str,
        help="path to vae torch checkpoint",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="models/tae_vae2d.ckpt",
        help="Filename to save. Specify folder, e.g., ./models, or file path which ends with .ckpt, e.g., ./models/vae.ckpt",
    )
    args = parser.parse_args()

    # ckpt_path = "/Users/Samit/Downloads/sd3.5_vae/diffusion_pytorch_model.safetensors"
    # get_pname_shape(ckpt_path)
    # plot_ms_vae2d5()

    # convert_vae2d(ckpt_path, "models/sd3.5_vae.ckpt")
    convert_vae2d(args.src, args.target, target_model="tae")
