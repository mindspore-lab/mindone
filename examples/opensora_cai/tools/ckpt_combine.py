import argparse
import os

import mindspore as ms
import mindspore.ops as ops


def get_parser():
    parser = argparse.ArgumentParser(description="combine distributed saved ckpts into one")
    parser.add_argument("--ori_ckpt_path", type=str, help="path of original pretrained integrated checkpoint")
    parser.add_argument(
        "--distri_ckpt_dir", type=str, help="directory of distributed saved checkpoints from optimizer parallel"
    )
    parser.add_argument("--epoch_num", type=int, help="target epoch of checkpoints want to use")
    # parser.add_argument("--step_num", type=int, help="target number of step want to use")
    parser.add_argument("--combined_ckpt_path", type=str, help="the path to save new combined checkpoint")

    return parser


def ckpt_backbone_remove(distri_ckpt_dir, epoch_num):
    ckpts_dict = {}
    for root, dirs, files in os.walk(distri_ckpt_dir):
        device_dir = root.split("_")[-1]
        for file in files:
            ckpt_name = "STDiT-e{}.ckpt".format(epoch_num)
            if file == ckpt_name:
                print("remove _backbone of {}".format(device_dir))
                ckpt_path = os.path.join(root, ckpt_name)
                old_ckpt = ms.load_checkpoint(ckpt_path)
                new_ckpt = {}
                for k in old_ckpt:
                    if "._backbone" in k:
                        _index = k.find(".backbone")
                        new_k = k[:_index] + k[_index + len("._backbone") :]
                    else:
                        new_k = k[:]
                    new_ckpt[new_k] = old_ckpt[k]
                ckpts_dict[device_dir] = new_ckpt

    return ckpts_dict


def combine(args):
    print("Start to remove unused _backbone keys...........")
    ckpts_dict = ckpt_backbone_remove(args.distri_ckpt_dir, args.epoch_num)
    ori_ckpt = ms.load_checkpoint(args.ori_ckpt_path)
    distri_first_ckpt = ckpts_dict["0"]
    rank_size = len(ckpts_dict.keys())

    print("Start to combine distributed checkpoints, it may take a bit long time.........")
    new_ckpt_data = {}
    for key, param in ori_ckpt.items():
        # import pdb; pdb.set_trace()
        if ("pos_embed" not in key) and (not key.startswith("network.")):
            key = "network." + key
        if key in distri_first_ckpt.keys():
            if param.value().shape:
                if param.value().shape[0] == (rank_size * distri_first_ckpt[key].value().shape[0]):
                    for i in range(rank_size):
                        rank_name = "{}".format(i)
                        distri_ckpt = ckpts_dict[rank_name]
                        if key not in new_ckpt_data:
                            new_ckpt_data[key] = distri_ckpt[key].value()
                        else:
                            param_data = new_ckpt_data[key].value()
                            param_data = ops.concat((param_data, distri_ckpt[key].value()))
                            new_ckpt_data[key] = param_data
                elif "scale_shift_table" in key:
                    # for scale_shift_table [6, 1152], the first card [3, 1152] and last card [3, 1152]
                    for i in range(rank_size):
                        rank_name = "{}".format(i)
                        distri_ckpt = ckpts_dict[rank_name]
                        # import pdb; pdb.set_trace()
                        assert (
                            distri_ckpt[key].shape[0] == param.shape[0] // 2
                        ), f"Get {key}, {distri_ckpt[key].shape}, rank {i}"
                        if key not in new_ckpt_data:
                            new_ckpt_data[key] = distri_ckpt[key].value()
                        elif i == (rank_size - 1):
                            param_data = new_ckpt_data[key].value()
                            param_data = ops.concat((param_data, distri_ckpt[key].value()))
                            new_ckpt_data[key] = param_data
                elif param.value().shape[0] == distri_first_ckpt[key].value().shape[0]:
                    new_ckpt_data[key] = distri_first_ckpt[key].value()
                else:
                    raise ValueError("{} shape is not right! ".format(key))

            else:
                new_ckpt_data[key] = distri_first_ckpt[key].value()

        else:
            new_ckpt_data[key] = ori_ckpt[key].value()

    new_ckpt = {}
    for key, param_data in new_ckpt_data.items():
        new_ckpt[key] = ori_ckpt[key].copy()
        new_ckpt[key].set_data(param_data)

    ms.save_checkpoint(new_ckpt, args.combined_ckpt_path)
    print("combine checkpoints successfully!")


if __name__ == "__main__":
    # ckpts_dict = ckpt_backbone_remove("/home/lty/seq/mindone/examples/opensora_cai/outputs/seq_para_formal_64", 850, 1)
    # #print(ckpts_dict)
    # distri_first_ckpt = ckpts_dict["0"]
    # print(distri_first_ckpt)
    parser = get_parser()
    args, _ = parser.parse_known_args()
    combine(args)
