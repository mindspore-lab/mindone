import argparse
import copy
import json
import logging
import os

import yaml

_logger = logging.getLogger(__name__)


def str2bool(b):
    if b.lower() not in ["false", "true"]:
        raise Exception("Invalid Bool Value")
    if b.lower() in ["false"]:
        return False
    return True


class Config(object):
    def __init__(self, load=True, cfg_dict=None, cfg_level=None):
        self._level = "cfg" + ("." + cfg_level if cfg_level is not None else "")
        if load:
            self.args = self._parse_args()
            _logger.info("Loading config from {}.".format(self.args.cfg_file))
            cfg_dict = self._load_yaml(self.args)
            cfg_dict = self._update_from_args(cfg_dict)
            self.cfg_dict = cfg_dict
        self._update_dict(cfg_dict)

    def _parse_args(self):
        parser = argparse.ArgumentParser(description="Argparser for configuring vidcomposer codebase")
        parser.add_argument(
            "-c",
            "--cfg",
            dest="cfg_file",
            help="Path to the configuration file",
            default="configs/train_exp02_motion_transfer.yaml",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=8888,
            help="Need to explore for different videos",
        )
        parser.add_argument(
            "--input_video",
            default="",
            help="input video for full task, or motion vector of input videos",
            type=str,
        ),
        parser.add_argument(
            "--image_path",
            default="",
            help="Single Image Input",
            type=str,
        )
        parser.add_argument(
            "--sketch_path",
            default="",
            help="Single Sketch Input",
            type=str,
        )
        parser.add_argument(
            "--style_image",
            default="",
            help="Style image input",
            type=str,
        )
        parser.add_argument(
            "--input_text_desc",
            default="",
            type=str,
        ),
        parser.add_argument(
            "--sample_scheduler",
            default="DDIM",
            choices=["DDIM", "DDPM", "PLMS"],
            help="Schduler method for using for inference. ",
        )
        parser.add_argument("--sample_steps", type=int, default=50, help="Sampling Step.")
        parser.add_argument(
            "--n_iter",
            type=int,
            default=4,
            help="number of iterations or trials. sample this often, ",
        )
        parser.add_argument(
            "--save_frames",
            action="store_true",
            help="save video frames",
        )
        parser.add_argument("--guidance_scale", type=float, default=9.0, help="The guidance scale value in inference.")
        parser.add_argument(
            "opts",
            help="other configurations",
            default=None,
            nargs=argparse.REMAINDER,
        )
        # new args for mindspore
        parser.add_argument(
            "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
        )
        parser.add_argument("--use_parallel", default=False, type=str2bool, help="use parallel")
        parser.add_argument(
            "--dataset_sink_mode",
            type=str2bool,
            help="use dataset_sink_mode in model.train. Enable it can boost the performance but step_end callback will be disabled.",
        )
        parser.add_argument(
            "--step_mode",
            default=None,
            type=str2bool,
            help="If True, checkpoints will be save in every `ckpt_save_interval` steps, which is useful when the training steps in \
                a epoch is extremely large. Otherwise, checkpoint will be save in every `ckpt_save_inteveral` epochs. Default: False",
        )
        parser.add_argument(
            "--use_recompute",
            type=str2bool,
            help="use recompute in UNet. Enable it can slow down the speed but save some memory.",
        )
        parser.add_argument("--profile", default=False, type=str2bool, help="Profile or not")
        parser.add_argument(
            "--resume_checkpoint",
            default=None,
            type=str,
            help="unet checkpoint path. If not None, it will overwrite the checkpiont path in yaml file config.",
        )
        parser.add_argument(
            "--output_dir", default="outputs/train", type=str, help="output directory to save training results"
        )
        return parser.parse_args()

    def _path_join(self, path_list):
        path = ""
        for p in path_list:
            path += p + "/"
        return path[:-1]

    def _update_from_args(self, cfg_dict):
        args = self.args
        for var in vars(args):
            if getattr(args, var) is not None:
                cfg_dict[var] = getattr(args, var)  # overwrite the key argument if provided by the command
        return cfg_dict

    def _load_yaml(self, args, file_name=""):
        assert args.cfg_file is not None
        if not file_name == "":  # reading from base file
            with open(file_name, "r") as f:
                cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
        else:
            if os.getcwd().split("/")[-1] == args.cfg_file.split("/")[0]:
                args.cfg_file = args.cfg_file.replace(os.getcwd().split("/")[-1], "./")
            try:
                with open(args.cfg_file, "r") as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
                    file_name = args.cfg_file
            except:  # noqa
                args.cfg_file = os.path.realpath(__file__).split("/")[-3] + "/" + args.cfg_file
                with open(args.cfg_file, "r") as f:
                    cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)
                    file_name = args.cfg_file

        if "_BASE_RUN" not in cfg.keys() and "_BASE_MODEL" not in cfg.keys() and "_BASE" not in cfg.keys():
            # return cfg if the base file is being accessed
            return cfg

        if "_BASE" in cfg.keys():
            if cfg["_BASE"][1] == ".":
                prev_count = cfg["_BASE"].count("..")
                cfg_base_file = self._path_join(
                    file_name.split("/")[: (-1 - cfg["_BASE"].count(".."))] + cfg["_BASE"].split("/")[prev_count:]
                )
            else:
                cfg_base_file = cfg["_BASE"].replace(
                    "./",
                    args.cfg_file.replace(args.cfg_file.split("/")[-1], ""),
                )
            cfg_base = self._load_yaml(args, cfg_base_file)
        else:
            if "_BASE_RUN" in cfg.keys():
                if cfg["_BASE_RUN"][1] == ".":
                    prev_count = cfg["_BASE_RUN"].count("..")
                    cfg_base_file = self._path_join(
                        file_name.split("/")[: (-1 - prev_count)] + cfg["_BASE_RUN"].split("/")[prev_count:]
                    )
                else:
                    cfg_base_file = cfg["_BASE_RUN"].replace(
                        "./",
                        args.cfg_file.replace(args.cfg_file.split("/")[-1], ""),
                    )
                cfg_base = self._load_yaml(args, cfg_base_file)
            if "_BASE_MODEL" in cfg.keys():
                if cfg["_BASE_MODEL"][1] == ".":
                    prev_count = cfg["_BASE_MODEL"].count("..")
                    cfg_base_file = self._path_join(
                        file_name.split("/")[: (-1 - cfg["_BASE_MODEL"].count(".."))]
                        + cfg["_BASE_MODEL"].split("/")[prev_count:]
                    )
                else:
                    cfg_base_file = cfg["_BASE_MODEL"].replace(
                        "./",
                        args.cfg_file.replace(args.cfg_file.split("/")[-1], ""),
                    )
                cfg_base = self._load_yaml(args, cfg_base_file)
                cfg = self._merge_cfg_from_base(cfg_base, cfg)
        cfg = self._merge_cfg_from_command(args, cfg)
        return cfg

    def _merge_cfg_from_command(self, args, cfg):
        assert len(args.opts) % 2 == 0, "Override list {} has odd length: {}.".format(args.opts, len(args.opts))
        keys = args.opts[0::2]
        vals = args.opts[1::2]

        # maximum supported depth 3
        for idx, key in enumerate(keys):
            key_split = key.split(".")
            assert len(key_split) <= 4, "Key depth error. \nMaximum depth: 3\n Get depth: {}".format(len(key_split))
            assert key_split[0] in cfg.keys(), "Non-existent key: {}.".format(key_split[0])
            if len(key_split) == 2:
                assert key_split[1] in cfg[key_split[0]].keys(), "Non-existent key: {}.".format(key)
            elif len(key_split) == 3:
                assert key_split[1] in cfg[key_split[0]].keys(), "Non-existent key: {}.".format(key)
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), "Non-existent key: {}.".format(key)
            elif len(key_split) == 4:
                assert key_split[1] in cfg[key_split[0]].keys(), "Non-existent key: {}.".format(key)
                assert key_split[2] in cfg[key_split[0]][key_split[1]].keys(), "Non-existent key: {}.".format(key)
                assert (
                    key_split[3] in cfg[key_split[0]][key_split[1]][key_split[2]].keys()
                ), "Non-existent key: {}.".format(key)
            if len(key_split) == 1:
                cfg[key_split[0]] = vals[idx]
            elif len(key_split) == 2:
                cfg[key_split[0]][key_split[1]] = vals[idx]
            elif len(key_split) == 3:
                cfg[key_split[0]][key_split[1]][key_split[2]] = vals[idx]
            elif len(key_split) == 4:
                cfg[key_split[0]][key_split[1]][key_split[2]][key_split[3]] = vals[idx]
        return cfg

    def _update_dict(self, cfg_dict):
        def recur(key, elem):
            if type(elem) is dict:
                return key, Config(load=False, cfg_dict=elem, cfg_level=key)
            else:
                if type(elem) is str and elem[1:3] == "e-":
                    elem = float(elem)
                return key, elem

        dic = dict(recur(k, v) for k, v in cfg_dict.items())
        self.__dict__.update(dic)

    def get_args(self):
        return self.args

    def __repr__(self):
        return "{}\n".format(self.dump())

    def dump(self):
        return json.dumps(self.cfg_dict, indent=2)

    def deep_copy(self):
        return copy.deepcopy(self)


if __name__ == "__main__":
    config = Config(load=True)
    print(config.DATA)
