import yaml
import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Inference Config", add_help=False)
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="",
        required=True,
        help="YAML config file specifying default arguments (default=" ")",
    )
    parser.add_argument(
        "-o",
        "--opt",
        nargs="+",
        help="Options to change yaml configuration values, "
        "e.g. `-o system.distribute=False eval.dataset.dataset_root=/my_path/to/ocr_data`",
    )
    parser.add_argument("--device_id", help="device_id", type=int, default=0)
    parser.add_argument(
        "--driven_audio", default='./examples/driven_audio/bus_chinese.wav', help="path to driven audio")
    parser.add_argument(
        "--source_image", default='./examples/source_image/full_body_1.png', help="path to source image")
    parser.add_argument("--ref_eyeblink", default=None,
                        help="path to reference video providing eye blinking")
    parser.add_argument("--ref_pose", default=None,
                        help="path to reference video providing pose")

    parser.add_argument("--result_dir", default='./results',
                        help="path to output")
    parser.add_argument("--pose_style", type=int, default=0,
                        help="input pose style from [0, 46)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="the batch size of facerender")
    parser.add_argument("--size", type=int, default=256,
                        help="the image size of the facerender")
    parser.add_argument("--expression_scale", type=float,
                        default=1.,  help="the batch size of facerender")
    parser.add_argument('--input_yaw', nargs='+', type=int,
                        default=None, help="the input yaw degree of the user ")
    parser.add_argument('--input_pitch', nargs='+', type=int,
                        default=None, help="the input pitch degree of the user")
    parser.add_argument('--input_roll', nargs='+', type=int,
                        default=None, help="the input roll degree of the user")
    parser.add_argument('--enhancer',  type=str, default=None,
                        help="Face enhancer, [gfpgan, RestoreFormer]")
    parser.add_argument('--background_enhancer',  type=str,
                        default=None, help="background enhancer, [realesrgan]")
    parser.add_argument("--still", action="store_true",
                        help="can crop back to the original videos for the full body aniamtion")
    parser.add_argument("--preprocess", default='crop', choices=[
                        'crop', 'extcrop', 'resize', 'full', 'extfull'], help="how to preprocess the images")
    parser.add_argument("--verbose", action="store_true",
                        help="saving the intermedia output or not")

    # net structure and parameters
    parser.add_argument('--net_recon', type=str, default='resnet50',
                        choices=['resnet18', 'resnet34', 'resnet50'], help='useless')
    parser.add_argument('--use_last_fc', default=False,
                        help='zero initialize the last fc')
    parser.add_argument('--bfm_folder', type=str,
                        default='./checkpoints/BFM_Fitting/')
    parser.add_argument('--bfm_model', type=str,
                        default='BFM_model_front.mat', help='bfm model')

    # default renderer parameters
    parser.add_argument('--focal', type=float, default=1015.)
    parser.add_argument('--center', type=float, default=112.)
    parser.add_argument('--camera_d', type=float, default=10.)
    parser.add_argument('--z_near', type=float, default=5.)
    parser.add_argument('--z_far', type=float, default=15.)

    return parser


def _parse_options(opts: list):
    """
    Args:
        opt: list of str, each str in form f"{key}={value}"
    """
    options = {}
    if not opts:
        return options
    for opt_str in opts:
        assert (
            "=" in opt_str
        ), "Invalid option {}. A valid option must be in the format of {{key_name}}={{value}}".format(opt_str)
        k, v = opt_str.strip().split("=")
        options[k] = yaml.load(v, Loader=yaml.Loader)
    # print('Parsed options: ', options)

    return options


def _merge_options(config, options):
    """
    Merge options (from CLI) to yaml config.
    """
    for opt in options:
        value = options[opt]

        # parse hierarchical key in option, e.g. eval.dataset.dataset_root
        hier_keys = opt.split(".")
        assert hier_keys[0] in config, f"Invalid option {opt}. The key {hier_keys[0]} is not in config."
        cur = config[hier_keys[0]]
        for level, key in enumerate(hier_keys[1:]):
            if level == len(hier_keys) - 2:
                assert key in cur, f"Invalid option {opt}. The key {key} is not in config."
                cur[key] = value
            else:
                assert key in cur, f"Invalid option {opt}. The key {key} is not in config."
                cur = cur[key]  # go to next level

    return config


def parse_args_and_config():
    """
    Return:
        args: command line argments
        cfg: train/eval config dict
    """
    parser = create_parser()
    args = parser.parse_args()  # CLI args

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        # TODO: check validity of config arguments to avoid invalid config caused by typo.
        # _check_cfgs_in_parser(cfg, parser)
        # parser.set_defaults(**cfg)
        # parser.set_defaults(config=args_config.config)

    if args.opt:
        options = _parse_options(args.opt)
        cfg = _merge_options(cfg, options)

    return args, cfg
