import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def create_parser():
    parser = argparse.ArgumentParser(description="Inference Config Args")
    # csv path
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    # OCR options - compute total # of boxes, max single box percentage, total text area percentage
    parser.add_argument("--num_boxes", action="store_true", help="Compute and store the total number of boxes")
    parser.add_argument(
        "--max_single_percentage",
        action="store_true",
        help="Compute and store the maximum single text box area percentage",
    )
    parser.add_argument(
        "--total_text_percentage", action="store_true", help="Compute and store the total text area percentage"
    )
    # batch size
    parser.add_argument("--bs", type=int, default=1, help="Batch size")  # larger batch size meaningless
    # skip ocr if the file already exists
    parser.add_argument("--skip_if_existing", action="store_true")
    # params for prediction engine
    parser.add_argument("--mode", type=int, default=0, help="0 for graph mode, 1 for pynative mode ")  # added
    # parser.add_argument("--page_num", type=int, default=0)
    parser.add_argument(
        "--det_algorithm",
        type=str,
        default="DB++",
        choices=["DB", "DB++", "DB_MV3", "DB_PPOCRv3", "PSE"],
        help="detection algorithm.",
    )  # determine the network architecture
    parser.add_argument(
        "--det_amp_level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2", "O3"],
        help="Auto Mixed Precision level. This setting only works on GPU and Ascend",
    )
    parser.add_argument(
        "--det_model_dir",
        type=str,
        default="pretrained_models/dbnetpp.ckpt",
        help="directory containing the detection model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )  # determine the network weights
    parser.add_argument(
        "--det_limit_side_len", type=int, default=960, help="side length limitation for image resizing"
    )  # increase if need
    parser.add_argument(
        "--det_limit_type",
        type=str,
        default="max",
        choices=["min", "max"],
        help="limitation type for image resize. If min, images will be resized by limiting the minimum side length "
        "to `limit_side_len` (prior to accuracy). If max, images will be resized by limiting the maximum side "
        "length to `limit_side_len` (prior to speed). Default: max",
    )
    # TODO: currently only support quad, poly may lead to undefined error
    parser.add_argument(
        "--det_box_type",
        type=str,
        default="quad",
        choices=["quad"],
        help="box type for text region representation",
    )

    # DB parmas
    parser.add_argument("--det_db_thresh", type=float, default=0.3)
    parser.add_argument("--det_db_box_thresh", type=float, default=0.6)
    parser.add_argument("--det_db_unclip_ratio", type=float, default=1.5)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--use_dilation", type=str2bool, default=False)
    parser.add_argument("--det_db_score_mode", type=str, default="fast")

    # params for text recognizer
    parser.add_argument(
        "--rec_algorithm",
        type=str,
        default="CRNN",
        choices=["CRNN", "RARE", "CRNN_CH", "RARE_CH", "SVTR", "SVTR_PPOCRv3_CH"],
        help="recognition algorithm",
    )
    parser.add_argument(
        "--rec_amp_level",
        type=str,
        default="O0",
        choices=["O0", "O1", "O2", "O3"],
        help="Auto Mixed Precision level. This setting only works on GPU and Ascend",
    )
    parser.add_argument(
        "--rec_model_dir",
        type=str,
        help="directory containing the recognition model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )  # determine the network weights
    parser.add_argument(
        "--rec_image_shape",
        type=str,
        default="3, 32, 320",
        help="C, H, W for target image shape. max_wh_ratio=W/H will be used to control the maximum width after "
        '"aspect-ratio-kept" resizing. Set W larger for longer text.',
    )

    parser.add_argument(
        "--rec_batch_mode",
        type=str2bool,
        default=True,
        help="Whether to run recognition inference in batch-mode, which is faster but may degrade the accuracy "
        "due to padding or resizing to the same shape.",
    )  # added
    parser.add_argument("--rec_batch_num", type=int, default=8)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument(
        "--rec_char_dict_path",
        type=str,
        default=None,
        help="path to character dictionary. If None, will pick according to rec_algorithm and red_model_dir.",
    )
    # uncomment it after model trained supporting space recognition.
    # parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument("--vis_font_path", type=str, default="docs/fonts/simfang.ttf")
    parser.add_argument("--drop_score", type=float, default=0.5)

    parser.add_argument(
        "--draw_img_save_dir",
        type=str,
        default="./inference_results",
        help="Dir to save visualization and detection/recogintion/system prediction results",
    )
    parser.add_argument(
        "--save_crop_res",
        type=str2bool,
        default=False,
        help="Whether to save images cropped from text detection results.",
    )
    parser.add_argument(
        "--crop_res_save_dir", type=str, default="./output", help="Dir to save the cropped images for text boxes"
    )

    parser.add_argument("--warmup", type=str2bool, default=False)
    parser.add_argument("--ocr_result_dir", type=str, default=None, help="path or directory of ocr results")
    parser.add_argument(
        "--ser_algorithm",
        type=str,
        default="VI_LAYOUTXLM",
        choices=["VI_LAYOUTXLM", "LAYOUTXLM"],
        help="ser algorithm",
    )
    parser.add_argument(
        "--ser_model_dir",
        type=str,
        help="directory containing the ser model checkpoint best.ckpt, or path to a specific checkpoint file.",
    )
    parser.add_argument(
        "--kie_batch_mode",
        type=str2bool,
        default=True,
        help="Whether to run recognition inference in batch-mode, which is faster but may degrade the accuracy "
        "due to padding or resizing to the same shape.",
    )
    parser.add_argument("--kie_batch_num", type=int, default=8)

    return parser


def parse_args():
    parser = create_parser()
    args = parser.parse_args()
    return args
