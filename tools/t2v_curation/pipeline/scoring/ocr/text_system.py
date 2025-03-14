import glob
import logging
import os
from pathlib import Path
from time import time
from typing import List, Union

import cv2
import numpy as np
from mindocr import build_model, build_postprocess
from mindocr.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mindocr.data.transforms import create_transforms, run_transforms
from shapely.geometry import Polygon

import mindspore as ms
from mindspore import ops
from mindspore.common import dtype as mstype

logger = logging.getLogger("mindocr")


def get_image_paths(img_dir: str) -> List[str]:
    """
    Args:
        img_dir: path to an image or a directory containing multiple images.

    Returns:
        List: list of image paths in the directory and its subdirectories.
    """
    img_dir = Path(img_dir)
    assert img_dir.exists(), f"{img_dir} does NOT exist. Please check the directory / file path."

    extensions = [".jpg", ".png", ".jpeg"]
    if img_dir.is_file():
        img_paths = [str(img_dir)]
    else:
        img_paths = [str(file) for file in img_dir.rglob("*.*") if file.suffix.lower() in extensions]

    assert (
        len(img_paths) > 0
    ), f"{img_dir} does NOT exist, or no image files exist in {img_dir}. Please check the `image_dir` arg value."
    return sorted(img_paths)


def get_ckpt_file(ckpt_dir):
    if os.path.isfile(ckpt_dir):
        ckpt_load_path = ckpt_dir
    else:
        # ckpt_load_path = os.path.join(ckpt_dir, 'best.ckpt')
        ckpt_paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckpt")))
        assert len(ckpt_paths) != 0, f"No .ckpt files found in {ckpt_dir}"
        ckpt_load_path = ckpt_paths[0]
        if len(ckpt_paths) > 1:
            logger.warning(f"More than one .ckpt files found in {ckpt_dir}. Pick {ckpt_load_path}")

    return ckpt_load_path


def crop_text_region(img, points, box_type="quad", rotate_if_vertical=True):  # polygon_type='poly'):
    # box_type: quad or poly
    def crop_img_box(img, points, rotate_if_vertical=True):
        assert len(points) == 4, "shape of points must be [4, 2]"
        img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))
        dst_pts = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])

        trans_matrix = cv2.getPerspectiveTransform(points, dst_pts)
        dst_img = cv2.warpPerspective(
            img, trans_matrix, (img_crop_width, img_crop_height), borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
        )

        if rotate_if_vertical:
            h, w = dst_img.shape[0:2]
            if h / float(w) >= 1.5:
                dst_img = np.rot90(dst_img)

        return dst_img

    if box_type[:4] != "poly":
        return crop_img_box(img, points, rotate_if_vertical=rotate_if_vertical)
    else:  # polygons
        bounding_box = cv2.minAreaRect(np.array(points).astype(np.int32))
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_a, index_b, index_c, index_d = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_a = 0
            index_d = 1
        else:
            index_a = 1
            index_d = 0
        if points[3][1] > points[2][1]:
            index_b = 2
            index_c = 3
        else:
            index_b = 3
            index_c = 2

        box = [points[index_a], points[index_b], points[index_c], points[index_d]]
        crop_img = crop_img_box(img, np.array(box))
        return crop_img


algo_to_model_name = {
    "DB": "dbnet_resnet50",
    "DB++": "dbnetpp_resnet50",
    "DB_MV3": "dbnet_mobilenetv3",
    "DB_PPOCRv3": "dbnet_ppocrv3",
    "PSE": "psenet_resnet152",
    "CRNN": "crnn_resnet34",
    "RARE": "rare_resnet34",
    "CRNN_CH": "crnn_resnet34_ch",
    "RARE_CH": "rare_resnet34_ch",
    "SVTR": "svtr_tiny",
    "SVTR_PPOCRv3_CH": "svtr_ppocrv3_ch",
}


class Preprocessor(object):
    def __init__(self, task="det", algo="DB", **kwargs):
        # algo = algo.lower()
        if task == "det":
            limit_side_len = kwargs.get("det_limit_side_len", 736)
            limit_type = kwargs.get("det_limit_type", "min")

            pipeline = [
                {"DecodeImage": {"img_mode": "RGB", "keep_ori": True, "to_float32": False}},
                {
                    "DetResize": {
                        "target_size": None,  # [ 1152, 2048 ]
                        "keep_ratio": True,
                        "limit_side_len": limit_side_len,
                        "limit_type": limit_type,
                        "padding": False,
                        "force_divisable": True,
                    }
                },
                {
                    "NormalizeImage": {
                        "bgr_to_rgb": False,
                        "is_hwc": True,
                        "mean": IMAGENET_DEFAULT_MEAN,
                        "std": IMAGENET_DEFAULT_STD,
                    }
                },
                {"ToCHWImage": None},
            ]

            if algo == "DB_PPOCRv3":
                pipeline = [
                    {"DecodeImage": {"img_mode": "RGB", "keep_ori": True, "to_float32": False}},
                    {"DetResize": {"limit_type": "min", "limit_side_len": 736}},
                    {
                        "NormalizeImage": {
                            "bgr_to_rgb": True,
                            "is_hwc": True,
                            "mean": IMAGENET_DEFAULT_MEAN,
                            "std": IMAGENET_DEFAULT_STD,
                        }
                    },
                    {"ToCHWImage": None},
                ]

            logger.info(f"Pick optimal preprocess hyper-params for det algo {algo}:\n {pipeline[1]}")
            # TODO: modify the base pipeline for non-DBNet network if needed

        elif task == "rec":
            # defalut value if not claim in optim_hparam
            DEFAULT_PADDING = True
            DEFAULT_KEEP_RATIO = True
            # TODO: norm before padding is more reasonable but the previous models (trained before 2023.05.26)
            #  is based on norm in the end.
            DEFAULT_NORM_BEFORE_PAD = False

            # register optimal hparam for each model
            optimal_hparam = {
                # 'CRNN': dict(target_height=32, target_width=100, padding=True, keep_ratio=True, norm_before_pad=True),
                "CRNN": dict(target_height=32, target_width=100, padding=False, keep_ratio=False),
                "CRNN_CH": dict(target_height=32, taget_width=320, padding=True, keep_ratio=True),
                "RARE": dict(target_height=32, target_width=100, padding=False, keep_ratio=False),
                "RARE_CH": dict(target_height=32, target_width=320, padding=True, keep_ratio=True),
                "SVTR": dict(target_height=64, target_width=256, padding=False, keep_ratio=False),
                "SVTR_PPOCRv3_CH": dict(target_height=48, target_width=320, padding=True, keep_ratio=True),
            }

            # get hparam by combining default value, optimal value, and arg parser value. Prior: optimal value ->
            # parser value -> default value
            parsed_img_shape = kwargs.get("rec_image_shape", "3, 32, 320").split(",")
            parsed_height, parsed_width = int(parsed_img_shape[1]), int(parsed_img_shape[2])
            if algo in optimal_hparam:
                target_height = optimal_hparam[algo]["target_height"]
            else:
                target_height = parsed_height

            norm_before_pad = optimal_hparam[algo].get("norm_before_pad", DEFAULT_NORM_BEFORE_PAD)

            # TODO: update max_wh_ratio for each batch
            # max_wh_ratio = parsed_width /  float(parsed_height)
            # batch_num = kwargs.get('rec_batch_num', 1)
            batch_mode = kwargs.get("rec_batch_mode", False)  # and (batch_num > 1)
            if not batch_mode:
                # For single infer, the optimal choice is to resize the image to target height while keeping
                # aspect ratio, no padding. limit the max width.
                padding = False
                keep_ratio = True
                target_width = None
            else:
                # parse optimal hparam
                if algo in optimal_hparam:
                    padding = optimal_hparam[algo].get("padding", DEFAULT_PADDING)
                    keep_ratio = optimal_hparam[algo].get("keep_ratio", DEFAULT_KEEP_RATIO)
                    target_width = optimal_hparam[algo].get("target_width", parsed_width)
                else:
                    padding = DEFAULT_PADDING
                    keep_ratio = DEFAULT_KEEP_RATIO
                    target_width = parsed_width

            if (target_height != parsed_height) or (target_width != parsed_width):
                logger.warning(
                    f"`rec_image_shape` {parsed_img_shape[1:]} dose not meet the network input requirement or "
                    f"is not optimal, which should be [{target_height}, {target_width}] under batch mode = {batch_mode}"
                )

            logger.info(
                f"Pick optimal preprocess hyper-params for rec algo {algo}:\n"
                + "\n".join(
                    [
                        f"{k}:\t{str(v)}"
                        for k, v in dict(
                            target_height=target_height,
                            target_width=target_width,
                            padding=padding,
                            keep_ratio=keep_ratio,
                            norm_before_pad=norm_before_pad,
                        ).items()
                    ]
                )
            )

            pipeline = [
                {"DecodeImage": {"img_mode": "RGB", "keep_ori": True, "to_float32": False}},
                {
                    "RecResizeNormForInfer": {
                        "target_height": target_height,
                        "target_width": target_width,  # 100,
                        "keep_ratio": keep_ratio,
                        "padding": padding,
                        "norm_before_pad": norm_before_pad,
                    }
                },
                {"ToCHWImage": None},
            ]
        elif task == "ser":
            pipeline = [
                {"DecodeImage": {"img_mode": "RGB", "infer_mode": True, "to_float32": False}},
                {
                    "VQATokenLabelEncode": {
                        "contains_re": False,
                        "infer_mode": True,
                        "algorithm": "LayoutXLM",
                        "class_path": "mindocr/utils/dict/class_list_xfun.txt",
                        "order_method": "tb-yx",
                    }
                },
                {"VQATokenPad": {"max_seq_len": 512, "infer_mode": True, "return_attention_mask": True}},
                {"VQASerTokenChunk": {"infer_mode": True, "max_seq_len": 512}},
                {"LayoutResize": {"infer_mode": True, "size": [224, 224]}},
                {
                    "NormalizeImage": {
                        "infer_mode": True,
                        "bgr_to_rgb": False,
                        "is_hwc": True,
                        "mean": "imagenet",
                        "std": "imagenet",
                    }
                },
                {"ToCHWImage": None},
            ]

        self.pipeline = pipeline
        self.transforms = create_transforms(pipeline)

    # TODO: allow multiple image inputs and preprocess them with multi-thread
    def __call__(self, img_or_path):
        """
        Return:
            dict, preprocessed data containing keys:
                - image: np.array, transfomred image
                - image_ori: np.array, original image
                - shape: list of [ori_h, ori_w, scale_h, scale_w]
                and other keys added in transform pipeline.
        """
        if isinstance(img_or_path, str):
            data = {"img_path": img_or_path}
            output = run_transforms(data, self.transforms)
        elif isinstance(img_or_path, dict):
            output = run_transforms(img_or_path, self.transforms)
        else:
            data = {"image": img_or_path}
            data["image_ori"] = img_or_path.copy()  # TODO
            data["image_shape"] = img_or_path.shape
            output = run_transforms(data, self.transforms[1:])

        return output


class Postprocessor(object):
    def __init__(self, task="det", algo="DB", rec_char_dict_path=None, **kwargs):
        # algo = algo.lower()
        if task == "det":
            if algo.startswith("DB"):
                if algo == "DB_PPOCRv3":
                    postproc_cfg = dict(
                        name="DBPostprocess",
                        box_type="quad",
                        binary_thresh=0.3,
                        box_thresh=0.7,
                        max_candidates=1000,
                        expand_ratio=1.5,
                    )
                else:
                    postproc_cfg = dict(
                        name="DBPostprocess",
                        box_type="quad",
                        binary_thresh=0.3,
                        box_thresh=0.6,
                        max_candidates=1000,
                        expand_ratio=1.5,
                    )
            elif algo.startswith("PSE"):
                postproc_cfg = dict(
                    name="PSEPostprocess",
                    box_type="quad",
                    binary_thresh=0.0,
                    box_thresh=0.85,
                    min_area=16,
                    scale=1,
                )
            else:
                raise ValueError(f"No postprocess config defined for {algo}. Please check the algorithm name.")
            self.rescale_internally = True
            self.round = True
        elif task == "rec":
            rec_char_dict_path = (
                rec_char_dict_path or "mindocr/utils/dict/ch_dict.txt"
                if algo in ["CRNN_CH", "SVTR_PPOCRv3_CH"]
                else rec_char_dict_path
            )
            # TODO: update character_dict_path and use_space_char after CRNN trained using en_dict.txt released
            if algo.startswith("CRNN") or algo.startswith("SVTR"):
                # TODO: allow users to input char dict path
                if algo == "SVTR_PPOCRv3_CH":
                    postproc_cfg = dict(
                        name="CTCLabelDecode",
                        character_dict_path=rec_char_dict_path,
                        use_space_char=True,
                    )
                else:
                    postproc_cfg = dict(
                        name="RecCTCLabelDecode",
                        character_dict_path=rec_char_dict_path,
                        use_space_char=False,
                    )
            elif algo.startswith("RARE"):
                rec_char_dict_path = (
                    rec_char_dict_path or "mindocr/utils/dict/ch_dict.txt" if algo == "RARE_CH" else rec_char_dict_path
                )
                postproc_cfg = dict(
                    name="RecAttnLabelDecode",
                    character_dict_path=rec_char_dict_path,
                    use_space_char=False,
                )

            else:
                raise ValueError(f"No postprocess config defined for {algo}. Please check the algorithm name.")
        elif task == "ser":
            class_path = "mindocr/utils/dict/class_list_xfun.txt"
            postproc_cfg = dict(name="VQASerTokenLayoutLMPostProcess", class_path=class_path)

        postproc_cfg.update(kwargs)
        self.task = task
        self.postprocess = build_postprocess(postproc_cfg)

    def __call__(self, pred, data=None, **kwargs):
        """
        Args:
            pred: network prediction
            data: (optional)
                preprocessed data, dict, which contains key `shape`
                    - shape: its values are [ori_img_h, ori_img_w, scale_h, scale_w]. scale_h, scale_w are needed to
                      map the predicted polygons back to the orignal image shape.

        return:
            det_res: dict, elements:
                    - polys: shape [num_polys, num_points, 2], point coordinate definition: width (horizontal),
                      height(vertical)
        """

        if self.task == "det":
            if self.rescale_internally:
                shape_list = np.array(data["shape_list"], dtype="float32")
                shape_list = np.expand_dims(shape_list, axis=0)
            else:
                shape_list = None

            output = self.postprocess(pred, shape_list=shape_list)

            if isinstance(output, dict):
                polys = output["polys"][0]
                scores = output["scores"][0]
            else:
                polys, scores = output[0]

            if not self.rescale_internally:
                scale_h, scale_w = data["shape_list"][2:]
                if len(polys) > 0:
                    if not isinstance(polys, list):
                        polys[:, :, 0] = polys[:, :, 0] / scale_w
                        polys[:, :, 1] = polys[:, :, 1] / scale_h
                        if self.round:
                            polys = np.round(polys)
                    else:
                        for i, poly in enumerate(polys):
                            polys[i][:, 0] = polys[i][:, 0] / scale_w
                            polys[i][:, 1] = polys[i][:, 1] / scale_h
                            if self.round:
                                polys[i] = np.round(polys[i])

            det_res = dict(polys=polys, scores=scores)

            return det_res
        elif self.task == "rec":
            output = self.postprocess(pred)
            return output
        elif self.task == "ser":
            output = self.postprocess(
                pred, segment_offset_ids=kwargs.get("segment_offset_ids"), ocr_infos=kwargs.get("ocr_infos")
            )
            return output


def order_points_clockwise(points):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = points.sum(axis=1)
    rect[0] = points[np.argmin(s)]
    rect[2] = points[np.argmax(s)]
    tmp = np.delete(points, (np.argmin(s), np.argmax(s)), axis=0)
    diff = np.diff(np.array(tmp), axis=1)
    rect[1] = tmp[np.argmin(diff)]
    rect[3] = tmp[np.argmax(diff)]

    return rect


def validate_det_res(det_res, img_shape, order_clockwise=True, min_poly_points=3, min_area=3):
    polys = det_res["polys"].copy()
    scores = det_res.get("scores", [])

    if len(polys) == 0:
        return dict(polys=[], scores=[])

    h, w = img_shape[:2]
    # clip if ouf of image
    if not isinstance(polys, list):
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)
    else:
        for i, poly in enumerate(polys):
            polys[i][:, 0] = np.clip(polys[i][:, 0], 0, w - 1)
            polys[i][:, 1] = np.clip(polys[i][:, 1], 0, h - 1)

    new_polys = []
    if scores is not None:
        new_scores = []
    for i, poly in enumerate(polys):
        # refine points to clockwise order
        if order_clockwise:
            if len(poly) == 4:
                poly = order_points_clockwise(poly)
            else:
                logger.warning("order_clockwise only supports quadril polygons currently")
        # filter
        if len(poly) < min_poly_points:
            continue

        if min_area > 0:
            p = Polygon(poly)
            if p.is_valid and not p.is_empty:
                if p.area >= min_area:
                    poly_np = np.array(p.exterior.coords)[:-1, :]
                    new_polys.append(poly_np)
                    if scores is not None:
                        new_scores.append(scores[i])
        else:
            new_polys.append(poly)
            if scores is not None:
                new_scores.append(scores[i])

    if len(scores) > 0:
        new_det_res = dict(polys=np.array(new_polys, dtype=int), scores=new_scores)
    else:
        new_det_res = dict(polys=np.array(new_polys, dtype=int))

    # TODO: sort polygons from top to bottom, left to right

    return new_det_res


class TextDetector(object):
    def __init__(self, args):
        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.det_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
        assert args.det_algorithm in algo_to_model_name, (
            f"Invalid det_algorithm {args.det_algorithm}. "
            f"Supported detection algorithms are {list(algo_to_model_name.keys())}"
        )
        model_name = algo_to_model_name[args.det_algorithm]
        amp_level = args.det_amp_level
        if ms.get_context("device_target") == "GPU" and amp_level == "O3":
            logger.warning(
                "Detection model prediction does not support amp_level O3 on GPU currently. "
                "The program has switched to amp_level O2 automatically."
            )
            amp_level = "O2"
        self.model = build_model(
            model_name,
            pretrained=pretrained,
            pretrained_backbone=False,
            ckpt_load_path=ckpt_load_path,
            amp_level=amp_level,
        )
        self.model.set_train(False)
        logger.info(
            "Init detection model: {} --> {}. Model weights loaded from {}".format(
                args.det_algorithm, model_name, "pretrained url" if pretrained else ckpt_load_path
            )
        )

        # build preprocess and postprocess
        self.preprocess = Preprocessor(
            task="det",
            algo=args.det_algorithm,
            det_limit_side_len=args.det_limit_side_len,
            det_limit_type=args.det_limit_type,
        )

        self.postprocess = Postprocessor(task="det", algo=args.det_algorithm, box_type=args.det_box_type)

        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

        self.box_type = args.det_box_type
        self.visualize_preprocess = False

    def __call__(self, img_or_path, do_visualize=True):
        """
            Args:
        img_or_path: str for img path or np.array for RGB image
        do_visualize: visualize preprocess and final result and save them

            Return:
        det_res_final (dict): detection result with keys:
                            - polys: np.array in shape [num_polygons, 4, 2] if det_box_type is 'quad'. Otherwise,
                              it is a list of np.array, each np.array is the polygon points.
                            - scores: np.array in shape [num_polygons], confidence of each detected text box.
        data (dict): input and preprocessed data with keys: (for visualization and debug)
            - image_ori (np.ndarray): original image in shape [h, w, c]
            - image (np.ndarray): preprocessed image feed for network, in shape [c, h, w]
            - shape (list): shape and scaling information [ori_h, ori_w, scale_ratio_h, scale_ratio_w]
        """
        # preprocess
        data = self.preprocess(img_or_path)
        logger.info(f"Original image shape: {data['image_ori'].shape}")
        logger.info(f"After det preprocess: {data['image'].shape}")

        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)

        net_output = self.model(ms.Tensor(net_input))

        # postprocess
        det_res = self.postprocess(net_output, data)

        # validate: filter polygons with too small number of points or area
        det_res_final = validate_det_res(det_res, data["image_ori"].shape[:2], min_poly_points=3, min_area=3)

        return det_res_final, data


class TextRecognizer(object):
    def __init__(self, args):
        self.batch_num = args.rec_batch_num
        self.batch_mode = args.rec_batch_mode
        # self.batch_mode = args.rec_batch_mode and (self.batch_num > 1)
        logger.info(
            "recognize in {} mode {}".format(
                "batch" if self.batch_mode else "serial",
                "batch_size: " + str(self.batch_num) if self.batch_mode else "",
            )
        )

        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.rec_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
        assert args.rec_algorithm in algo_to_model_name, (
            f"Invalid rec_algorithm {args.rec_algorithm}. "
            f"Supported recognition algorithms are {list(algo_to_model_name.keys())}"
        )
        model_name = algo_to_model_name[args.rec_algorithm]

        amp_level = args.rec_amp_level
        if args.rec_algorithm.startswith("SVTR") and amp_level != "O2":
            logger.warning(
                "SVTR recognition model is optimized for amp_level O2. ampl_level for rec model is changed to O2"
            )
            amp_level = "O2"
        if ms.get_context("device_target") == "GPU" and amp_level == "O3":
            logger.warning(
                "Recognition model prediction does not support amp_level O3 on GPU currently. "
                "The program has switched to amp_level O2 automatically."
            )
            amp_level = "O2"
        self.model = build_model(model_name, pretrained=pretrained, ckpt_load_path=ckpt_load_path, amp_level=amp_level)
        self.model.set_train(False)

        self.cast_pred_fp32 = amp_level != "O0"
        if self.cast_pred_fp32:
            self.cast = ops.Cast()
        logger.info(
            "Init recognition model: {} --> {}. Model weights loaded from {}".format(
                args.rec_algorithm, model_name, "pretrained url" if pretrained else ckpt_load_path
            )
        )

        # build preprocess and postprocess
        # NOTE: most process hyper-params should be set optimally for the pick algo.
        self.preprocess = Preprocessor(
            task="rec",
            algo=args.rec_algorithm,
            rec_image_shape=args.rec_image_shape,
            rec_batch_mode=self.batch_mode,
            rec_batch_num=self.batch_num,
        )

        # TODO: try GeneratorDataset to wrap preprocess transform on batch for possible speed-up.
        #  if use_ms_dataset: ds = ms.dataset.GeneratorDataset(wrap_preprocess, ) in run_batchwise
        self.postprocess = Postprocessor(
            task="rec", algo=args.rec_algorithm, rec_char_dict_path=args.rec_char_dict_path
        )

        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

    def __call__(self, img_or_path_list: list, do_visualize=False):
        """
        Run text recognition serially for input images

        Args:
            img_or_path_list: list of str for img path or np.array for RGB image
            do_visualize: visualize preprocess and final result and save them

        Return:
            list of dict, each contains the follow keys for recognition result.
            e.g. [{'texts': 'abc', 'confs': 0.9}, {'texts': 'cd', 'confs': 1.0}]
                - texts: text string
                - confs: prediction confidence
        """

        assert isinstance(img_or_path_list, list), "Input for text recognition must be list of images or image paths."
        logger.info(f"num images for rec: {len(img_or_path_list)}")
        if self.batch_mode:
            rec_res_all_crops = self.run_batchwise(img_or_path_list, do_visualize)
        else:
            rec_res_all_crops = []
            for i, img_or_path in enumerate(img_or_path_list):
                rec_res = self.run_single(img_or_path, i, do_visualize)
                rec_res_all_crops.append(rec_res)

        return rec_res_all_crops

    def run_batchwise(self, img_or_path_list: list, do_visualize=False):
        """
        Run text recognition serially for input images

                Args:
            img_or_path_list: list of str for img path or np.array for RGB image
            do_visualize: visualize preprocess and final result and save them

                Return:
            rec_res: list of tuple, where each tuple is  (text, score) - text recognition result for each input image
                in order.
                    where text is the predicted text string, score is its confidence score.
                    e.g. [('apple', 0.9), ('bike', 1.0)]
        """
        rec_res = []
        num_imgs = len(img_or_path_list)

        for idx in range(0, num_imgs, self.batch_num):  # batch begin index i
            batch_begin = idx
            batch_end = min(idx + self.batch_num, num_imgs)
            logger.info(f"Rec img idx range: [{batch_begin}, {batch_end})")
            # TODO: set max_wh_ratio to the maximum wh ratio of images in the batch. and update it for resize,
            #  which may improve recognition accuracy in batch-mode
            # especially for long text image. max_wh_ratio=max(max_wh_ratio, img_w / img_h).
            # The short ones should be scaled with a.r. unchanged and padded to max width in batch.

            # preprocess
            # TODO: run in parallel with multiprocessing
            img_batch = []
            for j in range(batch_begin, batch_end):  # image index j
                data = self.preprocess(img_or_path_list[j])
                img_batch.append(data["image"])

            img_batch = np.stack(img_batch) if len(img_batch) > 1 else np.expand_dims(img_batch[0], axis=0)

            # infer
            net_pred = self.model(ms.Tensor(img_batch))
            if self.cast_pred_fp32:
                if isinstance(net_pred, list) or isinstance(net_pred, tuple):
                    net_pred = [self.cast(p, mstype.float32) for p in net_pred]
                else:
                    net_pred = self.cast(net_pred, mstype.float32)

            # postprocess
            batch_res = self.postprocess(net_pred)
            rec_res.extend(list(zip(batch_res["texts"], batch_res["confs"])))

        return rec_res

    def run_single(self, img_or_path, crop_idx=0, do_visualize=True):
        """
        Text recognition inference on a single image
        Args:
            img_or_path: str for image path or np.array for image rgb value

        Return:
            dict with keys:
                - texts (str): preditive text string
                - confs (int): confidence of the prediction
        """
        # preprocess
        data = self.preprocess(img_or_path)

        logger.info(f"Origin image shape: {data['image_ori'].shape}")
        logger.info(f"Preprocessed image shape: {data['image'].shape}")

        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)

        net_pred = self.model(ms.Tensor(net_input))
        if self.cast_pred_fp32:
            if isinstance(net_pred, list) or isinstance(net_pred, tuple):
                net_pred = [self.cast(p, mstype.float32) for p in net_pred]
            else:
                net_pred = self.cast(net_pred, mstype.float32)

        # postprocess
        rec_res = self.postprocess(net_pred)
        # if 'raw_chars' in rec_res:
        #    rec_res.pop('raw_chars')

        rec_res = (rec_res["texts"][0], rec_res["confs"][0])

        logger.info(f"Crop {crop_idx} rec result: {rec_res}")

        return rec_res


class TextSystem(object):
    def __init__(self, args):
        self.text_detect = TextDetector(args)
        self.text_recognize = TextRecognizer(args)

        self.box_type = args.det_box_type
        self.drop_score = args.drop_score
        self.save_crop_res = args.save_crop_res
        self.crop_res_save_dir = args.crop_res_save_dir
        if self.save_crop_res:
            os.makedirs(self.crop_res_save_dir, exist_ok=True)
        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)
        self.vis_font_path = args.vis_font_path

    def __call__(self, img_or_path: Union[str, np.ndarray], do_visualize=True):
        """
        Detect and recognize texts in an image

        Args:
            img_or_path (str or np.ndarray): path to image or image rgb values as a numpy array

        Return:
            boxes (list): detected text boxes, in shape [num_boxes, num_points, 2], where the point coordinate (x, y)
                follows: x - horizontal (image width direction), y - vertical (image height)
            texts (list[tuple]): list of (text, score) where text is the recognized text string for each box,
                and score is the confidence score.
            time_profile (dict): record the time cost for each sub-task.
        """
        assert isinstance(img_or_path, str) or isinstance(
            img_or_path, np.ndarray
        ), "Input must be string of path to the image or numpy array of the image rgb values."
        fn = os.path.basename(img_or_path).rsplit(".", 1)[0] if isinstance(img_or_path, str) else "img"

        time_profile = {}
        start = time()

        # detect text regions on an image
        det_res, data = self.text_detect(img_or_path, do_visualize=False)
        time_profile["det"] = time() - start
        polys = det_res["polys"].copy()
        logger.info(f"Num detected text boxes: {len(polys)}\nDet time: {time_profile['det']}")

        # crop text regions
        crops = []
        for i in range(len(polys)):
            poly = polys[i].astype(np.float32)
            cropped_img = crop_text_region(data["image_ori"], poly, box_type=self.box_type)
            crops.append(cropped_img)

            if self.save_crop_res:
                cv2.imwrite(os.path.join(self.crop_res_save_dir, f"{fn}_crop_{i}.jpg"), cropped_img)
        # show_imgs(crops, is_bgr_img=False)

        # recognize cropped images
        rs = time()
        rec_res_all_crops = self.text_recognize(crops, do_visualize=False)
        time_profile["rec"] = time() - rs

        logger.info(
            "Recognized texts: \n"
            + "\n".join([f"{text}\t{score}" for text, score in rec_res_all_crops])
            + f"\nRec time: {time_profile['rec']}"
        )

        # filter out low-score texts and merge detection and recognition results
        boxes, text_scores = [], []
        for i in range(len(polys)):
            box = det_res["polys"][i]
            # box_score = det_res["scores"][i]
            text = rec_res_all_crops[i][0]
            text_score = rec_res_all_crops[i][1]
            if text_score >= self.drop_score:
                boxes.append(box)
                text_scores.append((text, text_score))

        time_profile["all"] = time() - start

        return boxes, text_scores, time_profile
