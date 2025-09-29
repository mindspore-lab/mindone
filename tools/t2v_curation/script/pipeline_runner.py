import argparse
import json
import os
import re
import shlex
import subprocess
import sys

import yaml


def load_config(config_path):
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r") as f:
        if ext == ".json":
            config = json.load(f)
        else:
            config = yaml.safe_load(f)

    # handle cases like `${paths.ROOT_META}/meta_fmin${meta_steps.remove_broken_videos.fmin}.csv` recursively
    def resolve_vars(item, vars_dict):
        if isinstance(item, dict):
            return {k: resolve_vars(v, vars_dict) for k, v in item.items()}
        elif isinstance(item, list):
            return [resolve_vars(elem, vars_dict) for elem in item]
        elif isinstance(item, str):
            # ${var} patterns
            pattern = re.compile(r"\$\{([^}^{]+)\}")
            while True:
                match = pattern.search(item)
                if not match:
                    break
                full_match = match.group(0)
                var_name = match.group(1)
                # split var_name by '.' to traverse config dictionary
                var_parts = var_name.split(".")
                var_value = vars_dict
                for part in var_parts:
                    var_value = var_value.get(part)
                    if var_value is None:
                        raise ValueError(f"Variable '{var_name}' not found in configuration.")
                item = item.replace(full_match, str(var_value))
            # handling shell command substitutions like $(pwd)
            shell_pattern = re.compile(r"\$\(([^)]+)\)")
            while True:
                match = shell_pattern.search(item)
                if not match:
                    break
                full_match = match.group(0)
                cmd = match.group(1)
                # only allow 'pwd' here - all we need
                if cmd.strip() == "pwd":
                    cmd_output = os.getcwd()
                else:
                    raise ValueError(f"Unsupported shell command '{cmd}' in configuration.")
                item = item.replace(full_match, cmd_output)
            return item
        else:
            return item

    config = resolve_vars(config, config)
    return config


def run_command(command):
    print(f"Running: {command}")
    subprocess.run(shlex.split(command), shell=False, check=True)


def sanitize_filename(s):
    # for naming during option filtering, replace ' ' with '_' and more to get a valid file name.
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="Path to the input config file, support yaml or json")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if not (args.config_path.endswith(".yaml") or args.config_path.endswith(".json")):
        print("Error: Config file must be .yaml or .json")
        sys.exit(1)
    config = load_config(args.config_path)

    # set environment variables
    os.environ["ROOT_VIDEO"] = config["paths"]["ROOT_VIDEO"]
    os.environ["ROOT_CLIPS"] = config["paths"]["ROOT_CLIPS"]
    os.environ["ROOT_META"] = config["paths"]["ROOT_META"]
    os.environ["PYTHONPATH"] = config["paths"]["PYTHONPATH"]

    meta_steps = config["meta_steps"]
    if meta_steps["run"]:
        # step 1: convert dataset
        if meta_steps["convert_dataset"]["run"]:
            input_video_path = config["paths"]["ROOT_VIDEO"]
            output_meta_csv = meta_steps["convert_dataset"]["output_meta_csv"]
            run_command(f"python -m pipeline.datasets.convert video {input_video_path} --output {output_meta_csv}")

            # remove broken videos
            if meta_steps["remove_broken_videos"]["run"]:
                input_meta_csv = output_meta_csv  # for readability, input is the same as output meta from above
                fmin = meta_steps["remove_broken_videos"]["fmin"]
                run_command(f"python -m pipeline.datasets.datautil {input_meta_csv} --info --fmin {fmin}")

        # step 2: split video
        split_video = meta_steps["split_video"]
        if split_video["run"]:
            # scene detection
            if split_video["scene_detection"]["run"]:
                input_meta_csv = split_video["scene_detection"]["input_meta_csv"]
                detector = split_video["scene_detection"]["detector"]
                max_cutscene_len = split_video["scene_detection"]["max_cutscene_len"]
                command = f"python -m pipeline.splitting.scene_detect {input_meta_csv} --detector {detector}"
                if max_cutscene_len is not None and max_cutscene_len != "None":  # just to play safe
                    command += f" --max_cutscene_len {max_cutscene_len}"
                run_command(command)

            # cut videos
            if split_video["cut_videos"]["run"]:
                min_seconds = split_video["cut_videos"]["min_seconds"]
                max_seconds = split_video["cut_videos"]["max_seconds"]
                target_fps = split_video["cut_videos"]["target_fps"]
                shorter_size = split_video["cut_videos"]["shorter_size"]
                drop_invalid_timestamps = split_video["cut_videos"]["drop_invalid_timestamps"]
                input_meta_csv = input_meta_csv[:-4] + "_timestamp.csv"  # inferred csv name from scene detection
                save_dir = config["paths"]["ROOT_CLIPS"]
                command = f"python -m pipeline.splitting.cut {input_meta_csv} --save_dir {save_dir}"
                if min_seconds is not None and min_seconds != "None":
                    command += f" --min_seconds {min_seconds}"
                if max_seconds is not None and max_seconds != "None":
                    command += f" --max_seconds {max_seconds}"
                if target_fps is not None and target_fps != "None":
                    command += f" --target_fps {target_fps}"
                if shorter_size is not None and shorter_size != "None":
                    command += f" --shorter_size {shorter_size}"
                if drop_invalid_timestamps is not None and drop_invalid_timestamps != "None":
                    command += f" --drop_invalid_timestamps {drop_invalid_timestamps}"
                run_command(command)

            # create clips meta info
            if split_video["create_clips_meta"]["run"]:
                input_clips_path = config["paths"]["ROOT_CLIPS"]
                output_meta_csv = split_video["create_clips_meta"]["output_meta_csv"]
                run_command(f"python -m pipeline.datasets.convert video {input_clips_path} --output {output_meta_csv}")

            # remove broken clips
            if split_video["remove_broken_clips"]["run"]:
                input_meta_csv = output_meta_csv  # for readability, input is the same as output meta from above
                fmin = split_video["remove_broken_clips"]["fmin"]
                run_command(f"python -m pipeline.datasets.datautil {input_meta_csv} --info --fmin {fmin}")

    pipeline_steps = config["pipeline_steps"]
    if pipeline_steps["run"]:
        input_meta_csv = pipeline_steps["input_meta_csv"]

        # step 3: deduplication
        if pipeline_steps["deduplication"]["run"]:
            hash = pipeline_steps["deduplication"]["hash"]
            threshold = pipeline_steps["deduplication"]["threshold"]
            run_command(
                f"python -m pipeline.datasets.deduplication {input_meta_csv} " f"--hash {hash} --threshold {threshold}"
            )
            # update `input_meta_csv` name for later steps, same for all below
            input_meta_csv = input_meta_csv[:-4] + "_dedup.csv"

        # step 4: scoring filtering
        scoring_filtering = pipeline_steps["scoring_filtering"]
        if scoring_filtering["run"]:
            # option matching
            if scoring_filtering["option_matching"]["run"]:
                option = scoring_filtering["option_matching"]["option"]
                bs = scoring_filtering["option_matching"]["batch_size"]
                num_frames = scoring_filtering["option_matching"]["num_frames"]
                if scoring_filtering["option_matching"]["use_ascend"]:
                    worker_num = scoring_filtering["option_matching"]["worker_num"]
                    run_command(
                        f"msrun --worker_num={worker_num} --local_worker_num={worker_num} --join=True "
                        f"--log_dir=msrun_log/option pipeline/scoring/matching/inference.py {input_meta_csv} "
                        f'--option "{option}" --bs {bs} --num_frames {num_frames}'
                    )
                else:
                    run_command(
                        f"python -m pipeline.scoring.matching.inference {input_meta_csv} "
                        f'--option "{option}" --use_cpu --bs {bs} --num_frames {num_frames}'
                    )
                option_safe = sanitize_filename(option)
                input_meta_csv = input_meta_csv[:-4] + f"_{option_safe}.csv"

                # option filtering
                if scoring_filtering["option_filtering"]["run"]:
                    matchmin = scoring_filtering["option_filtering"]["matchmin"]
                    run_command(f"python -m pipeline.datasets.datautil {input_meta_csv} --matchmin {matchmin}")
                    input_meta_csv = input_meta_csv[:-4] + f"_matchmin{matchmin:.1f}.csv"

            # ocr scoring
            if scoring_filtering["ocr_scoring"]["run"]:
                num_boxes = scoring_filtering["ocr_scoring"]["num_boxes"]
                max_single_percentage = scoring_filtering["ocr_scoring"]["max_single_percentage"]
                total_text_percentage = scoring_filtering["ocr_scoring"]["total_text_percentage"]
                command = f"msrun --worker_num=1 --local_worker_num=1 --join=True --log_dir=msrun_log/ocr pipeline/scoring/ocr/inference.py {input_meta_csv}"
                if num_boxes:
                    command += " --num_boxes"
                if max_single_percentage:
                    command += " --max_single_percentage"
                if total_text_percentage:
                    command += " --total_text_percentage"
                run_command(command)
                input_meta_csv = input_meta_csv[:-4] + "_ocr.csv"

                # ocr filtering
                if scoring_filtering["ocr_filtering"]["run"]:
                    ocr_box_max = scoring_filtering["ocr_filtering"]["ocr_box_max"]
                    if ocr_box_max is not None and ocr_box_max != "None":
                        output_meta_csv = input_meta_csv[:-4] + f"_ocrboxmax{int(ocr_box_max)}.csv"
                        run_command(
                            f"python -m pipeline.datasets.datautil {input_meta_csv} --ocr_box_max {ocr_box_max} --output {output_meta_csv}"
                        )
                        input_meta_csv = input_meta_csv[:-4] + f"_ocrboxmax{int(ocr_box_max)}.csv"

                    ocr_single_max = scoring_filtering["ocr_filtering"]["ocr_single_max"]
                    if ocr_single_max is not None and ocr_single_max != "None":
                        output_meta_csv = input_meta_csv[:-4] + f"_ocrsinglemax{ocr_single_max:.1f}.csv"
                        run_command(
                            f"python -m pipeline.datasets.datautil {input_meta_csv} --ocr_single_max {ocr_single_max} --output {output_meta_csv}"
                        )
                        input_meta_csv = input_meta_csv[:-4] + f"_ocrsinglemax{ocr_single_max:.1f}.csv"

                    ocr_total_max = scoring_filtering["ocr_filtering"]["ocr_total_max"]
                    if ocr_total_max is not None and ocr_total_max != "None":
                        output_meta_csv = input_meta_csv[:-4] + f"_ocrtotalmax{ocr_total_max:.1f}.csv"
                        run_command(
                            f"python -m pipeline.datasets.datautil {input_meta_csv} --ocr_total_max {ocr_total_max} --output {output_meta_csv}"
                        )
                        input_meta_csv = input_meta_csv[:-4] + f"_ocrtotalmax{ocr_total_max:.1f}.csv"

            # lpips scoring
            if scoring_filtering["lpips_scoring"]["run"]:
                seconds = scoring_filtering["lpips_scoring"]["seconds"]
                target_height = scoring_filtering["lpips_scoring"]["target_height"]
                target_width = scoring_filtering["lpips_scoring"]["target_width"]
                if scoring_filtering["lpips_scoring"]["use_ascend"]:
                    worker_num = scoring_filtering["lpips_scoring"]["worker_num"]
                    run_command(
                        f"msrun --worker_num={worker_num} --local_worker_num={worker_num} --join=True "
                        f"--log_dir=msrun_log/lpips pipeline/scoring/lpips/inference.py {input_meta_csv} "
                        f"--seconds {seconds} --target_height {target_height} --target_width {target_width}"
                    )
                else:
                    run_command(
                        f"python -m pipeline.scoring.lpips.inference {input_meta_csv} --use_cpu "
                        f"--seconds {seconds} --target_height {target_height} --target_width {target_width}"
                    )
                input_meta_csv = input_meta_csv[:-4] + "_lpips.csv"

                # lpips filtering
                if scoring_filtering["lpips_filtering"]["run"]:
                    lpipsmin = scoring_filtering["lpips_filtering"]["lpipsmin"]
                    output_meta_csv = input_meta_csv[:-4] + f"_lpipsmin{lpipsmin:.1f}.csv"
                    run_command(
                        f"python -m pipeline.datasets.datautil {input_meta_csv} --lpipsmin {lpipsmin} --output {output_meta_csv}"
                    )
                    input_meta_csv = output_meta_csv

            # aesthetic scoring
            if scoring_filtering["aesthetic_scoring"]["run"]:
                bs = scoring_filtering["aesthetic_scoring"]["batch_size"]
                num_frames = scoring_filtering["aesthetic_scoring"]["num_frames"]
                if scoring_filtering["aesthetic_scoring"]["use_ascend"]:
                    worker_num = scoring_filtering["aesthetic_scoring"]["worker_num"]
                    run_command(
                        f"msrun --worker_num={worker_num} --local_worker_num={worker_num} --join=True "
                        f"--log_dir=msrun_log/aes pipeline/scoring/aesthetic/inference.py {input_meta_csv} "
                        f"--bs {bs} --num_frames {num_frames}"
                    )
                else:
                    run_command(
                        f"python -m pipeline.scoring.aesthetic.inference {input_meta_csv} --use_cpu "
                        f"--bs {bs} --num_frames {num_frames}"
                    )
                input_meta_csv = input_meta_csv[:-4] + "_aes.csv"

                # aesthetic filtering
                if scoring_filtering["aesthetic_filtering"]["run"]:
                    aesmin = scoring_filtering["aesthetic_filtering"]["aesmin"]
                    output_meta_csv = input_meta_csv[:-4] + f"_aesmin{aesmin:.1f}.csv"
                    run_command(
                        f"python -m pipeline.datasets.datautil {input_meta_csv} --aesmin {aesmin} --output {output_meta_csv}"
                    )
                    input_meta_csv = output_meta_csv

            # nsfw scoring
            if scoring_filtering["nsfw_scoring"]["run"]:
                num_frames = scoring_filtering["nsfw_scoring"]["num_frames"]
                threshold = scoring_filtering["nsfw_scoring"]["threshold"]
                batch_size = scoring_filtering["nsfw_scoring"]["batch_size"]
                if scoring_filtering["nsfw_scoring"]["use_ascend"]:
                    worker_num = scoring_filtering["nsfw_scoring"]["worker_num"]
                    run_command(
                        f"msrun --worker_num={worker_num} --local_worker_num={worker_num} --join=True "
                        f"--log_dir=msrun_log/nsfw pipeline/scoring/nsfw/inference.py {input_meta_csv} "
                        f"--num_frames {num_frames} --threshold {threshold} --bs {batch_size}"
                    )
                else:
                    run_command(
                        f"python -m pipeline.scoring.nsfw.inference {input_meta_csv} "
                        f"--num_frames {num_frames} --threshold {threshold} --bs {batch_size}"
                    )
                input_meta_csv = input_meta_csv[:-4] + "_nsfw.csv"

                # nsfw filtering
                if scoring_filtering["nsfw_filtering"]["run"]:
                    run_command(f"python -m pipeline.datasets.datautil {input_meta_csv} --safety_check")
                    input_meta_csv = input_meta_csv[:-4] + "_safe.csv"
            # nsfw scoring

        # step 5: captioning
        captioning = pipeline_steps["captioning"]
        if captioning["run"]:
            # qwen2vl captioning
            if captioning["qwen2vl_caption"]["run"]:
                question = captioning["qwen2vl_caption"]["question"]
                height = captioning["qwen2vl_caption"]["height"]
                width = captioning["qwen2vl_caption"]["width"]
                fps = captioning["qwen2vl_caption"]["fps"]
                max_new_tokens = captioning["qwen2vl_caption"]["max_new_tokens"]
                worker_num = captioning["qwen2vl_caption"]["worker_num"]
                batch_size = captioning["qwen2vl_caption"]["batch_size"]
                run_command(
                    f"msrun --worker_num={worker_num} --local_worker_num={worker_num} --join=True "
                    f"--log_dir=msrun_log/qwen2vl pipeline/captioning/caption_qwen2vl.py {input_meta_csv} "
                    f'--question "{question}" --height {height} --width {width} '
                    f"--fps {fps} --bs {batch_size} --max_new_tokens {max_new_tokens}"
                )
                input_meta_csv = input_meta_csv[:-4] + "_caption_qwen2vl.csv"

            # Llava captioning
            if captioning["llava_caption"]["run"]:
                question = captioning["llava_caption"]["question"]
                max_new_tokens = captioning["llava_caption"]["max_new_tokens"]
                worker_num = captioning["llava_caption"]["worker_num"]

                run_command(
                    f"msrun --worker_num={worker_num} --local_worker_num={worker_num} --join=True "
                    f"--log_dir=msrun_log/llava pipeline/captioning/caption_llava.py {input_meta_csv} "
                    f'--question "{question}" --max_new_tokens {max_new_tokens}'
                )
                input_meta_csv = input_meta_csv[:-4] + "_caption_llava.csv"

            # pLlava captioning
            if captioning["pllava_caption"]["run"]:
                question = captioning["pllava_caption"]["question"]
                num_frames = captioning["pllava_caption"]["num_frames"]
                max_new_tokens = captioning["pllava_caption"]["max_new_tokens"]
                worker_num = captioning["pllava_caption"]["worker_num"]

                run_command(
                    f"msrun --worker_num={worker_num} --local_worker_num={worker_num} --join=True "
                    f"--log_dir=msrun_log/pllava pipeline/captioning/caption_pllava.py {input_meta_csv} "
                    f'--question "{question}" --num_frames {num_frames} --max_new_tokens {max_new_tokens}'
                )
                input_meta_csv = input_meta_csv[:-4] + "_caption_pllava.csv"

            # clean caption
            if captioning["clean_caption"]["run"]:
                clean_options = []
                output_meta_csv = input_meta_csv[:-4] + "_cleaned.csv"
                if captioning["clean_caption"]["clean_caption"]:
                    clean_options.append("--clean-caption")
                if captioning["clean_caption"]["refine_llm_caption"]:
                    clean_options.append("--refine-llm-caption")
                if captioning["clean_caption"]["remove_empty_caption"]:
                    clean_options.append("--remove-empty-caption")
                clean_options = " ".join(clean_options)
                run_command(
                    f"python -m pipeline.datasets.datautil {input_meta_csv} {clean_options} --output {output_meta_csv}"
                )
                input_meta_csv = output_meta_csv

            # matching score with captions
            if captioning["matching_with_captions"]["run"]:
                bs = captioning["matching_with_captions"]["batch_size"]
                num_frames = captioning["matching_with_captions"]["num_frames"]
                if captioning["matching_with_captions"]["use_ascend"]:
                    worker_num = captioning["matching_with_captions"]["worker_num"]
                    run_command(
                        f"msrun --worker_num={worker_num} --local_worker_num={worker_num} --join=True "
                        f"--log_dir=msrun_log/match pipeline/scoring/matching/inference.py {input_meta_csv} "
                        f"--bs {bs} --num_frames {num_frames}"
                    )
                else:
                    run_command(
                        f"python -m pipeline.scoring.matching.inference {input_meta_csv} --use_cpu "
                        f"--bs {bs} --num_frames {num_frames}"
                    )
                input_meta_csv = input_meta_csv[:-4] + "_matching.csv"

                if captioning["caption_filtering"]["run"]:
                    matchmin = captioning["caption_filtering"]["matchmin"]
                    run_command(f"python -m pipeline.datasets.datautil {input_meta_csv} --matchmin {matchmin}")
                    input_meta_csv = input_meta_csv[:-4] + f"_matchmin{matchmin:.1f}.csv"

    print(f"Finished processing. The final meta csv has been saved to {input_meta_csv}.")


if __name__ == "__main__":
    main()
