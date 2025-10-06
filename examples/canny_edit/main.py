import argparse
import os
import random
import sys
import warnings

import numpy as np
from PIL import Image
from src.cannyedit_pipeline import CannyEditPipeline
from src.util import plot_image_with_mask, process_mask

import mindspore as ms
import mindspore.nn as nn

# Suppress all warnings
warnings.filterwarnings("ignore")


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_checkpoint_path", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--prompt_source", type=str, help="The text prompt that describes the source image"  # required=True,
    )
    parser.add_argument(
        "--prompt_target", help="The text prompt that describes the targeted image after editing"  # required=True,
    )
    parser.add_argument(
        "--prompt_local",
        action="append",
        help="The local prompt(s) for edit region(s)",
    )
    parser.add_argument(
        "--mask_path",
        action="append",
        help="path(s) of mask(s) indicating the region to edit",
    )
    parser.add_argument("--dilate_mask", action="store_true", help="Dilate the mask")
    parser.add_argument(
        "--fill_hole_mask",
        action="store_true",
        default=True,
        help="Fill the holes in the mask, useful for the imprecise segmentation masks",
    )
    parser.add_argument("--width", type=int, default=768, help="The width for generated image")
    parser.add_argument("--height", type=int, default=768, help="The height for generated image")
    parser.add_argument(
        "--image_whratio_unchange",
        action="store_true",
        help="In default we use square input/output, set this to True if you wish to keep the original image width/height ratio unchanged.",
    )
    parser.add_argument("--save_folder", type=str, default="./cannyedit_outputs/", help="Folder to save")
    parser.add_argument(
        "--neg_prompt2",
        type=str,
        default="focus,centered foreground, humans, objects, noise, blurring, low resolution, artifacts, distortion, "
        "overexposure, and uneven lighting, bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs,"
        " bad arms, missing legs, missing arms, poorly drawn face,  disconnected limbs",
        help="The input text negative prompt2",
    )
    # 'oval, noise, plaid, polka-dot, leopard print, cartoon, unreal, animate, amputation, '
    parser.add_argument(
        "--neg_prompt",
        type=str,
        default="humans, objects, noise, blurring, low resolution, artifacts, distortion, overexposure, and uneven lighting,"
        " bad anatomy, bad hands, missing fingers, extra fingers, three hands, three legs, bad arms, missing legs, missing arms,"
        " poorly drawn face,  disconnected limbs",
        help="The input text negative prompt",
    )
    parser.add_argument("--control_weight2", type=float, default=0.7, help="Controlnet model strength (from 0 to 1.0)")
    parser.add_argument(
        "--multi_run",
        action="store_true",
        help="If true, we will cache the inversion result and previous generation result, and then allow the multi-run edits",
    )
    parser.add_argument("--inversion_save_path", type=str, default=None, help="Path to save the inversion result")
    parser.add_argument(
        "--generate_save_path", type=str, default=None, help="Path to save the previous generation result"
    )
    parser.add_argument("--img_prompt", type=str, default=None, help="Path to input image prompt")
    parser.add_argument("--neg_img_prompt", type=str, default=None, help="Path to input negative image prompt")
    parser.add_argument("--local_path", type=str, default=None, help="Local path to the model checkpoint (Controlnet)")
    parser.add_argument(
        "--repo_id", type=str, default=None, help="A HuggingFace repo id to download model (Controlnet)"
    )
    parser.add_argument("--name", type=str, default=None, help="A filename to download from HuggingFace")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--use_controlnet", action="store_true", help="Load Controlnet model")
    parser.add_argument("--use_paint", action="store_true", help="Load inpainting model")
    parser.add_argument("--image_path", type=str, default=None, help="Path to image")
    parser.add_argument("--control_weight", type=float, default=0.8, help="Controlnet model strength (from 0 to 1.0)")
    parser.add_argument(
        "--control_type",
        type=str,
        default="canny",
        choices=("canny", "openpose", "depth", "zoe", "hed", "hough", "tile"),
        help="Name of controlnet condition, example: canny",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="flux-dev",
        choices=("flux-dev", "flux-dev-fp8", "flux-schnell"),
        help="Model type to use (flux-dev, flux-dev-fp8, flux-schnell)",
    )
    parser.add_argument("--num_steps", type=int, default=50, help="The num_steps for diffusion process")
    parser.add_argument("--guidance", type=float, default=4, help="The guidance for diffusion process")
    parser.add_argument(
        "--seed", type=int, default=random.randint(0, 9999999), help="A seed for reproducible inference"
    )
    parser.add_argument("--true_gs", type=float, default=2, help="true guidance")
    parser.add_argument("--timestep_to_start_cfg", type=int, default=5, help="timestep to start true guidance")
    return parser


def generate_output_by_qwen(qwen_model, qwen_processor, image_path, height, width, prompt, max_new_tokens=128):
    """
    Processes an image and text input, passes them through a model, and generates output text.

    Args:
        qwen_model: The pre-trained model for inference.
        qwen_processor: The processor for text and vision inputs.
        image_path (str): Path to the input image.
        height (int): Original height of the input image.
        width (int): Original width of the input image.
        prompt (str): Text prompt to guide the model's generation.
        max_new_tokens (int): Maximum number of tokens to generate. Default is 128.

    Returns:
        str: Decoded output text from the model.
    """
    # Prepare the messages with resized image and the text prompt
    from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": int(height / 2.5),
                    "resized_width": int(width / 2.5),
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process the text template
    text = qwen_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process image and video inputs
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare the input tensors
    inputs = qwen_processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="np")
    for k, v in inputs.items():
        inputs[k] = ms.Tensor(v)

    # Generate the output
    generated_ids = qwen_model.generate(**inputs, max_new_tokens=max_new_tokens)

    # Trim the generated IDs to exclude input tokens
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

    # Decode the output text
    output_text = qwen_processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text


def main(args):
    removal_flag = False
    mask_path_list = []
    image = Image.open(args.image_path).convert("RGB")
    if args.image_whratio_unchange is True:
        widtho, heighto = image.size
        maxone = np.max([widtho, heighto])
        if maxone == widtho:
            args.width = args.width
            args.height = int(args.width * (heighto / widtho))
        else:
            args.height = args.height
            args.width = int(args.height * (widtho / heighto))
        print("Keep image width/height ratio unchanged, we now use：[width, height]=" + str([args.width, args.height]))

    cannyedit_pipeline = CannyEditPipeline("flux-dev", offload=args.offload)
    cannyedit_pipeline.set_controlnet(
        "canny", None, "XLabs-AI/flux-controlnet-canny-v3", "flux-canny-controlnet-v3.safetensors"
    )

    #  Input local prompt
    if args.prompt_local is None:
        args.prompt_local = []
        print("No local prompt provided. Do you want to enter the local prompt here?")
        resp = input("Press 'y' for yes, anything else for no and exit:").strip().lower()
        if resp == "y":
            args.prompt_local.append(input("Enter the first local prompt: "))
            for kk in range(10):
                resp = input(
                    "Enter the next local prompt if you may have, enter 'done' if you have finished all inputs: "
                )
                if resp == "done":
                    break
                else:
                    args.prompt_local.append(resp)
        else:
            print("\n")
            print("Exiting CannyEdit.")
            sys.exit(1)  # Exit with an error code

    for pp_ind in range(len(args.prompt_local)):
        if "[remove]" in args.prompt_local[pp_ind]:
            args.prompt_local[pp_ind] = "empty background" + " out-of-focus, atmospheric background"

    # --------------------------------------------------------------------------------------
    # Read the mask files is provided
    if args.mask_path is not None:
        dilate_mask = args.dilate_mask
        if "empty background" in args.prompt_local[0]:
            removal_flag = True
        local_mask = process_mask(
            args.mask_path[0],
            args.height,
            args.width,
            dilate=dilate_mask,
            dilation_kernel_size=(5, 5),
            fill_holes=args.fill_hole_mask,
            closing_kernel_size=(1, 1),
        )
        mask_path_list.append(args.mask_path[0])
        local_mask_addition = []
        mask_count = 1
        for maskp in args.mask_path[1:]:
            dilate_mask = args.dilate_mask
            # removal_add
            if "empty background" in args.prompt_local[mask_count]:
                removal_flag = True
            local_mask_addition.append(
                process_mask(
                    maskp,
                    args.height,
                    args.width,
                    dilate=dilate_mask,
                    dilation_kernel_size=(5, 5),
                    fill_holes=args.fill_hole_mask,
                    closing_kernel_size=(1, 1),
                )
            )
            mask_path_list.append(maskp)
            mask_count += 1

    else:
        raise ValueError("mask_path must be provided!")

    result_save_path = ""
    # Apply vlm to generate source prompt and target prompt if not provided
    if args.prompt_source is None or args.prompt_target is None:
        print("no source/target prompt is provided, using QWEN2.5-VL to generate the prompt automatically \n")
        from transformers import AutoProcessor

        from mindone.transformers import Qwen2_5_VLForConditionalGeneration

        with nn.no_init_parameters():
            qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.qwen_checkpoint_path, mindspore_dtype=ms.bfloat16
            )
        min_pixels = 256 * 28 * 28
        max_pixels = 512 * 28 * 28
        qwen_processor = AutoProcessor.from_pretrained(
            args.qwen_checkpoint_path, min_pixels=min_pixels, max_pixels=max_pixels
        )
        if args.prompt_source is None:
            output_text = generate_output_by_qwen(
                qwen_model,
                qwen_processor,
                args.image_path,
                args.height,
                args.width,
                "Describe this image in 15 words.",
                max_new_tokens=128,
            )
            args.prompt_source = output_text[0]
            print("\n")
            print("VLM generated source prompt: " + args.prompt_source)
            print("\n")
        if args.prompt_target is None:
            # removal_add
            print(
                "Important: Currently the auto generation of target prompts only support **adding and removal**. If the editing involves only "
                "edit tasks like replacement, please provide the target prompt here, you may refer to the VLM-generated source prompt.\n"
            )
            if removal_flag is False:
                resp = input(
                    "Press '1' for using VLM to geernate target prompt, other enter the target prompt directly: \n"
                )
            elif removal_flag is True:
                resp = "1"

            if resp != "1":
                args.prompt_target = resp
                print("Entered target prompt: " + args.prompt_target)
                print("\n")
            if resp == "1":
                if removal_flag is False:
                    prompt_for_target = (
                        "Given the caption for this image:"
                        + str(args.prompt_source)
                        + "Suppose there would be new objects in the image:"
                    )
                    words_count = 15
                    for object_add in args.prompt_local:
                        prompt_for_target += object_add + "; and "
                        words_count += 5
                    prompt_for_target += (
                        "\n Based on the original caption and the description to the new objects. Generate the new caption after the objects are added in "
                        + str(words_count)
                        + " words."
                    )  # Keep the original caption if possible'
                    output_text = generate_output_by_qwen(
                        qwen_model,
                        qwen_processor,
                        args.image_path,
                        args.height,
                        args.width,
                        prompt_for_target,
                        max_new_tokens=128,
                    )

                    args.prompt_target = output_text[0]
                    print("VLM generated target prompt: " + args.prompt_target)
                    print("\n")
                # removal_add
                elif removal_flag is True:
                    args.prompt_target = " "
                    save_path = plot_image_with_mask(
                        args.image_path,
                        mask_path_list,
                        width=args.width,
                        height=args.height,
                        save_path="assets/mask_temp/masktpimage.png",
                    )
                    output_text = generate_output_by_qwen(
                        qwen_model,
                        qwen_processor,
                        save_path,
                        args.height,
                        args.width,
                        "Support the objects within the red bounding box will be removed. Describe the image background excluding"
                        " the removed objects in 10 words.",
                        max_new_tokens=128,
                    )

                    args.prompt_target = output_text[0]
                    print("\n")
                    print("VLM generated target prompt for the removal task: " + args.prompt_target)
                    print("\n")

                    output_text = generate_output_by_qwen(
                        qwen_model,
                        qwen_processor,
                        save_path,
                        args.height,
                        args.width,
                        "describe the region within the red bounding box in 10 words.",
                        max_new_tokens=128,
                    )
                    args.neg_prompt = output_text[0]
                    print("\n")
                    print("VLM generated negative prompt for the removal task: " + output_text[0])
                    print("\n")

        del qwen_model
        del qwen_processor

        # --------------------------------------------------------------------------------------
        print("Running CannyEdit")
        # Stage 1: Generation
        stage1 = "stage_removal"
        result = cannyedit_pipeline(
            prompt_source=args.prompt_source,
            prompt_local1=args.prompt_local[0],
            prompt_target=args.prompt_target,
            prompt_local_addition=args.prompt_local[1:],
            controlnet_image=image,
            local_mask=local_mask,
            local_mask_addition=local_mask_addition,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed,
            true_gs=args.true_gs,
            control_weight=args.control_weight,
            control_weight2=args.control_weight2,
            neg_prompt=args.neg_prompt,
            # removal_add
            neg_prompt2=args.neg_prompt2,
            timestep_to_start_cfg=args.timestep_to_start_cfg,
            stage=stage1,
            generate_save_path=args.generate_save_path,
            inversion_save_path=args.inversion_save_path,
        )

        #  Save the edited image
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)
        ind = len(os.listdir(args.save_folder))
        result_save_path = os.path.join(args.save_folder, f"result_{ind}.png")
        result.save(result_save_path)

    if removal_flag is False:
        # Stage 1: Generation
        stage1 = "stage_generate"
        print("Running CannyEdit")
        result = cannyedit_pipeline(
            prompt_source=args.prompt_source,
            prompt_local1=args.prompt_local[0],
            prompt_target=args.prompt_target,
            prompt_local_addition=args.prompt_local[1:],
            controlnet_image=image,
            local_mask=local_mask,
            local_mask_addition=local_mask_addition,
            width=args.width,
            height=args.height,
            guidance=args.guidance,
            num_steps=args.num_steps,
            seed=args.seed,
            true_gs=args.true_gs,
            control_weight=args.control_weight,
            control_weight2=args.control_weight2,
            neg_prompt=args.neg_prompt,
            neg_prompt2=args.neg_prompt2,
            timestep_to_start_cfg=args.timestep_to_start_cfg,
            stage=stage1,
            generate_save_path=args.generate_save_path,
            inversion_save_path=args.inversion_save_path,
        )

        #  Save the edited image
        if not os.path.exists(args.save_folder):
            os.mkdir(args.save_folder)
        ind = len(os.listdir(args.save_folder))
        result_save_path = os.path.join(args.save_folder, f"result_{ind}.png")
        result.save(result_save_path)

    if result_save_path:
        print(f"Generated image saved in {result_save_path}")

    # remove all cached files
    if args.inversion_save_path is not None and os.path.exists(args.inversion_save_path):
        os.remove(args.inversion_save_path)
    if args.generate_save_path is not None and os.path.exists(args.generate_save_path):
        os.remove(args.generate_save_path)


if __name__ == "__main__":
    args = create_argparser().parse_args()
    main(args)
