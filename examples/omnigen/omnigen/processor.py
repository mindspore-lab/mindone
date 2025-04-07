import re
from typing import Dict, List

from omnigen.utils import center_crop_arr, crop_arr
from PIL import Image
from transformers import AutoTokenizer

import mindspore as ms
import mindspore.ops as ops
from mindspore.dataset import transforms, vision


class CropTransform:
    def __init__(self, max_image_size):
        self.max_image_size = max_image_size

    def __call__(self, img):
        return crop_arr(img, self.max_image_size)


class CenterCropTransform:
    def __init__(self, image_size):
        self.image_size = image_size

    def __call__(self, img):
        return center_crop_arr(img, self.image_size)


class OmniGenProcessor:
    def __init__(self, text_tokenizer: AutoTokenizer, max_image_size: int = 1024):
        self.text_tokenizer = text_tokenizer
        self.max_image_size = max_image_size

        self.image_transform = transforms.Compose(
            [
                CropTransform(max_image_size),
                vision.ToTensor(),
                vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], is_hwc=False),
            ]
        )

        self.collator = OmniGenCollator()
        self.separate_collator = OmniGenSeparateCollator()

    @classmethod
    def from_pretrained(cls, model_path, dtype=ms.float32):
        text_tokenizer = AutoTokenizer.from_pretrained(model_path)
        return cls(text_tokenizer)

    def process_image(self, image):
        image = Image.open(image).convert("RGB")
        return ms.Tensor.from_numpy(self.image_transform(image)[0])

    def add_prefix_instruction(self, prompt):
        user_prompt = "<|user|>\n"
        generation_prompt = "Generate an image according to the following instructions\n"
        assistant_prompt = "<|assistant|>\n<|diffusion|>"
        prompt_suffix = "<|end|>\n"
        prompt = f"{user_prompt}{generation_prompt}{prompt}{prompt_suffix}{assistant_prompt}"
        return prompt

    def process_multi_modal_prompt(self, text, input_images):
        text = self.add_prefix_instruction(text)
        if input_images is None or len(input_images) == 0:
            model_inputs = self.text_tokenizer(text)
            return {"input_ids": model_inputs.input_ids, "pixel_values": None, "image_sizes": None}

        pattern = r"<\|image_\d+\|>"
        prompt_chunks = [self.text_tokenizer(chunk).input_ids for chunk in re.split(pattern, text)]

        for i in range(1, len(prompt_chunks)):
            if prompt_chunks[i][0] == 1:
                prompt_chunks[i] = prompt_chunks[i][1:]

        image_tags = re.findall(pattern, text)
        image_ids = [int(s.split("|")[1].split("_")[-1]) for s in image_tags]

        unique_image_ids = sorted(list(set(image_ids)))
        assert unique_image_ids == list(
            range(1, len(unique_image_ids) + 1)
        ), f"image_ids must start from 1 and be continuous, e.g. [1, 2, 3], got {unique_image_ids}"
        assert len(unique_image_ids) == len(
            input_images
        ), f"total images must match number of image tags, got {len(unique_image_ids)} tags and {len(input_images)} images"

        input_images = [input_images[x - 1] for x in image_ids]

        all_input_ids = []
        img_inx = []
        # idx = 0
        for i in range(len(prompt_chunks)):
            all_input_ids.extend(prompt_chunks[i])
            if i != len(prompt_chunks) - 1:
                start_inx = len(all_input_ids)
                size = (
                    input_images[i].shape[-2] * input_images[i].shape[-1] // 16 // 16
                )  # Assumes MindSpore tensor shape
                img_inx.append([start_inx, start_inx + size])
                all_input_ids.extend([0] * size)

        return {"input_ids": all_input_ids, "pixel_values": input_images, "image_sizes": img_inx}

    def __call__(
        self,
        instructions: List[str],
        input_images: List[List[str]] = None,
        height: int = 1024,
        width: int = 1024,
        negative_prompt: str = "low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, extra fingers, mutated hands, poorly drawn hands, "
        "poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, "
        "malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers.",
        use_img_cfg: bool = True,
        separate_cfg_input: bool = False,
        use_input_image_size_as_output: bool = False,
    ) -> Dict:
        if input_images is None:
            use_img_cfg = False
        if isinstance(instructions, str):
            instructions = [instructions]
            input_images = [input_images]

        input_data = []
        for i in range(len(instructions)):
            cur_instruction = instructions[i]
            cur_input_images = None if input_images is None else input_images[i]
            if cur_input_images is not None and len(cur_input_images) > 0:
                cur_input_images = [self.process_image(x) for x in cur_input_images]
            else:
                cur_input_images = None
                assert "<img><|image_1|></img>" not in cur_instruction
            mllm_input = self.process_multi_modal_prompt(cur_instruction, cur_input_images)

            neg_mllm_input, img_cfg_mllm_input = None, None
            neg_mllm_input = self.process_multi_modal_prompt(negative_prompt, None)
            if use_img_cfg:
                if cur_input_images is not None and len(cur_input_images) >= 1:
                    img_cfg_prompt = [f"<img><|image_{i+1}|></img>" for i in range(len(cur_input_images))]
                    img_cfg_mllm_input = self.process_multi_modal_prompt(" ".join(img_cfg_prompt), cur_input_images)
                else:
                    img_cfg_mllm_input = neg_mllm_input

            if use_input_image_size_as_output:
                input_data.append(
                    (
                        mllm_input,
                        neg_mllm_input,
                        img_cfg_mllm_input,
                        [mllm_input["pixel_values"][0].shape[-2], mllm_input["pixel_values"][0].shape[-1]],
                    )
                )
            else:
                input_data.append((mllm_input, neg_mllm_input, img_cfg_mllm_input, [height, width]))

        if separate_cfg_input:
            return self.separate_collator(input_data)
        return self.collator(input_data)


class OmniGenCollator:
    def __init__(self, pad_token_id=2, hidden_size=3072):
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size

    def create_position(self, attention_mask, num_tokens_for_output_images):
        position_ids = []

        text_length = attention_mask.shape[-1]
        img_length = max(num_tokens_for_output_images)
        for mask in attention_mask:
            temp_l = ops.sum(mask)
            temp_l = temp_l.asnumpy().item()
            temp_position = [0] * (text_length - temp_l) + [i for i in range(temp_l + img_length + 1)]
            position_ids.append(temp_position)

        return ms.Tensor(position_ids, dtype=ms.int32)

    def create_mask(self, attention_mask, num_tokens_for_output_images):
        extended_mask = []
        padding_images = []
        text_length = attention_mask.shape[-1]
        img_length = max(num_tokens_for_output_images)
        seq_len = text_length + img_length + 1
        inx = 0
        for mask in attention_mask:
            temp_l = ops.sum(mask)
            temp_l = temp_l.asnumpy().item()  # Mindspore need integer, not tensor
            pad_l = text_length - temp_l
            temp_mask = ops.tril(ops.ones((temp_l + 1, temp_l + 1)))

            image_mask = ops.zeros((temp_l + 1, img_length))
            temp_mask = ops.concat([temp_mask, image_mask], axis=-1)

            image_mask = ops.ones((img_length, temp_l + img_length + 1))
            temp_mask = ops.concat([temp_mask, image_mask], axis=0)

            if pad_l > 0:
                pad_mask = ops.zeros((temp_l + 1 + img_length, pad_l))
                temp_mask = ops.concat([pad_mask, temp_mask], axis=-1)

                pad_mask = ops.ones((pad_l, seq_len))
                temp_mask = ops.concat([pad_mask, temp_mask], axis=0)

            true_img_length = num_tokens_for_output_images[inx]
            pad_img_length = img_length - true_img_length
            if pad_img_length > 0:
                temp_mask[:, -pad_img_length:] = 0
                temp_padding_imgs = ops.zeros((1, pad_img_length, self.hidden_size))
            else:
                temp_padding_imgs = None

            extended_mask.append(temp_mask.expand_dims(0))
            padding_images.append(temp_padding_imgs)
            inx += 1
        return ops.concat(extended_mask, axis=0), padding_images

    def adjust_attention_for_input_images(self, attention_mask, image_sizes):
        for b_inx in image_sizes.keys():
            for start_inx, end_inx in image_sizes[b_inx]:
                attention_mask[b_inx][start_inx:end_inx, start_inx:end_inx] = 1
        return attention_mask

    def pad_input_ids(self, input_ids, image_sizes):
        max_l = max([len(x) for x in input_ids])
        padded_ids = []
        attention_mask = []
        # new_image_sizes = []

        for i in range(len(input_ids)):
            temp_ids = input_ids[i]
            temp_l = len(temp_ids)
            pad_l = max_l - temp_l
            if pad_l == 0:
                attention_mask.append([1] * max_l)
                padded_ids.append(temp_ids)
            else:
                attention_mask.append([0] * pad_l + [1] * temp_l)
                padded_ids.append([self.pad_token_id] * pad_l + temp_ids)

            if i in image_sizes:
                new_inx = []
                for old_inx in image_sizes[i]:
                    new_inx.append([x + pad_l for x in old_inx])
                image_sizes[i] = new_inx

        return ms.Tensor(padded_ids, ms.int32), ms.Tensor(attention_mask, ms.int32), image_sizes

    def process_mllm_input(self, mllm_inputs, target_img_size):
        num_tokens_for_output_images = []
        for img_size in target_img_size:
            num_tokens_for_output_images.append(img_size[0] * img_size[1] // 16 // 16)

        pixel_values, image_sizes = [], {}
        b_inx = 0
        for x in mllm_inputs:
            if x["pixel_values"] is not None:
                pixel_values.extend(x["pixel_values"])
                for size in x["image_sizes"]:
                    if b_inx not in image_sizes:
                        image_sizes[b_inx] = [size]
                    else:
                        image_sizes[b_inx].append(size)
            b_inx += 1
        pixel_values = [x.unsqueeze(0) for x in pixel_values]

        input_ids = [x["input_ids"] for x in mllm_inputs]
        padded_input_ids, attention_mask, image_sizes = self.pad_input_ids(input_ids, image_sizes)
        position_ids = self.create_position(attention_mask, num_tokens_for_output_images)
        attention_mask, padding_images = self.create_mask(attention_mask, num_tokens_for_output_images)
        attention_mask = self.adjust_attention_for_input_images(attention_mask, image_sizes)

        return padded_input_ids, position_ids, attention_mask, padding_images, pixel_values, image_sizes

    def __call__(self, features):
        mllm_inputs = [f[0] for f in features]
        cfg_mllm_inputs = [f[1] for f in features]
        img_cfg_mllm_input = [f[2] for f in features]
        target_img_size = [f[3] for f in features]

        if img_cfg_mllm_input[0] is not None:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs + img_cfg_mllm_input
            target_img_size = target_img_size + target_img_size + target_img_size
        else:
            mllm_inputs = mllm_inputs + cfg_mllm_inputs
            target_img_size = target_img_size + target_img_size

        (
            all_padded_input_ids,
            all_position_ids,
            all_attention_mask,
            all_padding_images,
            all_pixel_values,
            all_image_sizes,
        ) = self.process_mllm_input(mllm_inputs, target_img_size)

        data = {
            "input_ids": all_padded_input_ids,
            "attention_mask": all_attention_mask,
            "position_ids": all_position_ids,
            "input_pixel_values": all_pixel_values,
            "input_image_sizes": all_image_sizes,
            "padding_images": all_padding_images,
        }
        return data


class OmniGenSeparateCollator(OmniGenCollator):
    def __call__(self, features):
        mllm_inputs = [f[0] for f in features]
        cfg_mllm_inputs = [f[1] for f in features]
        img_cfg_mllm_input = [f[2] for f in features]
        target_img_size = [f[3] for f in features]

        all_padded_input_ids, all_attention_mask, all_position_ids = [], [], []
        all_pixel_values, all_image_sizes, all_padding_images = [], [], []

        (
            padded_input_ids,
            position_ids,
            attention_mask,
            padding_images,
            pixel_values,
            image_sizes,
        ) = self.process_mllm_input(mllm_inputs, target_img_size)
        all_padded_input_ids.append(padded_input_ids)
        all_attention_mask.append(attention_mask)
        all_position_ids.append(position_ids)
        all_pixel_values.append(pixel_values)
        all_image_sizes.append(image_sizes)
        all_padding_images.append(padding_images)

        if cfg_mllm_inputs[0] is not None:
            (
                padded_input_ids,
                position_ids,
                attention_mask,
                padding_images,
                pixel_values,
                image_sizes,
            ) = self.process_mllm_input(cfg_mllm_inputs, target_img_size)
            all_padded_input_ids.append(padded_input_ids)
            all_attention_mask.append(attention_mask)
            all_position_ids.append(position_ids)
            all_pixel_values.append(pixel_values)
            all_image_sizes.append(image_sizes)
            all_padding_images.append(padding_images)

        if img_cfg_mllm_input[0] is not None:
            (
                padded_input_ids,
                position_ids,
                attention_mask,
                padding_images,
                pixel_values,
                image_sizes,
            ) = self.process_mllm_input(img_cfg_mllm_input, target_img_size)
            all_padded_input_ids.append(padded_input_ids)
            all_attention_mask.append(attention_mask)
            all_position_ids.append(position_ids)
            all_pixel_values.append(pixel_values)
            all_image_sizes.append(image_sizes)
            all_padding_images.append(padding_images)

        data = {
            "input_ids": all_padded_input_ids,
            "attention_mask": all_attention_mask,
            "position_ids": all_position_ids,
            "input_pixel_values": all_pixel_values,
            "input_image_sizes": all_image_sizes,
            "padding_images": all_padding_images,
        }
        return data
