import base64
import random
from io import BytesIO

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageOps

import mindspore as ms
from mindspore import ops

MAX_COLORS = 12


def get_random_bool():
    return random.choice([True, False])


def add_white_border(input_image, border_width=10):
    """
    为PIL图像添加指定宽度的白色边框。

    :param input_image: PIL图像对象
    :param border_width: 边框宽度（单位：像素）
    :return: 带有白色边框的PIL图像对象
    """
    border_color = "white"
    img_with_border = ImageOps.expand(input_image, border=border_width, fill=border_color)
    return img_with_border


def process_mulline_text(draw, text, font, max_width):
    """
    Draw the text on an image with word wrapping.
    """
    lines = []  # Store the lines of text here
    words = text.split()

    # Start building lines of text, and wrap when necessary
    current_line = ""
    for word in words:
        test_line = f"{current_line} {word}".strip()
        # Check the width of the line with this word added
        bbox = draw.textbbox((0, 0), test_line, font=font)
        text_left, text_top, text_right, text_bottom = bbox

        width, _ = (text_right - text_left, text_bottom - text_top)

        if width <= max_width:
            # If it fits, add this word to the current line
            current_line = test_line
        else:
            # If not, store the line and start a new one
            lines.append(current_line)
            current_line = word
    # Add the last line
    lines.append(current_line)
    return lines


def add_caption(
    image, text, position="bottom-mid", font=None, text_color="black", bg_color=(255, 255, 255), bg_opacity=200
):
    if text == "":
        return image
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    lines = process_mulline_text(draw, text, font, width)
    text_positions = []
    maxwidth = 0
    for ind, line in enumerate(lines[::-1]):
        bbox = draw.textbbox((0, 0), line, font=font)
        text_left, text_top, text_right, text_bottom = bbox
        text_width, text_height = (text_right - text_left, text_bottom - text_top)
        if position == "bottom-right":
            text_position = (width - text_width - 10, height - (text_height + 20))
        elif position == "bottom-left":
            text_position = (10, height - (text_height + 20))
        elif position == "bottom-mid":
            text_position = ((width - text_width) // 2, height - (text_height + 20))  # 居中文本
        height = text_position[1]
        maxwidth = max(maxwidth, text_width)
        text_positions.append(text_position)
    rectpos = (width - maxwidth) // 2
    rectangle_position = [
        rectpos - 5,
        text_positions[-1][1] - 5,
        rectpos + maxwidth + 5,
        text_positions[0][1] + text_height + 5,
    ]
    image_with_transparency = Image.new("RGBA", image.size)
    draw_with_transparency = ImageDraw.Draw(image_with_transparency)
    draw_with_transparency.rectangle(rectangle_position, fill=bg_color + (bg_opacity,))

    image.paste(Image.alpha_composite(image.convert("RGBA"), image_with_transparency))
    print(ind, text_position)
    draw = ImageDraw.Draw(image)
    for ind, line in enumerate(lines[::-1]):
        text_position = text_positions[ind]
        draw.text(text_position, line, fill=text_color, font=font)

    return image.convert("RGB")


def get_comic(images, types="4panel", captions=[], font=None, pad_image=None):
    if pad_image is None:
        pad_image = Image.open("./images/pad_images.png")

    if types == "No typesetting (default)":
        return images
    elif types == "Four Pannel":
        return get_comic_4panel(images, captions, font, pad_image)
    else:  # "Classic Comic Style"
        return get_comic_classical(images, captions, font, pad_image)


def get_caption_group(images_groups, captions=[]):
    caption_groups = []
    for i in range(len(images_groups)):
        length = len(images_groups[i])
        caption_groups.append(captions[:length])
        captions = captions[length:]
    if len(caption_groups[-1]) < len(images_groups[-1]):
        caption_groups[-1] = caption_groups[-1] + [""] * (len(images_groups[-1]) - len(caption_groups[-1]))
    return caption_groups


def get_comic_classical(images, captions=None, font=None, pad_image=None):
    if pad_image is None:
        raise ValueError("pad_image is None")
    images = [add_white_border(image) for image in images]
    pad_image = pad_image.resize(images[0].size, Image.LANCZOS)
    images_groups = distribute_images2(images, pad_image)
    print(images_groups)
    if captions is not None:
        captions_groups = get_caption_group(images_groups, captions)
    # print(images_groups)
    row_images = []
    for ind, img_group in enumerate(images_groups):
        row_images.append(
            get_row_image2(img_group, captions=captions_groups[ind] if captions is not None else None, font=font)
        )

    return [combine_images_vertically_with_resize(row_images)]


def get_comic_4panel(images, captions=[], font=None, pad_image=None):
    if pad_image is None:
        pad_image = Image.fromarray(np.ones_like(images[0]) * 255)
    pad_image = pad_image.resize(images[0].size, Image.LANCZOS)
    images = [add_white_border(image) for image in images]
    assert len(captions) == len(images)
    for i, caption in enumerate(captions):
        images[i] = add_caption(images[i], caption, font=font)
    images_nums = len(images)
    pad_nums = int((4 - images_nums % 4) % 4)
    images = images + [pad_image for _ in range(pad_nums)]
    comics = []
    assert len(images) % 4 == 0
    for i in range(len(images) // 4):
        comics.append(
            combine_images_vertically_with_resize(
                [
                    combine_images_horizontally(images[i * 4 : i * 4 + 2]),
                    combine_images_horizontally(images[i * 4 + 2 : i * 4 + 4]),
                ]
            )
        )

    return comics


def get_row_image(images):
    row_image_arr = []
    if len(images) > 3:
        stack_img_nums = (len(images) - 2) // 2
    else:
        stack_img_nums = 0
    while len(images) > 0:
        if stack_img_nums <= 0:
            row_image_arr.append(images[0])
            images = images[1:]
        elif len(images) > stack_img_nums * 2:
            if get_random_bool():
                row_image_arr.append(concat_images_vertically_and_scale(images[:2]))
                images = images[2:]
                stack_img_nums -= 1
            else:
                row_image_arr.append(images[0])
                images = images[1:]
        else:
            row_image_arr.append(concat_images_vertically_and_scale(images[:2]))
            images = images[2:]
            stack_img_nums -= 1
    return combine_images_horizontally(row_image_arr)


def get_row_image2(images, captions=None, font=None):
    row_image_arr = []
    if len(images) == 6:
        sequence_list = [1, 1, 2, 2]
    elif len(images) == 4:
        sequence_list = [1, 1, 2]
    else:
        raise ValueError("images nums is not 4 or 6 found", len(images))
    random.shuffle(sequence_list)
    index = 0
    for length in sequence_list:
        if length == 1:
            if captions is not None:
                images_tmp = add_caption(images[0], text=captions[index], font=font)
            else:
                images_tmp = images[0]
            row_image_arr.append(images_tmp)
            images = images[1:]
            index += 1
        elif length == 2:
            row_image_arr.append(concat_images_vertically_and_scale(images[:2]))
            images = images[2:]
            index += 2

    return combine_images_horizontally(row_image_arr)


def concat_images_vertically_and_scale(images, scale_factor=2):
    widths = [img.width for img in images]
    if not all(width == widths[0] for width in widths):
        raise ValueError("All images must have the same width.")

    total_height = sum(img.height for img in images)

    max_width = max(widths)
    concatenated_image = Image.new("RGB", (max_width, total_height))

    current_height = 0
    for img in images:
        concatenated_image.paste(img, (0, current_height))
        current_height += img.height

    new_height = concatenated_image.height // scale_factor
    new_width = concatenated_image.width // scale_factor
    resized_image = concatenated_image.resize((new_width, new_height), Image.LANCZOS)

    return resized_image


def combine_images_horizontally(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new("RGB", (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.width

    return new_im


def combine_images_vertically_with_resize(images):
    widths, heights = zip(*(i.size for i in images))

    min_width = min(widths)

    resized_images = []
    for img in images:
        new_height = int(min_width * img.height / img.width)
        resized_img = img.resize((min_width, new_height), Image.LANCZOS)
        resized_images.append(resized_img)

    total_height = sum(img.height for img in resized_images)

    new_im = Image.new("RGB", (min_width, total_height))

    y_offset = 0
    for im in resized_images:
        new_im.paste(im, (0, y_offset))
        y_offset += im.height

    return new_im


def distribute_images2(images, pad_image):
    groups = []
    remaining = len(images)
    if len(images) <= 8:
        group_sizes = [4]
    else:
        group_sizes = [4, 6]

    size_index = 0
    while remaining > 0:
        size = group_sizes[size_index % len(group_sizes)]
        if remaining < size and remaining < min(group_sizes):
            size = min(group_sizes)
        if remaining > size:
            new_group = images[-remaining : -remaining + size]
        else:
            new_group = images[-remaining:]
        groups.append(new_group)
        size_index += 1
        remaining -= size
        print(remaining, groups)
    groups[-1] = groups[-1] + [pad_image for _ in range(-remaining)]

    return groups


def distribute_images(images, group_sizes=(4, 3, 2)):
    groups = []
    remaining = len(images)

    while remaining > 0:
        for size in sorted(group_sizes, reverse=True):
            if remaining >= size or remaining == len(images):
                if remaining > size:
                    new_group = images[-remaining : -remaining + size]
                else:
                    new_group = images[-remaining:]
                groups.append(new_group)
                remaining -= size
                break
            elif remaining < min(group_sizes) and groups:
                groups[-1].extend(images[-remaining:])
                remaining = 0

    return groups


def create_binary_matrix(img_arr, target_color):
    mask = np.all(img_arr == target_color, axis=-1)
    binary_matrix = mask.astype(int)
    return binary_matrix


def preprocess_mask(mask_, h, w, device):
    mask = np.array(mask_)
    mask = mask.astype(np.float32)
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = ms.Tensor(mask)
    mask = ops.interpolate(mask, size=(h, w), mode="nearest")
    return mask


def process_sketch(canvas_data):
    binary_matrixes = []
    base64_img = canvas_data["image"]
    image_data = base64.b64decode(base64_img.split(",")[1])
    image = Image.open(BytesIO(image_data)).convert("RGB")
    im2arr = np.array(image)
    colors = [tuple(map(int, rgb[4:-1].split(","))) for rgb in canvas_data["colors"]]
    colors_fixed = []

    r, g, b = 255, 255, 255
    binary_matrix = create_binary_matrix(im2arr, (r, g, b))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_ * (r, g, b) + (1 - binary_matrix_) * (50, 50, 50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    for color in colors:
        r, g, b = color
        if any(c != 255 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r, g, b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_ * (r, g, b) + (1 - binary_matrix_) * (50, 50, 50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    visibilities = []
    colors = []
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]

    return [gr.update(visible=True), binary_matrixes, *visibilities, *colors]


def process_prompts(binary_matrixes, *seg_prompts):
    return [gr.update(visible=True), gr.update(value=" , ".join(seg_prompts[: len(binary_matrixes)]))]


def process_example(layout_path, all_prompts, seed_):
    all_prompts = all_prompts.split("***")

    binary_matrixes = []
    colors_fixed = []

    im2arr = np.array(Image.open(layout_path))[:, :, :3]
    unique, counts = np.unique(np.reshape(im2arr, (-1, 3)), axis=0, return_counts=True)
    sorted_idx = np.argsort(-counts)

    binary_matrix = create_binary_matrix(im2arr, (0, 0, 0))
    binary_matrixes.append(binary_matrix)
    binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
    colored_map = binary_matrix_ * (255, 255, 255) + (1 - binary_matrix_) * (50, 50, 50)
    colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    for i in range(len(all_prompts) - 1):
        r, g, b = unique[sorted_idx[i]]
        if any(c != 255 for c in (r, g, b)) and any(c != 0 for c in (r, g, b)):
            binary_matrix = create_binary_matrix(im2arr, (r, g, b))
            binary_matrixes.append(binary_matrix)
            binary_matrix_ = np.repeat(np.expand_dims(binary_matrix, axis=(-1)), 3, axis=(-1))
            colored_map = binary_matrix_ * (r, g, b) + (1 - binary_matrix_) * (50, 50, 50)
            colors_fixed.append(gr.update(value=colored_map.astype(np.uint8)))

    visibilities = []
    colors = []
    prompts = []
    for n in range(MAX_COLORS):
        visibilities.append(gr.update(visible=False))
        colors.append(gr.update())
        prompts.append(gr.update())

    for n in range(len(colors_fixed)):
        visibilities[n] = gr.update(visible=True)
        colors[n] = colors_fixed[n]
        prompts[n] = all_prompts[n + 1]

    return [
        gr.update(visible=True),
        binary_matrixes,
        *visibilities,
        *colors,
        *prompts,
        gr.update(visible=True),
        gr.update(value=all_prompts[0]),
        int(seed_),
    ]
