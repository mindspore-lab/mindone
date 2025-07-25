# Adapted from https://github.com/Gen-Verse/MMaDA/blob/main/parquet/my_dataset.py

import glob
import json
import logging
import os
import random

import pyarrow.parquet as pq
from parquet.loader import create_dataloader
from PIL import Image

import mindspore.dataset.vision as transforms
from mindspore.dataset.vision import Inter

logger = logging.getLogger(__name__)


class RefinedWebDataset:
    def __init__(
        self,
        data_path,
        rank: int = 0,
        world_size: int = 1,
        shuffle=True,
        repeat=True,
        buffer_size=1000,
        max_length=8000,
        num_workers=1,
    ):
        super().__init__()
        self.files = sorted(glob.glob(data_path))
        print(f"{len(self.files)} files detected in {data_path}")
        assert len(self.files) > 0, f"No files found in {data_path}"
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.num_workers = num_workers

        if len(self.files) < self.world_size:
            logger.warning(
                f"Expect to have more files to be sharded for data parallel, but got {len(self.files)} files and {self.world_size} devices."
            )
            logger.warning("Replicating the files...")
            self.files = self.files * (self.world_size // len(self.files))
            if len(self.files) < self.world_size:
                self.files += self.files

        self.files = self.files[self.rank :: self.world_size]

        # compute length ahead
        self.length = sum(len(pq.read_table(file, columns=["content"])) for file in self.files)

    def __len__(self):
        return self.length

    def read_parquet_file(self, file_path):
        table = pq.read_table(file_path, columns=["content"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            yield {"content": row["content"]}

    def __iter__(self):
        while True:
            file_list = self.files
            if self.shuffle:
                random.shuffle(file_list)

            for file in file_list:
                data_generator = self.read_parquet_file(file)
                buffer = []

                for data in data_generator:
                    text = data["content"].replace("\n", "")
                    if len(text) > self.max_length:
                        start_index = random.randint(0, len(text) - self.max_length - 1)
                        selected_text = text[start_index : start_index + self.max_length]
                    else:
                        selected_text = text

                    buffer.append({"input_ids": selected_text})

                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for item in buffer:
                            yield item["input_ids"]
                        buffer = []

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item["input_ids"]

            if not self.repeat:
                break


class ChatDataset:
    def __init__(
        self,
        data_path,
        rank: int = 0,
        world_size: int = 1,
        shuffle=True,
        repeat=True,
        buffer_size=1000,
        max_length=8000,
        num_workers=1,
        tokenizer=None,
    ):
        super().__init__()
        self.files = sorted(glob.glob(data_path))
        print(f"{len(self.files)} files detected in {data_path}")
        assert len(self.files) > 0, f"No files found in {data_path}"
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        self.files = self.files[self.rank :: self.world_size]

    def read_parquet_file(self, file_path):
        table = pq.read_table(file_path, columns=["content"])
        df = table.to_pandas()
        for _, row in df.iterrows():
            yield {"content": row["content"]}

    def __iter__(self):
        while True:
            file_list = self.files
            if self.shuffle:
                random.shuffle(file_list)

            for file in file_list:
                data_generator = self.read_parquet_file(file)
                buffer = []

                for data in data_generator:
                    text = data["content"]
                    if self.tokenizer is None:
                        if len(text) > self.max_length:
                            start_index = random.randint(0, len(text) - self.max_length - 1)
                            selected_text = text[start_index : start_index + self.max_length]
                        else:
                            selected_text = text
                    else:
                        if len(self.tokenizer(text)["input_ids"]) < self.max_length:
                            selected_text = text
                        else:
                            continue

                    buffer.append({"input_ids": selected_text})

                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for item in buffer:
                            yield item["input_ids"]
                        buffer = []

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item["input_ids"]

            if not self.repeat:
                break


class R2iDataset:
    def __init__(
        self,
        data_path,
        rank: int = 0,
        world_size: int = 1,
        shuffle=True,
        repeat=True,
        buffer_size=1000,
        max_length=8000,
        num_workers=1,
        resolution=256,
        tokenizer=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.resolution = resolution

    def __iter__(self):
        while True:
            subdirs = sorted([d for d in glob.glob(os.path.join(self.data_path, "*")) if os.path.isdir(d)])

            if self.shuffle:
                random.shuffle(subdirs)

            subdirs = subdirs[self.rank :: self.world_size]

            subdirs = ["/data_storage/lbw/datasets/laion-aesthetics-12m-images-2/00000"]

            for subdir in subdirs:
                all_files = glob.glob(os.path.join(subdir, "*.*"))
                base_names = set()

                for file_path in all_files:
                    base_name = os.path.splitext(os.path.basename(file_path))[0]
                    base_names.add(base_name)

                base_names = list(base_names)
                if self.shuffle:
                    random.shuffle(base_names)

                buffer = []

                for base_name in base_names:
                    jpg_path = os.path.join(subdir, f"{base_name}.jpg")
                    caption_path = os.path.join(subdir, f"{base_name}.caption")
                    shortcaption_path = os.path.join(subdir, f"{base_name}.shortcaption")

                    if not os.path.exists(jpg_path):
                        continue

                    try:
                        image = Image.open(jpg_path).convert("RGB")

                        caption = ""
                        if os.path.exists(caption_path):
                            with open(caption_path, "r", encoding="utf-8") as f:
                                caption = f.read().strip()

                        short_caption = ""
                        if os.path.exists(shortcaption_path):
                            with open(shortcaption_path, "r", encoding="utf-8") as f:
                                short_caption = f.read().strip()

                        transformed_image = image_transform_clip({"images": image}, resolution=self.resolution)[
                            "images"
                        ]

                        if self.tokenizer is not None:
                            if len(self.tokenizer(caption)["input_ids"]) > self.max_length - 2:
                                continue

                        prompt = (
                            "<|start_header_id|>user<|end_header_id|>\n"
                            "You should first think out a more detailed version of the description and then provide the user with the image. \
                                The detailed description is enclosed within <think> </think> tags, i.e. <think> detailed description here </think> image here\n"
                            f"{short_caption}"
                            "<eot_id><|start_header_id|>assistant<|end_header_id|>\n"
                            f"<think>{caption}</think>"
                        )

                        sample = {
                            "images": transformed_image,
                            "input_ids": prompt,
                        }

                        buffer.append(sample)

                        if len(buffer) >= self.buffer_size:
                            if self.shuffle:
                                random.shuffle(buffer)
                            for item in buffer:
                                yield item["images"], item["input_ids"]
                            buffer = []

                    except Exception as e:
                        print(f"Error processing {jpg_path}: {e}")
                        continue

                if buffer:
                    if self.shuffle:
                        random.shuffle(buffer)
                    for item in buffer:
                        yield item["images"], item["input_ids"]

            if not self.repeat:
                break


class VQADataset:
    def __init__(
        self,
        json_path: str,
        image_root: str,
        tokenizer=None,
        rank: int = 0,
        world_size: int = 1,
        shuffle: bool = True,
        repeat: bool = True,
        buffer_size: int = 100,
        resolution: int = 256,
        max_length: int = 8000,
        num_workers: int = 1,
        image_transform_method: str = "squash",
    ):
        super().__init__()
        self.json_path = json_path
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.repeat = repeat
        self.buffer_size = buffer_size
        self.resolution = resolution
        self.max_length = max_length
        self.num_workers = num_workers
        self.image_transform_method = image_transform_method
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.json_path}")
            self.list_data_dict = []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.json_path}")
            self.list_data_dict = []
        else:
            self.list_data_dict = [item for item in raw_data if "image" in item and "conversations" in item]
        self.list_data_dict = self.list_data_dict[self.rank :: self.world_size]

    def __iter__(self):
        sot_token = "<|startoftext|>"
        assistant_prompt_suffix = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        while True:
            current_data_list = list(self.list_data_dict)
            if self.shuffle:
                random.shuffle(current_data_list)
            buffer = []
            for item in current_data_list:
                image_relative_path = item.get("image")
                conversations = item.get("conversations", [])
                if not image_relative_path or not conversations or len(conversations) < 2:
                    continue
                num_total_messages = len(conversations)
                if num_total_messages % 2 != 0:
                    conversations = conversations[:-1]
                    num_total_messages -= 1
                    if num_total_messages < 2:
                        continue
                num_turns = num_total_messages // 2
                if num_turns == 0:
                    continue
                selected_num_turns = random.randint(1, num_turns)
                selected_conversations = conversations[: selected_num_turns * 2]
                image_path = os.path.join(self.image_root, image_relative_path)
                try:
                    image = Image.open(image_path).convert("RGB")
                    if self.image_transform_method == "squash":
                        transformed_image = image_transform_squash({"images": image}, resolution=self.resolution)[
                            "images"
                        ]
                    elif self.image_transform_method == "pad":
                        transformed_image = image_transform_pad({"images": image}, resolution=self.resolution)["images"]
                    else:
                        transformed_image = image_transform_clip({"images": image}, resolution=self.resolution)[
                            "images"
                        ]
                    first_human_message = selected_conversations[0]["value"]
                    processed_message = first_human_message.replace("<image>\n", "").replace("\n<image>", "")
                    current_selection_messages = list(selected_conversations)
                    current_selection_messages[0] = dict(current_selection_messages[0])
                    current_selection_messages[0]["value"] = processed_message
                    messages = []
                    for turn in current_selection_messages:
                        role = "user" if turn["from"] == "human" else "assistant"
                        messages.append({"role": role, "content": turn["value"]})
                    formatted_text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if formatted_text.startswith(sot_token):
                        formatted_text = formatted_text[len(sot_token) :]
                    if formatted_text.endswith(assistant_prompt_suffix):
                        formatted_text = formatted_text[: -len(assistant_prompt_suffix)]
                    token_ids = self.tokenizer(formatted_text)["input_ids"]
                    if len(token_ids) > self.max_length:
                        continue
                    sample = {
                        "images": transformed_image,
                        "input_ids": formatted_text,
                    }
                    buffer.append(sample)
                    if len(buffer) >= self.buffer_size:
                        if self.shuffle:
                            random.shuffle(buffer)
                        for buf_item in buffer:
                            yield buf_item["images"], buf_item["input_ids"]
                        buffer = []
                except FileNotFoundError:
                    print(f"Warning: Image file not found at {image_path}, skipping item.")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing item with image {image_path}: {e}, skipping.")
                    continue
            if buffer:
                if self.shuffle:
                    random.shuffle(buffer)
                for buf_item in buffer:
                    yield buf_item["images"], buf_item["input_ids"]
            if not self.repeat:
                break


def image_transform_clip(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize(resolution, interpolation=Inter.BICUBIC)(image)
    image = transforms.CenterCrop((resolution, resolution))(image)
    image = transforms.ToTensor()(image)
    image = (image - 0.5) / 0.5
    sample["images"] = image
    return sample


def image_transform_squash(sample, resolution=256):
    image = sample["images"]
    image = transforms.Resize((resolution, resolution), interpolation=Inter.BICUBIC)(image)
    image = transforms.ToTensor()(image)
    image = (image - 0.5) / 0.5
    sample["images"] = image
    return sample


def image_transform_pad(sample, resolution=256, fill_color=(255, 255, 255)):
    image = sample["images"]
    w, h = image.size
    if w == h:
        padded_image = image
    elif w < h:
        padding_needed = h - w
        padding_left = padding_needed // 2
        padding_right = padding_needed - padding_left
        pad_transform = transforms.Pad((padding_left, 0, padding_right, 0), fill=fill_color, padding_mode="constant")
        padded_image = pad_transform(image)
    else:
        padding_needed = w - h
        padding_top = padding_needed // 2
        padding_bottom = padding_needed - padding_top
        pad_transform = transforms.Pad((0, padding_top, 0, padding_bottom), fill=fill_color, padding_mode="constant")
        padded_image = pad_transform(image)
    image_resized = transforms.Resize((resolution, resolution), interpolation=Inter.BICUBIC)(padded_image)
    image_tensor = transforms.ToTensor()(image_resized)
    image_normalized = (image_tensor - 0.5) / 0.5
    sample["images"] = image_normalized
    return sample


if __name__ == "__main__":
    data_path = "train_datasets/falcon-refinedweb/data/*parquet"
    dataset = RefinedWebDataset(
        data_path=data_path,
        max_length=8000,
        buffer_size=0,
    )

    train_dataloader = create_dataloader(dataset, column_names=["input_ids"], batch_size=1, sampler=None, num_workers=1)
    train_dataloader = train_dataloader.create_dict_iterator(num_epochs=1, output_numpy=True)
    print("Starting data loading test...")
    for i, batch in enumerate(train_dataloader):
        if i == 0:
            print(batch)
            print(f"Batch size: {len(batch['input_ids'])}")
            print(f"First sample length: {len(batch['input_ids'][0])}")
        if i >= 5:
            break
    print("Data loading test complete")
