import argparse
import os

import pandas as pd
import mindspore as ms
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size, init

from tqdm import tqdm

from pipeline.captioning.pllava.tasks.eval.model_utils import load_pllava
from pipeline.captioning.pllava.tasks.eval.eval_utils import (
    ChatPllava,
    Conversation,
    conv_templates
)

SYSTEM = ("Describe this video. Pay attention to all objects in the video. "
           "The description should be useful for AI to re-generate the video. "
           "The description should be no more than six sentences. "
           "Here are some examples of good descriptions: "
           "1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. "
           "She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. "
           "She wears sunglasses and red lipstick. She walks confidently and casually. "
           "The street is damp and reflective, creating a mirror effect of the colorful lights. "
           "Many pedestrians walk about. "
           "2. Several giant wooly mammoths approach treading through a snowy meadow, "
           "their long wooly fur lightly blows in the wind as they walk, "
           "snow covered trees and dramatic snow capped mountains in the distance, "
           "mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, "
           "the low camera view is stunning capturing the large furry mammal with beautiful photography, "
           "depth of field. "
           "3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. "
           "The crashing blue waters create white-tipped waves, "
           "while the golden light of the setting sun illuminates the rocky shore. "
           "A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. "
           "The steep drop from the road down to the beach is a dramatic feat, "
           "with the cliff’s edges jutting out over the sea. "
           "This is a view that captures the raw beauty of the coast "
          "and the rugged landscape of the Pacific Coast Highway.")

conv_template = Conversation(
    system= SYSTEM,
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=(" ", "</s>"),
    mm_token="<image>",
)

# **REMARK**: image captioning is NOT supported by PLLaVA by default
class VideoTextDataset:
    def __init__(self, meta_path, num_frames):
        self.meta_path = meta_path
        self.meta = pd.read_csv(meta_path)
        self.num_frames = num_frames

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        path = sample['path']
        return path, index

    def __len__(self):
        return len(self.meta)

def get_response(chat, chat_state, img_list, question, num_beams, temperature, max_new_tokens):
    chat_state = chat.ask(question, chat_state, SYSTEM)
    llm_message, output_tokens, chat_state = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        temperature=temperature
    )
    return output_tokens

def pad_tensor(tensor, max_length=4096, pad_value=0):
    return ops.pad(tensor, (0, max_length - tensor.shape[0]), value = pad_value)

def parse_args():
    parser = argparse.ArgumentParser()
    # input paths
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="pretrained_models/pllava")
    # PLLaVA parameters
    parser.add_argument("--pooling_shape", type=str, required=False, default=None)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--weight_dir", type=str, default=None)
    parser.add_argument("--conv_mode", type=str, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--question", type=str, default="Describe the video in details.")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--pad_length", type=int, default=4096)
    # dataset management
    parser.add_argument("--bs", type=int, default=1, help="Batch size")
    parser.add_argument("--skip_if_existing", action='store_true', help="Skip processing if output already exists")
    return parser.parse_args()


def main():
    args = parse_args()

    meta_path = args.meta_path
    if not os.path.exists(meta_path):
        print(f"Meta file '{meta_path}' not found. Exit.")
        exit()

    wo_ext, ext = os.path.splitext(meta_path)
    out_path = f"{wo_ext}_caption{ext}"
    if args.skip_if_existing and os.path.exists(out_path):
        print(f"Output meta file '{out_path}' already exists. Exit.")
        exit()

    # TODO: or if I should put this one at the beginning?
    # Ascend support only, currently PyNative mode
    ms.set_context(jit_config=dict(jit_level="O1"), device_target="Ascend")
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    init()

    # initialize model, chat inference
    print('Initializing PLLaVA')
    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        args.num_frames,
        use_lora=args.use_lora,
        weight_dir=args.weight_dir,
        lora_alpha=args.lora_alpha,
        pooling_shape = args.pooling_shape if args.pooling_shape is not None else (16, 12, 12),
    )
    chat = ChatPllava(model, processor)
    if args.conv_mode is None:
        INIT_CONVERSATION = conv_template.copy()
    else:
        INIT_CONVERSATION = conv_templates[args.conv_mode]

    raw_dataset = VideoTextDataset(args.meta_path, num_frames=args.num_frames)
    rank_id = get_rank()
    rank_size = get_group_size()
    dataset = ds.GeneratorDataset(source=raw_dataset, column_names=['video_path', 'index'], shuffle=False,
                                    num_shards=rank_size, shard_id=rank_id)
    dataset = dataset.batch(args.bs, drop_remainder=False)
    iterator = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)

    # captioning
    indices_list = []
    token_length_list = []
    token_list = []
    model.set_train(False)
    for batch in tqdm(iterator):
        video_paths = batch['video_path']
        print(video_paths)
        indices = batch['index']

        for video_path, idx in zip(video_paths, indices):
            chat_state = INIT_CONVERSATION.copy()
            img_list = []
            llm_message, img_list, chat_state = chat.upload_video(video_path, chat_state, img_list)
            if img_list is None:
                # skip if error occurs when handling a video, in which case the caption for this video would be empty
                indices_list.append(idx) # these three lines fix AllGather issue - consistent Tensor length
                token_length_list.append(0)  # indicate no tokens generated
                token_list.append(pad_tensor(ms.Tensor([0], dtype=ms.int64), args.pad_length))
                continue
            out_tokens = get_response(chat, chat_state, img_list, args.question,
                                      args.num_beams, args.temperature, args.max_new_tokens)
            out_tokens = ops.squeeze(ms.Tensor(out_tokens, dtype=ms.int64), axis = 0)
            token_length_list.append(len(out_tokens))
            # pad out_tokens to the same length
            out_tokens = pad_tensor(out_tokens, args.pad_length)
            indices_list.append(idx)
            token_list.append(out_tokens)

    allgather = ops.AllGather()
    indices_list = ms.Tensor(indices_list, dtype=ms.int64)
    indices_list = allgather(indices_list).asnumpy().tolist()
    token_length_list = ms.Tensor(token_length_list, dtype=ms.int64)
    token_length_list = allgather(token_length_list).asnumpy().tolist()
    token_list = ops.concat(token_list, axis=0)
    token_list = allgather(token_list)

    if rank_id == 0:
        meta_local = raw_dataset.meta
        # recover tokens after gathering
        start_idx = 0
        separated_tokens = []
        for length in token_length_list:
            end_idx = start_idx + length
            # skip erroneous video - marked by length = 0
            if length == 0:
                # process here append something
                separated_tokens.append(None)
            else:
                separated_tokens.append(token_list[start_idx:end_idx])
            start_idx += args.pad_length
        # get text
        decoded_texts = []
        for idx, output_token in enumerate(separated_tokens):
            if output_token is None:
                # assign a default caption for placeholders
                decoded_texts.append(" ")
            else:
                decoded_text = chat.processor.batch_decode(
                    ops.unsqueeze(output_token, dim=0),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]
                decoded_texts.append(decoded_text)
        # clean up
        for i in range(len(decoded_texts)):
            if chat_state.roles[-1] == "<|im_start|>assistant\n":
                split_tag = "<|im_start|> assistant\n"
            else:
                split_tag = chat_state.roles[-1]
            decoded_texts[i] = decoded_texts[i].split(split_tag)[-1]
            ending = chat_state.sep if isinstance(chat_state.sep, str) else chat_state.sep[1]
            decoded_texts[i] = decoded_texts[i].removesuffix(ending).strip()
            chat_state.messages[-1][1] = decoded_texts[i]

        # save to csv
        meta_local.loc[indices_list, 'text'] = decoded_texts
        meta_local.to_csv(out_path, index=False)
        print(meta_local)
        print(f"New meta with PLLaVA Caption saved to '{out_path}'.")

if __name__ == "__main__":
    main()


