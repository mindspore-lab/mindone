from argparse import ArgumentParser
import mindspore as ms
import time
from tasks.eval.model_utils import load_pllava
from tasks.eval.eval_utils import (
    ChatPllava,
    conv_plain_v1,
    Conversation,
    conv_templates
)

ms.set_context(pynative_synchronize = True, jit_config = dict(jit_level = "O1"))

SYSTEM = """You are a powerful Video Magic ChatBot, a large vision-language assistant. 
You are able to understand the video content that the user provides and assist the user in a video-language related task.
The user might provide you with the video and maybe some extra noisy information to help you out or ask you a question. Make use of the information in a proper way to be competent for the job.
### INSTRUCTIONS:
1. Follow the user's instruction.
2. Be critical yet believe in yourself.
"""

INIT_CONVERSATION: Conversation = conv_plain_v1.copy()

def init_model(args):
    print('Initializing PLLaVA')
    model, processor = load_pllava(
        args.pretrained_model_name_or_path,
        args.num_frames,
        use_lora=args.use_lora,
        weight_dir=args.weight_dir,
        lora_alpha=args.lora_alpha)
    chat = ChatPllava(model, processor)
    return chat

def process_input(args, chat):
    chat_state = INIT_CONVERSATION.copy()
    img_list = []

    if args.video:
        llm_message, img_list, chat_state = chat.upload_video(args.video, chat_state, img_list)
    elif args.image:
        llm_message, img_list, chat_state = chat.upload_img(args.image, chat_state, img_list)
    else:
        raise ValueError("You must provide either an image or video file.")

    return llm_message, chat_state, img_list

def get_response(chat, chat_state, img_list, question, num_beams, temperature):
    chat_state = chat.ask(question, chat_state, SYSTEM)
    start_time = time.time()
    llm_message, llm_token, chat_state = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=200,
        num_beams=num_beams,
        temperature=temperature
    )
    end_time = time.time()
    time_elapsed = end_time - start_time
    return llm_message, llm_token, time_elapsed


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=False, default='./models/pllava-7b')
    parser.add_argument("--num_frames", type=int, required=False, default=8)
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--use_lora", action='store_true')
    parser.add_argument("--weight_dir", type=str, required=False, default=None)
    parser.add_argument("--conv_mode", type=str, required=False, default="plain")
    parser.add_argument("--lora_alpha", type=int, required=False, default=None)
    parser.add_argument("--video", type=str, help="Path to the video file", default="video.mp4")
    parser.add_argument("--question", type=str, help="Question to ask the model", required=False,
                        default="What is shown in this video?")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search numbers")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    chat = init_model(args)
    INIT_CONVERSATION = conv_templates[args.conv_mode]

    llm_message, chat_state, img_list = process_input(args, chat)
    response, tokens, time_elapsed = get_response(chat, chat_state, img_list, args.question, args.num_beams, args.temperature)
    if args.benchmark:
        # run again, use the result from second round
        response, tokens, time_elapsed = get_response(chat, chat_state, img_list, args.question, args.num_beams,
                                                      args.temperature)
        print(f"Tokens length: {tokens.shape[1]}")
        print(f"Time elapsed: {time_elapsed:.4f}")
        print(f'tokens per second: {(tokens.shape[1] / time_elapsed):.4f}')
    print(f"Response: {response}")
