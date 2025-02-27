import os
import sys

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../../"))
sys.path.insert(0, mindone_lib_path)
sys.path.append(".")
from janus.models.processing_vlm import VLChatProcessor
from janus.utils.io import load_pil_images


def test_chat_proc():
    # specify the path to the model
    model_path = "ckpts/Janus-Pro-1B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

    question = "explain this meme"
    image = "images/doge.png"

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)

    prepare_inputs = vl_chat_processor(conversations=conversation, images=pil_images, force_batchify=True)

    print(prepare_inputs)
    print(prepare_inputs.pixel_values)


if __name__ == "__main__":
    ms.set_context(mode=0)
    test_chat_proc()
