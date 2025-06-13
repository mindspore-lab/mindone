from transformers import AutoProcessor

import mindspore as ms
from mindspore.nn import no_init_parameters

from mindone.transformers import SmolVLMForConditionalGeneration

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-256M-Video-Instruct")
with no_init_parameters():
    model = SmolVLMForConditionalGeneration.from_pretrained(
        "HuggingFaceTB/SmolVLM2-256M-Video-Instruct", mindspore_dtype=ms.bfloat16
    )

# Conversation for the first image
conversation1 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://github.com/townwish4git/mindone/assets/143256262/8c25ae9a-67b1-436f-abf6-eca36738cd17",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# Conversation with two images
conversation2 = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
            },
            {"type": "image", "url": "https://cdn.britannica.com/59/94459-050-DBA42467/Skyline-Chicago.jpg"},
            {"type": "text", "text": "What is written in the pictures?"},
        ],
    }
]

# Conversation with pure text
conversation3 = [{"role": "user", "content": "who are you?"}]


conversations = [conversation1, conversation2, conversation3]
for conversation in conversations:
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="np",
    )
    for k, v in inputs.items():
        inputs[k] = ms.tensor(v, dtype=ms.bfloat16 if k != "input_ids" else ms.int32)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=100)
    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
    print(generated_texts[0])
