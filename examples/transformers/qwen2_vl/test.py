'''
This script is a quick test of image and video inputs.

'''


print("*************************************************")
print("********** Test Image Understanding *************")
print("*************************************************")

from transformers import AutoTokenizer, AutoProcessor
from mindone.transformers import Qwen2VLForConditionalGeneration #, Qwen2VLProcessor, Qwen2VLImageProcessor
# from mindone.transformers.models.qwen2_vl.qwen_vl_utils import process_vision_info # TODO change importing way
from qwen_vl_utils import process_vision_info # local
import mindspore as ms
from mindspore import Tensor
import logging
logging.basicConfig(filename='./test.log', filemode='w',  encoding='utf-8', level=logging.DEBUG)
logger = logging.getLogger(__name__)

ms.set_context(mode=ms.PYNATIVE_MODE) #default mode
# ms.set_context(mode=0) # graph mode

# print("Loading Qwen2VLForConditionalGeneration Model")
# default: Load the model on the available device(s)
# model = Qwen2VLForConditionalGeneration.from_pretrained("/home/susan/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct")
model = Qwen2VLForConditionalGeneration.from_pretrained("/home/susan/workspace/checkpoints/Qwen2-VL-7B-Instruct",
    mindspore_dtype=ms.float32) # ms.bfloat16 also okay but slow, ms.float32 faster, "auto" failed
# model = Qwen2VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2-VL-2B-Instruct")  # 2B and 7B have different dim config
# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2VLForConditionalGeneration.from_torchpretrained(
#     "Qwen/Qwen2-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )
# print(model.visual)
logger.info(model)
logger.info(model.generation_config)
print("generation_config", model.generation_config)
# breakpoint()

print("Loading AutoProcessor")
# default processer
# processor = AutoProcessor.from_pretrained("/home/susan/workspace/checkpoints/Qwen2-VL-7B-Instruct")
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("/home/susan/workspace/checkpoints/Qwen2-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
# print(processor)
logger.info(processor)

# /home/susan/workspace/Data/qwen2vl_media/demo.jpeg
# https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/qwen2_vl.jpg
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/boxes.png",
#             },
#             {"type": "text", "text": "Output the color and number of each box."},
#         ],
#     }
# ]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print("text", text)
# breakpoint()
# add_special_tokens=True, split_special_tokens=True
image_inputs, video_inputs = process_vision_info(messages) # use MS version cannot process
print("image_inputs", image_inputs)
print("video_inputs", video_inputs)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="np",
)
# print("tokenized inputs", inputs)
# attention_mask = Tensor(inputs["attention_mask"]) #https://github.com/suno-ai/bark/issues/402
# print(attention_mask)
# Convert to ms.Tensor
for key, value in inputs.items(): # by default input numpy array or list
    if isinstance(value, np.ndarray):
        inputs[key] = ms.Tensor(value)
    elif isinstance(value, list):
        inputs[key] = ms.Tensor(value)
# Inference: Generation of the output
# TODO: to include add inputs values as input, and conver to MS Tensors, current data convesion does not perform properly yet
generated_ids = model.generate(**inputs, max_new_tokens=128)
print("generated_ids.shape", generated_ids.shape)
print("generated_ids", generated_ids)
logger.info(generated_ids.shape)
logger.info(generated_ids)
logger.info(model.generation_config)
print("generation_config", model.generation_config)
# breakpoint()
# output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] # also work
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print("generated_ids_trimmed[0] len=", len(generated_ids_trimmed[0]))
print("generated_ids_trimmed", generated_ids_trimmed)
# logger.info("generated_ids_trimmed0 len", str(len(generated_ids_trimmed[0])))
logger.info(generated_ids_trimmed)
# breakpoint()
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)


print("*************************************************")
print("Input: %s"%str(messages))
print("Response:", output_text[0])
print("******** End of Test Image Understanding ********\n")

'''
print("*************************************************")
print("********** Test Video Understanding *************")
print("*************************************************")


messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/home/susan/workspace/Data/qwen2vl_media/operate_on_phone.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="np", #MS return
)
print("image_inputs", image_inputs)
print("video_inputs", video_inputs)
# Inference
generated_ids = model.generate(Tensor(inputs.input_ids), max_new_tokens=128) #MS input
print(generated_ids)
logger.info(generated_ids)
breakpoint()

# output_text = processor.batch_decode(
#     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("*************************************************")
print("Input: %s"%str(messages))
print("Response:", output_text[0])
print("******** End of Test Video Understanding ********")
'''