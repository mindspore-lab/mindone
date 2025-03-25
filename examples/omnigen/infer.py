import os
import time

from omnigen import OmniGenPipeline

import mindspore as ms

ms.set_context(mode=1, device_target="Ascend", jit_config={"jit_level": "O0"})
OUTPUT_PATH = "./final_inference_test"
os.makedirs(OUTPUT_PATH, exist_ok=True)
# Load the OmniGen pipeline

pipe = OmniGenPipeline.from_pretrained("./pretrained_model")

# # 1. Text to Image
prompts = [
    "A vintage camera placed on the ground, ejecting a swirling cloud of Polaroid-style photographs into the air. "
    "The photos, showing landscapes, wildlife, and travel scenes, seem to defy gravity as they float upwards.",
    "A curly-haired man in a red shirt is drinking tea.",
]
for i, prompt in enumerate(prompts):
    t1 = time.time()
    images = pipe(
        prompt=prompts[i],
        height=1024,
        width=1024,
        guidance_scale=2.5,
        separate_cfg_infer=False,
        seed=0,
        use_kv_cache=False,
        dtype=ms.bfloat16,
    )
    print("Finish inference Task 1 in ", time.time() - t1)
    images[0].save(f"{OUTPUT_PATH}/text_to_image_{i}.png")
# 2. Subject-driven Generation or Identity-Preserving Generation
input_images = ["./imgs/test_cases/zhang.png"]
prompt = "The woman in <img><|image_1|></img> waves her hand happily in the crowd"
t1 = time.time()
images = pipe(
    prompt=prompt,
    input_images=input_images,
    height=1024,
    width=1024,
    guidance_scale=2.5,
    img_guidance_scale=1.8,
    seed=42,
    use_kv_cache=True,
    dtype=ms.bfloat16,
)
print("Finish inference task 2 in ", time.time() - t1)
images[0].save(f"{OUTPUT_PATH}/subject_driven_1.png")

input_images = ["./imgs/test_cases/mckenna.jpg", "./imgs/test_cases/Amanda.jpg"]
prompt = "Two women are raising fried chicken legs in a bar. A woman is <img><|image_1|></img>. Another woman is <img><|image_2|></img>."
t1 = time.time()
images = pipe(
    prompt=prompt,
    input_images=input_images,
    height=1024,
    width=1024,
    guidance_scale=2.5,
    img_guidance_scale=1.8,
    max_input_image_size=512,
    seed=168,
    use_kv_cache=True,
    dtype=ms.bfloat16,
)
print("Finish inference task 2 in ", time.time() - t1)
images[0].save(f"{OUTPUT_PATH}/subject_driven_2.png")

# # # 3. Image-conditional Generation
input_images = ["./imgs/test_cases/control.jpg"]
prompt = "Detect the skeleton of human in this image: <img><|image_1|></img>."
t1 = time.time()
images = pipe(
    prompt=prompt,
    input_images=input_images,
    height=512,
    width=512,
    guidance_scale=2.5,
    img_guidance_scale=1.6,
    separate_cfg_infer=False,
    seed=0,
    use_kv_cache=True,
)
print("Finish inference task 3 in ", time.time() - t1)
images[0].save(f"{OUTPUT_PATH}/image_conditional_1.png")

input_images = ["./imgs/test_cases/pose.png"]
prompt = (
    "Generate a new photo using the following picture and text as conditions: <img><|image_1|><img>\n "
    "An elderly man wearing gold-framed glasses stands dignified in front of an elegant villa."
)
t1 = time.time()
images = pipe(
    prompt=prompt,
    input_images=input_images,
    height=512,
    width=512,
    guidance_scale=2.5,
    img_guidance_scale=1.6,
    seed=0,
    use_kv_cache=True,
)
print("Finish inference task 3 in ", time.time() - t1)
images[0].save(f"{OUTPUT_PATH}/image_conditional_2.png")

print("All images have been generated and saved to the output_images directory.")
