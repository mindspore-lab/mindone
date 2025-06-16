"""
Before running this script, please download the images via:

mkdir -p assets
wget -P assets https://raw.githubusercontent.com/luca-medeiros/lang-segment-anything/refs/heads/main/assets/car.jpeg

Then run `python predict_image.py`
"""
import numpy as np
from lang_sam import LangSAM
from lang_sam.utils import draw_image
from PIL import Image


def main():
    model = LangSAM(sam_type="sam2.1_hiera_small", ckpt_path="./checkpoints//sam2.1_hiera_small.pt")
    image_pil = Image.open("./assets/cars.jpeg").convert("RGB")
    text_prompt = "wheel."
    results = model.predict([image_pil], [text_prompt])
    results = results[0]

    if not len(results["masks"]):
        print("No masks detected!")
        return

    # Draw results on the image
    image_array = np.asarray(image_pil)
    output_image = draw_image(
        image_array,
        results["masks"],
        results["boxes"],
        results["scores"],
        results["labels"],
    )
    output_image = Image.fromarray(np.uint8(output_image)).convert("RGB")
    # save image
    output_image_path = "./assets/cars_with_mask.png"
    output_image.save(output_image_path)
    return output_image


if __name__ == "__main__":
    main()
