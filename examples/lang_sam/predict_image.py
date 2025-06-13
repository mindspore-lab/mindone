"""
Before running this script, please download the images via:

mkdir -p assets
wget -P assets https://raw.githubusercontent.com/luca-medeiros/lang-segment-anything/refs/heads/main/assets/car.jpeg

Then run `python predict_image.py`
"""
from lang_sam import LangSAM
from PIL import Image

model = LangSAM()
image_pil = Image.open("./assets/cars.jpeg").convert("RGB")
text_prompt = "wheel."
results = model.predict([image_pil], [text_prompt])
