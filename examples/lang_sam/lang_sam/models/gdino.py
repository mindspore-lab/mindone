# Adapted from https://github.com/luca-medeiros/lang-segment-anything/blob/main/lang_sam/models/gdino.py
from PIL import Image

import mindspore as ms

from mindone.transformers import GroundingDinoForObjectDetection, GroundingDinoProcessor


class GDINO:
    def build_model(self, ckpt_path: str | None = None, dtype=ms.float32):
        model_id = "IDEA-Research/grounding-dino-base" if ckpt_path is None else ckpt_path
        self.processor = GroundingDinoProcessor.from_pretrained(model_id)
        self.dtype = dtype
        self.model = GroundingDinoForObjectDetection.from_pretrained(model_id, mindspore_dtype=dtype)

    def predict(
        self,
        images_pil: list[Image.Image],
        texts_prompt: list[str],
        box_threshold: float,
        text_threshold: float,
    ) -> list[dict]:
        for i, prompt in enumerate(texts_prompt):
            if prompt[-1] != ".":
                texts_prompt[i] += "."
        inputs = self.processor(images=images_pil, text=texts_prompt, return_tensors="np")
        inputs = {k: ms.Tensor(inputs[k]) for k in inputs.keys()}

        inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.size[::-1] for k in images_pil],
        )
        return results


if __name__ == "__main__":
    gdino = GDINO()
    gdino.build_model()
    out = gdino.predict(
        [Image.open("./assets/car.jpeg"), Image.open("./assets/car.jpeg")],
        ["wheel", "wheel"],
        0.3,
        0.25,
    )
    print(out)
