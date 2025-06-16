from PIL import Image
from transformers import AutoProcessor

import mindspore as ms

from mindone.transformers import GroundingDinoForObjectDetection


class GDINO:
    def build_model(self, ckpt_path: str | None = None, dtype=ms.float16):
        model_id = "IDEA-Research/grounding-dino-base" if ckpt_path is None else ckpt_path
        self.processor = AutoProcessor.from_pretrained(model_id)
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

        outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[k.shape[::-1] for k in images_pil],
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
