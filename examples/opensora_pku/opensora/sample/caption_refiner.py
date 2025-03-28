from mindnlp.transformers import AutoModelForCausalLM, AutoTokenizer

import mindspore as ms

TEMPLATE = """
Refine the sentence: \"{}\" to contain subject description, action, scene description. " \
"(Optional: camera language, light and shadow, atmosphere) and conceive some additional actions to make the sentence more dynamic. " \
"Make sure it is a fluent sentence, not nonsense.
"""


class OpenSoraCaptionRefiner:
    def __init__(self, caption_refiner, dtype):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(caption_refiner, ms_dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(caption_refiner, ms_dtype=dtype)

    def get_refiner_output(self, prompt):
        prompt = TEMPLATE.format(prompt)
        messages = [{"role": "system", "content": "You are a caption refiner."}, {"role": "user", "content": prompt}]
        input_ids = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([input_ids], return_tensors="np")
        generated_ids = self.model.generate(ms.Tensor(model_inputs.input_ids), max_new_tokens=512)
        generated_ids = [
            output_ids[len(input_ids) :] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


if __name__ == "__main__":
    pretrained_model_name_or_path = "LanguageBind/Open-Sora-Plan-v1.3.0/prompt_refiner/"
    caption_refiner = OpenSoraCaptionRefiner(pretrained_model_name_or_path, dtype=ms.float16)
    prompt = "a video of a girl playing in the park"
    response = caption_refiner.get_refiner_output(prompt)
    print(response)
