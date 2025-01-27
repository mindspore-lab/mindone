import os
from dataclasses import dataclass
from typing import List, Tuple, Union

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessor, hash_prompt
from transformers import AutoTokenizer

import mindspore as ms
from mindspore import Tensor

from mindone.transformers import CLIPTextModel


@threestudio.register("stable-diffusion-prompt-processor")
class StableDiffusionPromptProcessor(PromptProcessor):
    @dataclass
    class Config(PromptProcessor.Config):
        pass

    cfg: Config

    # these functions are unused, kept for debugging
    def configure_text_encoder(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.pretrained_model_name_or_path, subfolder="tokenizer")
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.cfg.pretrained_model_name_or_path, subfolder="text_encoder"
        )

        for p in self.text_encoder.parameters():
            p.requires_grad = False

    def destroy_text_encoder(self) -> None:
        del self.tokenizer
        del self.text_encoder

    def get_text_embeddings(
        self, prompt: Union[str, List[str]], negative_prompt: Union[str, List[str]]
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt]
        # Tokenize text and get embeddings
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="np",
        )
        uncond_tokens = self.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="np",
        )

        with ms._no_grad():
            text_embeddings = self.text_encoder(Tensor(tokens.input_ids))[0]
            uncond_text_embeddings = self.text_encoder(Tensor(uncond_tokens.input_ids))[0]

        return text_embeddings, uncond_text_embeddings

    @staticmethod
    def spawn_func(pretrained_model_name_or_path, prompts, cache_dir):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="text_encoder",
            # low_cpu_mem_usage=False
        )

        with ms._no_grad():
            tokens = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                return_tensors="np",
            )
            text_embeddings = text_encoder(Tensor(tokens.input_ids))[0]

        for prompt, embedding in zip(prompts, text_embeddings):
            ms.save_checkpoint(
                [{"name": "prompt", "data": embedding}],
                os.path.join(
                    cache_dir,
                    f"{hash_prompt(pretrained_model_name_or_path, prompt)}.ckpt",
                ),
            )

        del text_encoder
