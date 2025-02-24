import os

from transformers import BertTokenizer

from mindspore import Tensor, nn, ops

from mindone.transformers import BertModel


class HunyuanClip(nn.Cell):
    """
    Hunyuan clip code copied from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/hunyuandit/pipeline_hunyuandit.py
    hunyuan's clip used BertModel and BertTokenizer, so we copy it.
    """

    def __init__(self, model_dir, max_length=77):
        super(HunyuanClip, self).__init__()

        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
        self.text_encoder = BertModel.from_pretrained(os.path.join(model_dir, "clip_text_encoder"))

    def prompts_to_tokens(self, prompts):
        text_inputs = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_attention_mask=True,
            return_tensors="np",
        )
        return Tensor(text_inputs.input_ids), Tensor(text_inputs.attention_mask)

    # @no_grad
    def construct(self, input_ids, attention_mask, with_mask=True):
        prompt_embeds = self.text_encoder(
            input_ids,
            attention_mask=attention_mask if with_mask else None,
        )

        last_hidden_state, pooler_output = (
            prompt_embeds[0],
            prompt_embeds[1] if self.text_encoder.pooler is not None else None,
        )

        return ops.stop_gradient(last_hidden_state), ops.stop_gradient(pooler_output)
