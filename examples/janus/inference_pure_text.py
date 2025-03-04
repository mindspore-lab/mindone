import urllib3
import os, sys
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import mindspore as ms

__dir__ = os.path.dirname(os.path.abspath(__file__))
mindone_lib_path = os.path.abspath(os.path.join(__dir__, "../../"))
sys.path.insert(0, mindone_lib_path)

from transformers import AutoConfig, AutoTokenizer
from mindone.transformers import LlamaForCausalLM
from janus.models import VLChatProcessor

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/Janus-Pro-1B")

system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )
input_text = "Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?"
conversations = [
            {
                "role": "<|User|>",
                "content": input_text,
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
vl_chat_processor = VLChatProcessor = VLChatProcessor.from_pretrained("deepseek-ai/Janus-Pro-1B")
sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversations,
    system_prompt=system_prompt,
)

inputs = tokenizer(sft_format)

for k in inputs:
    inputs[k] = ms.Tensor(inputs[k])
    inputs[k] = inputs[k].unsqueeze(dim=0)

config = AutoConfig.from_pretrained("deepseek-ai/Janus-Pro-1B")
language_config = config.language_config
language_config._attn_implementation = "eager"
model = LlamaForCausalLM.from_pretrained(
    "results_graphs/checkpoint-10000",
    trust_remote_code = True,
)

inputs_embeds = model.get_input_embeddings()(inputs["input_ids"])

outputs = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=inputs["attention_mask"],
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=1024,
        do_sample=False,
        use_cache=True,
    )

answer = tokenizer.decode(outputs[0].asnumpy().tolist(), skip_special_tokens=True)

print("****************reuslts****************")
print(answer)
