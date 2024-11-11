from compel import Compel
from transformers import CLIPTokenizer

import mindspore as ms

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# whether to set pad_token to "!"?

ms.set_context(mode=ms.PYNATIVE_MODE)
# ms.set_context(mode=ms.GRAPH_MODE)
import sys

SDV2_ROOTDIR = "../stable_diffusion_v2"
sys.path.append(SDV2_ROOTDIR)
from ldm.util import instantiate_from_config, load_pretrained_model

PATHs = {
    "2.0": {
        "model_config": f"{SDV2_ROOTDIR}/configs/v2-inference.yaml",
        "pretrained_ckpt": f"{SDV2_ROOTDIR}/models/sd_v2_base-57526ee4.ckpt",
    },
    "1.5": {
        "model_config": f"{SDV2_ROOTDIR}/configs/v1-inference.yaml",
        "pretrained_ckpt": f"{SDV2_ROOTDIR}/models/sd_v1.5-d0ab7146.ckpt",
    },
}


def test_sdv2(model_config, pretrained_ckpt):
    latent_diffusion_with_loss = instantiate_from_config(model_config)
    load_pretrained_model(pretrained_ckpt, latent_diffusion_with_loss)
    text_encoder = latent_diffusion_with_loss.cond_stage_model
    text_encoder.tokenizer = tokenizer  # replace tokenizer by CLIPTokenizer
    text_encoder.set_train(False)
    for param in text_encoder.get_parameters():
        param.requires_grad = False
    compel = Compel(tokenizer=tokenizer, text_encoder=text_encoder)
    prompts = ["a cat playing with a <ball>-- in the forest"]
    prompt_embeds = compel(prompts)
    return prompt_embeds


if __name__ == "__main__":
    for version in ["2.0", "1.5"]:
        test_sdv2(**PATHs[version])
