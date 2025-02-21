import sys

import numpy as np

pt_code_path = "/Users/Samit/Data/Work/HW/ms_kit/aigc/Janus"
sys.path.append(pt_code_path)
import torch
from janus.models.processing_vlm import VLChatProcessor
from janus.models.vq_model import VQ_16
from janus.utils.io import load_pil_images
from torch import Tensor

np.random.seed(42)
torch.manual_seed(42)


def test_decode(pt_ckpt=None, pt_np=None, dtype=torch.float32):
    if pt_np:
        pt_data = np.load(pt_np)
        z = pt_data["quant"]

    vq = VQ_16()
    if pt_ckpt is not None:
        sd = torch.load(pt_ckpt)
        pnames = list(sd.keys())
        for p in pnames:
            if "gen_vision_model" not in p:
                sd.pop(p)
            else:
                # remove prefix
                new_pname = p.replace("gen_vision_model.", "")
                sd[new_pname] = sd.pop(p)
        vq.load_state_dict(sd)

    out = vq.decode(Tensor(z))
    np.savez(
        "tests/vq_dec_io.npz",
        quant=z,
        dec=out.detach().cpu().numpy(),
    )

    print(out.shape)
    print(out.sum(), out.std())

    return out.detach().cpu().numpy()


def test_vlm_proc():
    # specify the path to the model
    model_path = "ckpts/Janus-Pro-1B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)

    question = "explain this meme"
    image = "images/doge.png"

    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)

    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    )  # .to(vl_gpt.device)

    print(prepare_inputs)


if __name__ == "__main__":
    # test_decode("ckpts/Janus-Pro-1B/pytorch_model.bin")
    test_vlm_proc()
