import logging
import os
import sys

from utils.model_utils import remove_pname_prefix

import mindspore as ms

sys.path.append("../stable_diffusion_v2")
from ldm.modules.encoders.modules import FrozenCLIPEmbedder

logger = logging.getLogger(__name__)


def initiate_clip_text_encoder(use_fp16: bool = True, ckpt_path: str = None, trainable: bool = False):
    text_encoder = FrozenCLIPEmbedder(
        use_fp16=use_fp16,
        tokenizer_name="BpeTokenizer",
        context_length=77,
        vocab_size=49408,
        output_dim=768,
        width=768,
        layers=12,
        heads=12,
        epsilon=1e-5,
        use_quick_gelu=True,
    )

    if ckpt_path is not None and len(ckpt_path) > 0:
        assert os.path.exists(ckpt_path), f"{ckpt_path} does not exist!"
        logger.info(f"Loading {ckpt_path} params into CLIP Text Encoder...")
        param_dict = ms.load_checkpoint(ckpt_path)
        # in case it's stable diffusion ckpt, remove prefix
        param_dict = remove_pname_prefix(param_dict, prefix="model.cond_stage_model.")
        param_not_load, _ = ms.load_param_into_net(
            text_encoder,
            param_dict,
        )
        assert len(param_not_load) == 0, f"params should be all loaded, but found {param_not_load} are not loaded."
    text_encoder.set_train(trainable)
    for param in text_encoder.get_parameters():
        param.requires_grad = trainable

    return text_encoder


def initiate_t5_text_encoder():
    raise NotImplementedError
