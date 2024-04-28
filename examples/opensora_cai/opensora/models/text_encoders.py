import logging
import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sdv2_lib_path = os.path.abspath(os.path.join(__dir__, "../../../stable_diffusion_v2"))
sys.path.insert(0, sdv2_lib_path)

import mindspore as ms

from .t5 import T5Embedder
from opensora.utils.model_utils import remove_pname_prefix
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


def get_text_encoder_and_tokenizer(name, ckpt_path):
    if name == "clip":
        logger.info("CLIP text encoder init")
        text_encoder = initiate_clip_text_encoder(
            use_fp16=True,  # TODO: set by config file
            ckpt_path=ckpt_path,
            trainable=False,
        )
        tokenizer = text_encoder.tokenizer
    elif name == "t5":
        logger.info("T5 init")
        text_encoder = T5Embedder(cache_dir=ckpt_path, pretrained_ckpt=os.path.join(ckpt_path, "model.ckpt"))
        tokenizer = text_encoder.tokenizer
    return text_encoder, tokenizer
