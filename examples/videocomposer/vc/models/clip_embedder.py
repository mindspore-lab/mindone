import sys
from typing import Optional

import numpy as np

sys.path.append("../stable_diffusion_v2/")
import os

from PIL import Image
from tools._common.clip import CLIPImageProcessor, CLIPModel, CLIPTokenizer, parse, support_list
from utils.download import download_checkpoint

import mindspore as ms
from mindspore import nn, ops

__all__ = [
    "FrozenOpenCLIPEmbedder",
    "FrozenOpenCLIPVisualEmbedder",
]

_CKPT_URL = {
    "open_clip_vit_h_14": "https://download.mindspore.cn/toolkits/mindone/videocomposer/model_weights/open_clip_vit_h_14-9bb07a10.ckpt"
}


def load_clip_model(arch, pretrained_ckpt_path, dtype):
    """
    Load CLIP model.

    Args:
        arch (str): Model architecture.
        pretrained_ckpt_path (str): Path of the pretrained checkpoint.
    Returns:
        model (CLIPModel): CLIP model.
    """
    if arch.lower() not in support_list:
        raise ValueError(f"arch {arch} is not supported")
    config_path = support_list[arch.lower()]
    # download the clip model if
    if arch.lower() != "open_clip_vit_h_14":
        raise ValueError(f"currently not support {arch.lower()}")
    if not os.path.exists(pretrained_ckpt_path):
        download_checkpoint(_CKPT_URL[arch.lower()], "model_weights/")
    if not os.path.exists(pretrained_ckpt_path):
        raise ValueError(
            f"Maybe download failed. Please download it manually from {_CKPT_URL[arch.lower()]} and place it under `model_weights/`"
        )

    config = parse(config_path, pretrained_ckpt_path)
    config.dtype = dtype
    model = CLIPModel(config)
    return model


def load_ckpt_tokenizer(tokenizer_path):
    # TODO: check diff for using defualt pad token and "!" 
    text_processor = CLIPTokenizer(tokenizer_path, pad_token="!")
    #text_processor = CLIPTokenizer(tokenizer_path)
    return text_processor


class FrozenOpenCLIPEmbedder(nn.Cell):
    def __init__(
        self,
        arch="open_clip_vit_h_14",
        pretrained_ckpt_path="./vit-h-14-laion-2b/open_clip_vit_h_14.ckpt",
        tokenizer_path="./vit-h-14-laion-2b/bpe_simple_vocab_16e6.txt.gz",
        freeze=True,
        layer="penultimate",
        use_fp16=False,
    ):
        super().__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        model = load_clip_model(arch, pretrained_ckpt_path, str(self.dtype).lower())
        del model.visual
        self.model = model
        self.layer = layer
        self.freeze = freeze
        if self.freeze:
            self.model.set_train(False)
            for name, param in self.model.parameters_and_names():
                param.requires_grad = False

        if self.layer == "last":
            layer_index = 0
        elif self.layer == "penultimate":
            layer_index = 1
            old_layers = len(self.model.transformer.resblocks)
            self.delete_last_n_layers_from_resblocks(layer_index)
            new_layers = len(self.model.transformer.resblocks)
            print(f"Transformer Resblocks Layers change from {old_layers} to {new_layers}")
        else:
            raise ValueError(f"layer {layer} is not supported")

        self.tokenizer = load_ckpt_tokenizer(tokenizer_path)

    def delete_last_n_layers_from_resblocks(self, layer_index):
        assert layer_index < len(self.model.transformer.resblocks) and layer_index >= 0
        N = len(self.model.transformer.resblocks)
        index = N - 1
        for _ in range(layer_index):
            del self.model.transformer.resblocks[index]
            index -= 1
        return

    def process_text(self, text_prompt):
        return ms.Tensor(self.tokenizer(text_prompt, padding="max_length", max_length=77)["input_ids"])

    def construct(self, token_ids: ms.Tensor):
        text_features = self.get_text_features(token_ids)
        return text_features

    def encode(self, text):
        if isinstance(text, str):
            text = [text]
        token_ids = self.process_text(text)
        text_features = self.construct(token_ids)
        return text_features

    def get_text_features(self, text: ms.Tensor, input_ids: Optional[ms.Tensor] = None):
        r"""Get_text_features

        Args:
            text (ms.Tensor): A text id tensor processed by tokenizer.
            input_ids (Optional[ms.Tensor]): Equal to "text",
                if "input_ids" is set, "text" is useless.

        Returns:
            Text feature.
        """
        if input_ids is not None:
            text = input_ids
        text_ = self.model.token_embedding(text)
        text_ = text_.astype(self.model.dtype)
        text_ = ops.Add()(text_, self.model.positional_embedding)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.model.transformer(text_)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.model.ln_final(text_)

        return text_


class FrozenOpenCLIPVisualEmbedder(nn.Cell):
    def __init__(
        self,
        arch="open_clip_vit_h_14",
        pretrained_ckpt_path="./vit-h-14-laion-2b/open_clip_vit_h_14.ckpt",
        freeze=True,
        layer="penultimate",
        resolution=224,
        use_fp16=False,
    ):
        super().__init__()
        self.dtype = ms.float16 if use_fp16 else ms.float32
        model = load_clip_model(arch, pretrained_ckpt_path, str(self.dtype).lower())
        del model.transformer

        self.model = model
        self.image_processor = CLIPImageProcessor(resolution)

        data_white = np.ones((resolution, resolution, 3)) * 255
        self.black_image = Image.fromarray(data_white.astype(np.uint8)).convert("RGB")

        self.layer = layer  # the layer does not apply to visual embedder
        if self.layer == "last":
            self.layer_index = 0
        elif self.layer == "penultimate":
            self.layer_index = 1
        else:
            raise ValueError(f"layer {layer} is not supported")

        self.freeze = freeze
        if self.freeze:
            self.model.set_train(False)
            for name, param in self.model.parameters_and_names():
                param.requires_grad = False

    def encode(self, image):
        if not isinstance(image, list):
            image_ = self.image_processor(image)
            image_features = self.construct(image_)
        else:
            image_ = [self.image_processor(img) for img in image]
            image_features = [self.construct(img) for img in image_]
        # the returned features are non-normalilzed

        # normalization
        # if not is_old_ms_version("2.0.0-alpha"):
        #     L2_norm_ops = partial(ops.norm, ord=2, dim=1, keepdim=True)
        # else:
        #     L2_norm_ops = partial(ops.norm, p=2, axis=1, keep_dims=True)

        # image_features = L2_norm_ops(image_features) if not isinstance(image_features, list) else [
        #     L2_norm_ops(img_feat) for img_feat in image_features]
        return image_features

    def construct(self, image):
        return self.model.get_image_features(image)
