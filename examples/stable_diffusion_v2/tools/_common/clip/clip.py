"""
CLIPModel
"""
import os
from functools import partial
from typing import Optional, Union

import numpy as np
from packaging import version

import mindspore as ms
import mindspore.ops as ops
from mindspore import Parameter, Tensor, nn
from mindspore.common.initializer import Normal, initializer
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from .clip_config import CLIPConfig
from .clip_modules import LayerNorm, Transformer, VisionTransformer


class CLIPModel(nn.Cell):
    r"""CLIPModel.
    The supported model name could be selected from CLIPModel.show_support_list().

    Args:
        config (CLIPConfig): The config of clip model, which could be obtained by CLIPConfig class.
    """

    def __init__(self, config: CLIPConfig):
        super().__init__()
        self.dtype = self.get_dtype(config.dtype)
        self.cross_entropy = nn.SoftmaxCrossEntropyWithLogits(reduction="mean", sparse=True)

        self.max_position_embeddings = config.text_config.max_position_embeddings
        self.visual = VisionTransformer(
            input_resolution=config.vision_config.image_size,
            patch_size=config.vision_config.patch_size,
            width=config.vision_config.hidden_size,
            layers=config.vision_config.num_hidden_layers,
            heads=config.vision_config.num_attention_heads,
            output_dim=config.projection_dim,
            dtype=self.dtype,
            hidden_act=config.vision_config.hidden_act,
        )
        self.visual = self.visual.to_float(self.dtype)

        self.transformer = Transformer(
            width=config.text_config.hidden_size,
            layers=config.text_config.num_hidden_layers,
            heads=config.text_config.num_attention_heads,
            dtype=self.dtype,
            hidden_act=config.text_config.hidden_act,
            attn_mask=self.build_attention_mask(),
        )
        self.transformer = self.transformer.to_float(self.dtype)

        self.token_embedding = nn.Embedding(
            config.text_config.vocab_size,
            config.text_config.hidden_size,
            embedding_table=Normal(mean=0.0, sigma=0.02),
            dtype=self.dtype,
        )
        self.positional_embedding = Parameter(
            initializer(
                Normal(mean=0.0, sigma=0.01),
                [config.text_config.max_position_embeddings, config.text_config.hidden_size],
                self.dtype,
            )
        )
        self.ln_final = LayerNorm([config.text_config.hidden_size])

        self.text_projection = Parameter(
            initializer(
                Normal(mean=0.0, sigma=config.text_config.hidden_size**-0.5),
                [config.text_config.hidden_size, config.projection_dim],
                self.dtype,
            )
        )
        self.logit_scale = Parameter(Tensor(np.log(1 / 0.07), self.dtype))
        self.exp = ops.Exp()

        self.load_checkpoint(config)

    def get_dtype(self, dtype: str):
        """Get_dtype"""
        if dtype == "float16":
            return ms.float16
        if dtype == "float32":
            return ms.float32
        raise TypeError("unsupported data type.")

    def construct(
        self,
        image: ms.Tensor,
        text: ms.Tensor,
        label: Optional[Union[ms.Tensor, np.ndarray]] = None,
        input_ids: Optional[ms.Tensor] = None,
        pixel_values: Optional[ms.Tensor] = None,
    ):
        r"""Construct

        Args:
            image (Tensor): A image tensor processed by image_processor.
            text (Tensor): A text id tensor processed by tokenizer.
            input_ids (Optional[ms.Tensor]): Equal to "text",
                if "input_ids" is set, "text" is useless.
            pixel_values (Optional[ms.Tensor]): Equal to "image",
                if "pixel_values" is set, "image" is useless.
            label (Optional[Union[ms.Tensor, np.ndarray]]): The classification label.

        Returns:
            if not self.trainining:
                if label is None:
                    logits_per_image: Similarity between image and text.
                    logits_per_text: Similarity between text and image.
                else:
                    logits_per_image: Similarity between image and text.
                    label: The classification label.
            else:
                loss: Constructive language image pretraining loss.
        """
        if pixel_values is not None:
            image = pixel_values

        if input_ids is not None:
            text = input_ids

        if len(text.shape) == 3:
            text = text[0].squeeze()

        image_features = self.get_image_features(image)
        text_features = self.get_text_features(text)
        if version.parse(ms.__version__) > version.parse("2.0.0-alpha"):
            L2_norm_ops = partial(ops.norm, ord=2, dim=1, keepdim=True)
        else:
            L2_norm_ops = partial(ops.norm, p=2, axis=1, keep_dims=True)

        image_features = image_features / L2_norm_ops(image_features)
        text_features = text_features / L2_norm_ops(text_features)
        logit_scale = self.exp(self.logit_scale)

        if label is None:
            logits_per_image = ops.matmul(logit_scale * image_features, text_features.T)
            logits_per_text = logits_per_image.T
            return logits_per_image, logits_per_text

        logits_per_image = ops.matmul(logit_scale * image_features, text_features.T)
        return logits_per_image, label

    def build_attention_mask(self):
        """Build_attention_mask"""
        mask = np.ones((self.max_position_embeddings, self.max_position_embeddings))
        mask = np.triu(mask * float("-inf"), k=1)
        return Tensor(mask, self.dtype)

    def get_image_features(self, image: ms.Tensor, pixel_values: Optional[ms.Tensor] = None):
        r"""Get_image_features

        Args:
            image (ms.Tensor): A image tensor processed by image_processor.
            pixel_values (Optional[ms.Tensor]): Equal to "image",
                if "pixel_values" is set, "image" is useless.

        Returns:
            Image feature.
        """
        if pixel_values is not None:
            image = pixel_values

        image = image.astype(self.dtype)
        return self.visual(image)

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
        text_ = self.token_embedding(text)
        text_ = text_.astype(self.dtype)
        text_ = ops.Add()(text_, self.positional_embedding)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.transformer(text_)
        text_ = text_.transpose(1, 0, 2)
        text_ = self.ln_final(text_)

        text_ = ops.matmul(text_[ms.numpy.arange(text_.shape[0]), text.argmax(-1)], self.text_projection)
        return text_

    def load_checkpoint(self, config):
        """
        Args:
            config (ModelConfig): a model config instance, which could have attribute
            "checkpoint_name_or_path (str)". set checkpoint_name_or_path to a supported
            model name or a path to checkpoint, to load model weights.
        """
        checkpoint_name_or_path = config.checkpoint_name_or_path
        if checkpoint_name_or_path:
            if not isinstance(checkpoint_name_or_path, str):
                raise TypeError(f"checkpoint_name_or_path should be a str," f" but got {type(checkpoint_name_or_path)}")

            if os.path.exists(checkpoint_name_or_path):
                param = load_checkpoint(checkpoint_name_or_path)
                ckpt_file = checkpoint_name_or_path

                try:
                    load_param_into_net(self, param)
                    print("weights in {} are loaded".format(ckpt_file))
                except RuntimeError:
                    print(
                        "the given config and weights in {} are"
                        " mismatched, and weights load failed".format(ckpt_file)
                    )
            else:
                checkpoint_name = checkpoint_name_or_path

                default_checkpoint_download_folder = os.path.join("download", "clip")

                if not os.path.exists(default_checkpoint_download_folder):
                    os.makedirs(default_checkpoint_download_folder, exist_ok=True)

                ckpt_file = os.path.join(default_checkpoint_download_folder, checkpoint_name + ".ckpt")

                print("start to read the ckpt file: {}".format(os.path.getsize(ckpt_file)))
                param = load_checkpoint(ckpt_file)
                try:
                    load_param_into_net(self, param)
                    print("weights in {} are loaded".format(ckpt_file))
                except RuntimeError:
                    print("the given config and weights in {} are" " mismatched, and weights load failed", ckpt_file)
        else:
            print(
                "model built, but weights is unloaded, since the config has no"
                " checkpoint_name_or_path attribute or"
                " checkpoint_name_or_path is None."
            )
