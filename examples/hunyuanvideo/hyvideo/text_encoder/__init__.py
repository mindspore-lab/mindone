from dataclasses import dataclass
from typing import Optional, Tuple

from hyvideo.constants import PRECISION_TO_TYPE, TEXT_ENCODER_PATH, TOKENIZER_PATH
from hyvideo.utils.helpers import set_model_param_dtype
from transformers import AutoTokenizer, CLIPTokenizer
from transformers.utils import ModelOutput

import mindspore as ms
from mindspore import Tensor, nn

from mindone.transformers import CLIPTextModel, LlamaModel
from mindone.transformers.models.llama.modeling_llama import ALL_LAYERNORM_LAYERS
from mindone.utils.amp import auto_mixed_precision


def use_default(value, default):
    return value if value is not None else default


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
):
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    if logger is not None:
        logger.info(f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}")
    if text_encoder_type == "clipL":
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = LlamaModel.from_pretrained(text_encoder_path, use_flash_attention_2=True)
        text_encoder.final_layer_norm = text_encoder.norm
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

    if text_encoder_precision is not None:
        # text_encoder = text_encoder.to_float(PRECISION_TO_TYPE[text_encoder_precision])
        # text encoder param: half
        dtype = PRECISION_TO_TYPE[text_encoder_precision]
        if dtype != ms.float32:
            set_model_param_dtype(text_encoder, dtype=dtype)

        amp_level = "O2"
        custom_fp32_cells = ALL_LAYERNORM_LAYERS
        text_encoder = auto_mixed_precision(
            text_encoder, amp_level=amp_level, dtype=dtype, custom_fp32_cells=custom_fp32_cells
        )
        logger.info(
            f"Set text encoder mixed precision to {amp_level} with dtype={dtype}, custom fp32_cells {custom_fp32_cells}"
        )

    text_encoder.set_train(False)

    return text_encoder, text_encoder_path


def load_tokenizer(tokenizer_type, tokenizer_path=None, padding_side="right", logger=None):
    if tokenizer_path is None:
        tokenizer_path = TOKENIZER_PATH[tokenizer_type]
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llm":
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding_side=padding_side)
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    return tokenizer, tokenizer_path


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`ms.Tensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(ms.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `ms.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: Tensor = None
    attention_mask: Optional[Tensor] = None
    hidden_states_list: Optional[Tuple[Tensor, ...]] = None
    text_outputs: Optional[list] = None


class TextEncoder(nn.Cell):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        input_max_length: Optional[int] = None,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        logger=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = tokenizer_type if tokenizer_type is not None else text_encoder_type
        self.tokenizer_path = tokenizer_path if tokenizer_path is not None else text_encoder_path
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert use_attention_mask is True, "Attention mask is True required when training videos."
        self.input_max_length = input_max_length if input_max_length is not None else max_length
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.logger = logger

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert (
                isinstance(self.prompt_template, dict) and "template" in self.prompt_template
            ), f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                    isinstance(self.prompt_template_video, dict) and "template" in self.prompt_template_video
                ), f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        # clip:  last_hidden_state [1 77 768], pooler_output[1 768], hidden_states None, attentions=None
        # llm: last_hidden_state, past_key_values, hidden_states, attentions,
        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
            # self.key_idx = 0
            raise ValueError("set key_idx for t5")
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
            self.key_idx = {"last_hidden_state": 0, "pooler_output": 1}[self.output_key]
        elif "llm" in text_encoder_type or "glm" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
            self.key_idx = {"last_hidden_state": 0, "hidden_states": 2}[self.output_key]
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
        )

        self.tokenizer, self.tokenizer_path = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )
        self.dtype = PRECISION_TO_TYPE[self.precision]

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    def text2tokens(self, text, data_type="image"):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        tokenize_input_type = "str"
        if self.use_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if isinstance(text, (list, tuple)):
                text = [self.apply_text_to_template(one_text, prompt_template) for one_text in text]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

        kwargs = dict(
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="np",
        )
        if tokenize_input_type == "str":
            return self.tokenizer(
                text,
                return_length=False,
                return_overflowing_tokens=False,
                return_attention_mask=True,
                **kwargs,
            )
        elif tokenize_input_type == "list":
            return self.tokenizer.apply_chat_template(
                text,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        hidden_state_skip_layer=None,
        return_texts=False,
        model_return_dict=True,
        data_type="image",
    ):
        """
        Args:
            batch_encoding (dict): Batch encoding from tokenizer.
            use_attention_mask (bool): Whether to use attention mask. If None, use self.use_attention_mask.
                Defaults to None.
            output_hidden_states (bool): Whether to output hidden states. If False, return the value of
                self.output_key. If True, return the entire output. If set self.hidden_state_skip_layer,
                output_hidden_states will be set True. Defaults to False.
            hidden_state_skip_layer (int): Number of hidden states to hidden_state_skip_layer. 0 means the last layer.
                If None, self.output_key will be used. Defaults to None.
            return_texts (bool): Whether to return the decoded texts. Defaults to False.
        """
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(hidden_state_skip_layer, self.hidden_state_skip_layer)
        attention_mask = Tensor(batch_encoding["attention_mask"]) if use_attention_mask else None
        outputs = self.model(
            input_ids=Tensor(batch_encoding["input_ids"]),
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states or hidden_state_skip_layer is not None,
            return_dict=model_return_dict,
        )

        # import pdb; pdb.set_trace()
        # clip:  last_hidden_state [1 77 768], pooler_output[1 768], hidden_states None, attentions=None
        # llm: last_hidden_state, past_key_values, hidden_states, attentions,
        if hidden_state_skip_layer is not None:
            if model_return_dict:
                last_hidden_state = outputs.hidden_states[-(hidden_state_skip_layer + 1)]
            else:
                last_hidden_state = outputs[2][-(hidden_state_skip_layer + 1)]
            # last_hidden_state = outputs[0][-(hidden_state_skip_layer + 1)]
            # Real last hidden state already has layer norm applied. So here we only apply it
            # for intermediate layers.
            if hidden_state_skip_layer > 0 and self.apply_final_norm:
                last_hidden_state = self.model.final_layer_norm(last_hidden_state)
        else:
            if model_return_dict:
                last_hidden_state = outputs[self.output_key]  # pooler_output for clip
                outputs_hidden_states = outputs.hidden_states
            else:
                last_hidden_state = outputs[self.key_idx]
                outputs_hidden_states = outputs[2] if len(outputs) >= 3 else None  # TODO: double-check if use t5

        # Remove hidden states of instruction tokens, only keep prompt tokens.
        if self.use_template:
            if data_type == "image":
                crop_start = self.prompt_template.get("crop_start", -1)
            elif data_type == "video":
                crop_start = self.prompt_template_video.get("crop_start", -1)
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if crop_start > 0:
                last_hidden_state = last_hidden_state[:, crop_start:]
                attention_mask = attention_mask[:, crop_start:] if use_attention_mask else None
        if output_hidden_states:
            return TextEncoderModelOutput(last_hidden_state, attention_mask, outputs_hidden_states)
        return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def construct(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
    ):
        batch_encoding = self.text2tokens(text, data_type)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
            data_type=data_type,
        )
