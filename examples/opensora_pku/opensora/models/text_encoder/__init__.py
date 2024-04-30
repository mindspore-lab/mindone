from opensora.utils.utils import get_precision

from mindspore import nn

from .clip import CLIPEmbedder
from .t5 import T5Embedder


class T5Wrapper(nn.Cell):
    def __init__(self, args):
        super(T5Wrapper, self).__init__()
        self.model_name = args.text_encoder_name
        self.model_max_length = args.token_max_length
        dtype = get_precision(args.precision)
        t5_model_kwargs = {
            "cache_dir": "./cache_dir",
            "low_cpu_mem_usage": True,
            "use_text_preprocessing": True,
            "model_max_length": self.model_max_length,
            "dtype": dtype,
        }
        self.text_enc = T5Embedder(self.model_name, **t5_model_kwargs)

    def construct(self, input_ids, attention_mask):
        text_encoder_embs = self.text_enc(input_ids=input_ids, attention_mask=attention_mask)
        return text_encoder_embs


class CLIPWrapper(nn.Cell):
    def __init__(self, args):
        super(CLIPWrapper, self).__init__()
        self.model_name = args.text_encoder_name
        self.model_max_length = args.token_max_length
        dtype = get_precision(args.precision)
        model_kwargs = {"cache_dir": "./cache_dir", "low_cpu_mem_usage": True, "dtype": dtype}
        self.text_enc = CLIPEmbedder(self.model_name, **model_kwargs)

    def construct(self, input_ids, attention_mask):
        text_encoder_embs, _ = self.text_enc.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        return text_encoder_embs


text_encoder = {"DeepFloyd/t5-v1_1-xxl": T5Wrapper, "openai/clip-vit-large-patch14": CLIPWrapper}


def get_text_enc(args):
    """deprecation"""
    text_enc = text_encoder.get(args.text_encoder_name, None)
    assert text_enc is not None
    return text_enc(args)
