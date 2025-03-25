__version__ = "4.46.3"
# Feature Extractor
from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .image_processing_base import ImageProcessingMixin
from .image_processing_utils import BaseImageProcessor
from .image_utils import ImageFeatureExtractionMixin
from .modeling_utils import MSPreTrainedModel
from .models.auto import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM
from .models.bert import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLayer,
    BertLMHeadModel,
    BertModel,
    BertPreTrainedModel,
)
from .models.bit import BitBackbone
from .models.blip_2 import (
    Blip2ForConditionalGeneration,
    Blip2Model,
    Blip2PreTrainedModel,
    Blip2QFormerModel,
    Blip2VisionModel,
)
from .models.clap import (
    ClapAudioModel,
    ClapAudioModelWithProjection,
    ClapFeatureExtractor,
    ClapModel,
    ClapPreTrainedModel,
    ClapTextModel,
    ClapTextModelWithProjection,
)
from .models.clip import (
    CLIP_PRETRAINED_MODEL_ARCHIVE_LIST,
    CLIPModel,
    CLIPPreTrainedModel,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPVisionModel,
    CLIPVisionModelWithProjection,
)
from .models.dpt import DPTForDepthEstimation
from .models.gemma import (
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
    GemmaModel,
    GemmaPreTrainedModel,
)
from .models.gemma2 import Gemma2Model, Gemma2PreTrainedModel
from .models.glm import (
    GlmForCausalLM,
    GlmForSequenceClassification,
    GlmForTokenClassification,
    GlmModel,
    GlmPreTrainedModel,
)
from .models.gpt2 import (
    GPT2DoubleHeadsModel,
    GPT2ForQuestionAnswering,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
    GPT2LMHeadModel,
    GPT2Model,
    GPT2PreTrainedModel,
)
from .models.llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel
from .models.llava import LlavaConfig, LlavaForConditionalGeneration
from .models.mt5 import (
    MT5_PRETRAINED_MODEL_ARCHIVE_LIST,
    MT5EncoderModel,
    MT5ForConditionalGeneration,
    MT5Model,
    MT5PreTrainedModel,
)
from .models.phi3 import (
    Phi3ForCausalLM,
    Phi3ForSequenceClassification,
    Phi3ForTokenClassification,
    Phi3Model,
    Phi3PreTrainedModel,
)
from .models.qwen2 import Qwen2ForCausalLM, Qwen2ForSequenceClassification, Qwen2ForTokenClassification, Qwen2Model
from .models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel
from .models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel, Qwen2VLPreTrainedModel
from .models.siglip import SiglipModel, SiglipPreTrainedModel, SiglipTextModel, SiglipVisionModel
from .models.speecht5 import (
    SpeechT5ForSpeechToSpeech,
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Model,
    SpeechT5PreTrainedModel,
)
from .models.t5 import (
    T5_PRETRAINED_MODEL_ARCHIVE_LIST,
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5Model,
    T5PreTrainedModel,
)
from .models.umt5 import (
    UMT5EncoderModel,
    UMT5ForQuestionAnswering,
    UMT5ForSequenceClassification,
    UMT5ForTokenClassification,
    UMT5Model,
    UMT5PreTrainedModel,
)
from .models.xlm_roberta import XLMRobertaModel, XLMRobertaPreTrainedModel
from .processing_utils import ProcessorMixin
