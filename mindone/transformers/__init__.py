__version__ = "4.46.3"
# Feature Extractor
from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .image_processing_base import ImageProcessingMixin
from .image_processing_utils import BaseImageProcessor
from .image_utils import ImageFeatureExtractionMixin
from .modeling_utils import MSPreTrainedModel
from .models.albert import (
    AlbertForMaskedLM,
    AlbertForMultipleChoice,
    AlbertForPreTraining,
    AlbertForQuestionAnswering,
    AlbertForSequenceClassification,
    AlbertForTokenClassification,
    AlbertModel,
    AlbertPreTrainedModel,
)
from .models.auto import AutoConfig, AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM
from .models.bart import (
    BartForCausalLM,
    BartForConditionalGeneration,
    BartForQuestionAnswering,
    BartForSequenceClassification,
    BartModel,
    BartPretrainedModel,
    BartPreTrainedModel,
    PretrainedBartModel,
)
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
from .models.big_bird import (
    BigBirdForCausalLM,
    BigBirdForMaskedLM,
    BigBirdForMultipleChoice,
    BigBirdForPreTraining,
    BigBirdForQuestionAnswering,
    BigBirdForSequenceClassification,
    BigBirdForTokenClassification,
    BigBirdLayer,
    BigBirdModel,
    BigBirdPreTrainedModel,
)
from .models.bigbird_pegasus import (
    BigBirdPegasusForCausalLM,
    BigBirdPegasusForConditionalGeneration,
    BigBirdPegasusForQuestionAnswering,
    BigBirdPegasusForSequenceClassification,
    BigBirdPegasusModel,
    BigBirdPegasusPreTrainedModel,
)
from .models.bit import BitBackbone
from .models.blip_2 import (
    Blip2ForConditionalGeneration,
    Blip2Model,
    Blip2PreTrainedModel,
    Blip2QFormerModel,
    Blip2VisionModel,
)
from .models.bloom import (
    BloomForCausalLM,
    BloomForQuestionAnswering,
    BloomForSequenceClassification,
    BloomForTokenClassification,
    BloomModel,
    BloomPreTrainedModel,
)
from .models.chameleon import (
    ChameleonForConditionalGeneration,
    ChameleonModel,
    ChameleonPreTrainedModel,
    ChameleonVQVAE,
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
from .models.deberta_v2 import (
    DebertaV2ForMaskedLM,
    DebertaV2ForMultipleChoice,
    DebertaV2ForQuestionAnswering,
    DebertaV2ForSequenceClassification,
    DebertaV2ForTokenClassification,
    DebertaV2Model,
    DebertaV2PreTrainedModel,
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
from .models.gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration, Gemma3PreTrainedModel, Gemma3TextModel
from .models.glm import (
    GlmForCausalLM,
    GlmForSequenceClassification,
    GlmForTokenClassification,
    GlmModel,
    GlmPreTrainedModel,
)
from .models.glpn import (
    GLPNFeatureExtractor,
    GLPNForDepthEstimation,
    GLPNImageProcessor,
    GLPNModel,
    GLPNPreTrainedModel,
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
from .models.hiera import (
    HieraBackbone,
    HieraForImageClassification,
    HieraForPreTraining,
    HieraModel,
    HieraPreTrainedModel,
)
from .models.ijepa import IJepaForImageClassification, IJepaModel, IJepaPreTrainedModel
from .models.imagegpt import (
    ImageGPTFeatureExtractor,
    ImageGPTForCausalImageModeling,
    ImageGPTForImageClassification,
    ImageGPTImageProcessor,
    ImageGPTModel,
    ImageGPTPreTrainedModel,
)
from .models.levit import (
    LevitFeatureExtractor,
    LevitForImageClassification,
    LevitForImageClassificationWithTeacher,
    LevitModel,
    LevitPreTrainedModel,
)
from .models.llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel, LlamaPreTrainedModel
from .models.llava import LlavaConfig, LlavaForConditionalGeneration
from .models.minicpm4 import MiniCPMForCausalLM, MiniCPMForSequenceClassification, MiniCPMModel
from .models.mixtral import (
    MixtralForCausalLM,
    MixtralForQuestionAnswering,
    MixtralForSequenceClassification,
    MixtralForTokenClassification,
    MixtralModel,
    MixtralPreTrainedModel,
)
from .models.mobilebert import (
    MobileBertForMaskedLM,
    MobileBertForMultipleChoice,
    MobileBertForNextSentencePrediction,
    MobileBertForPreTraining,
    MobileBertForQuestionAnswering,
    MobileBertForSequenceClassification,
    MobileBertForTokenClassification,
    MobileBertLayer,
    MobileBertModel,
    MobileBertPreTrainedModel,
)
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
from .models.qwen2_5_omni import (
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniPreTrainedModel,
    Qwen2_5OmniPreTrainedModelForConditionalGeneration,
    Qwen2_5OmniTalkerForConditionalGeneration,
    Qwen2_5OmniTalkerModel,
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniThinkerTextModel,
    Qwen2_5OmniToken2WavBigVGANModel,
    Qwen2_5OmniToken2WavDiTModel,
    Qwen2_5OmniToken2WavModel,
)
from .models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLModel, Qwen2_5_VLPreTrainedModel
from .models.qwen2_audio import Qwen2AudioEncoder, Qwen2AudioForConditionalGeneration, Qwen2AudioPreTrainedModel
from .models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel, Qwen2VLPreTrainedModel

# from .models.qwen3 import Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel
from .models.siglip import SiglipModel, SiglipPreTrainedModel, SiglipTextModel, SiglipVisionModel
from .models.speecht5 import (
    SpeechT5ForSpeechToSpeech,
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Model,
    SpeechT5PreTrainedModel,
)
from .models.switch_transformers import (
    SwitchTransformersEncoderModel,
    SwitchTransformersForConditionalGeneration,
    SwitchTransformersModel,
    SwitchTransformersPreTrainedModel,
    SwitchTransformersSparseMLP,
    SwitchTransformersTop1Router,
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
from .models.vits import VitsModel, VitsPreTrainedModel
from .models.wav2vec2 import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForAudioFrameClassification,
    Wav2Vec2ForCTC,
    Wav2Vec2ForMaskedLM,
    Wav2Vec2ForPreTraining,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2ForXVector,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
)
from .models.whisper import (
    WhisperForAudioClassification,
    WhisperForCausalLM,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperPreTrainedModel,
    WhisperProcessor,
)
from .models.xlm_roberta import XLMRobertaModel, XLMRobertaPreTrainedModel
from .pipelines import TextGenerationPipeline, pipeline
from .processing_utils import ProcessorMixin
