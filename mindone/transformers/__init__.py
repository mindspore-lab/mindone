# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/transformers
# with modifications to run transformers on mindspore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# When adding a new object to this init, remember to add it twice: once inside the `_import_structure` dictionary and
# once inside the `if TYPE_CHECKING` branch. The `TYPE_CHECKING` should have import statements as usual, but they are
# only there for type checking. The `_import_structure` is a dictionary submodule to list of object names, and is used
# to defer the actual importing for when the objects are requested. This way `import transformers` provides the names
# in the namespace without actually importing anything (and especially none of the backends).

__version__ = "4.50.0"
import transformers
from packaging import version

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
from .models.aria import (
    AriaForConditionalGeneration,
    AriaPreTrainedModel,
    AriaTextForCausalLM,
    AriaTextModel,
    AriaTextPreTrainedModel,
)
from .models.auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING,
    MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING,
    MODEL_FOR_AUDIO_XVECTOR_MAPPING,
    MODEL_FOR_BACKBONE_MAPPING,
    MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    MODEL_FOR_CTC_MAPPING,
    MODEL_FOR_DEPTH_ESTIMATION_MAPPING,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_IMAGE_MAPPING,
    MODEL_FOR_IMAGE_SEGMENTATION_MAPPING,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING,
    MODEL_FOR_IMAGE_TO_IMAGE_MAPPING,
    MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING,
    MODEL_FOR_KEYPOINT_DETECTION_MAPPING,
    MODEL_FOR_MASK_GENERATION_MAPPING,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING,
    MODEL_FOR_OBJECT_DETECTION_MAPPING,
    MODEL_FOR_PRETRAINING_MAPPING,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_RETRIEVAL_MAPPING,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING,
    MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_TEXT_ENCODING_MAPPING,
    MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING,
    MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING,
    MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING,
    MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING,
    MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING,
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING,
    MODEL_FOR_VISION_2_SEQ_MAPPING,
    MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING,
    MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING,
    MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING,
    MODEL_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoBackbone,
    AutoConfig,
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForDepthEstimation,
    AutoModelForDocumentQuestionAnswering,
    AutoModelForImageClassification,
    AutoModelForImageSegmentation,
    AutoModelForImageTextToText,
    AutoModelForImageToImage,
    AutoModelForInstanceSegmentation,
    AutoModelForKeypointDetection,
    AutoModelForMaskedImageModeling,
    AutoModelForMaskedLM,
    AutoModelForMaskGeneration,
    AutoModelForMultipleChoice,
    AutoModelForNextSentencePrediction,
    AutoModelForObjectDetection,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTableQuestionAnswering,
    AutoModelForTextEncoding,
    AutoModelForTextToSpectrogram,
    AutoModelForTextToWaveform,
    AutoModelForTokenClassification,
    AutoModelForUniversalSegmentation,
    AutoModelForVideoClassification,
    AutoModelForVision2Seq,
    AutoModelForVisualQuestionAnswering,
    AutoModelForZeroShotImageClassification,
    AutoModelForZeroShotObjectDetection,
    AutoModelWithLMHead,
    AutoProcessor,
)
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
from .models.blip import (
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
    BlipForQuestionAnswering,
    BlipImageProcessor,
    BlipImageProcessorFast,
    BlipModel,
    BlipPreTrainedModel,
    BlipProcessor,
    BlipTextModel,
    BlipVisionModel,
)
from .models.blip_2 import (
    Blip2ForConditionalGeneration,
    Blip2Model,
    Blip2PreTrainedModel,
    Blip2QFormerModel,
    Blip2VisionModel,
)
from .models.camembert import (
    CamembertForCausalLM,
    CamembertForMaskedLM,
    CamembertForMultipleChoice,
    CamembertForQuestionAnswering,
    CamembertForSequenceClassification,
    CamembertForTokenClassification,
    CamembertModel,
    CamembertPreTrainedModel,
)
from .models.canine import (
    CanineForMultipleChoice,
    CanineForQuestionAnswering,
    CanineForSequenceClassification,
    CanineForTokenClassification,
    CanineLayer,
    CanineModel,
    CaninePreTrainedModel,
)
from .models.chameleon import (
    ChameleonForConditionalGeneration,
    ChameleonImageProcessor,
    ChameleonModel,
    ChameleonPreTrainedModel,
    ChameleonProcessor,
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
from .models.cohere2 import Cohere2ForCausalLM, Cohere2Model, Cohere2PreTrainedModel
from .models.convbert import (
    ConvBertForMaskedLM,
    ConvBertForMultipleChoice,
    ConvBertForQuestionAnswering,
    ConvBertForSequenceClassification,
    ConvBertForTokenClassification,
    ConvBertLayer,
    ConvBertModel,
)
from .models.convnext import (
    ConvNextBackbone,
    ConvNextFeatureExtractor,
    ConvNextForImageClassification,
    ConvNextImageProcessor,
    ConvNextModel,
    ConvNextPreTrainedModel,
)
from .models.convnextv2 import (
    ConvNextV2Backbone,
    ConvNextV2ForImageClassification,
    ConvNextV2Model,
    ConvNextV2PreTrainedModel,
)
from .models.deberta import (
    DebertaForMaskedLM,
    DebertaForQuestionAnswering,
    DebertaForSequenceClassification,
    DebertaForTokenClassification,
    DebertaModel,
    DebertaPreTrainedModel,
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
from .models.depth_anything import DepthAnythingForDepthEstimation, DepthAnythingPreTrainedModel
from .models.dinov2 import Dinov2Backbone, Dinov2ForImageClassification, Dinov2Model, Dinov2PreTrainedModel
from .models.dpt import DPTForDepthEstimation, DPTImageProcessor, DPTModel, DPTPreTrainedModel
from .models.fuyu import FuyuForCausalLM, FuyuPreTrainedModel
from .models.gemma import (
    GemmaForCausalLM,
    GemmaForSequenceClassification,
    GemmaForTokenClassification,
    GemmaModel,
    GemmaPreTrainedModel,
)
from .models.gemma2 import (
    Gemma2ForCausalLM,
    Gemma2ForSequenceClassification,
    Gemma2ForTokenClassification,
    Gemma2Model,
    Gemma2PreTrainedModel,
)
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
from .models.gpt_bigcode import (
    GPTBigCodeForCausalLM,
    GPTBigCodeForSequenceClassification,
    GPTBigCodeForTokenClassification,
    GPTBigCodeModel,
    GPTBigCodePreTrainedModel,
)
from .models.gpt_neox import (
    GPTNeoXForCausalLM,
    GPTNeoXForQuestionAnswering,
    GPTNeoXForSequenceClassification,
    GPTNeoXLayer,
    GPTNeoXModel,
    GPTNeoXPreTrainedModel,
)
from .models.gpt_neox_japanese import (
    GPTNeoXJapaneseForCausalLM,
    GPTNeoXJapaneseLayer,
    GPTNeoXJapaneseModel,
    GPTNeoXJapanesePreTrainedModel,
)
from .models.gptj import (
    GPTJForCausalLM,
    GPTJForQuestionAnswering,
    GPTJForSequenceClassification,
    GPTJModel,
    GPTJPreTrainedModel,
)
from .models.granite import GraniteForCausalLM, GraniteModel, GranitePreTrainedModel
from .models.granitemoe import GraniteMoeForCausalLM, GraniteMoeModel, GraniteMoePreTrainedModel
from .models.granitemoeshared import GraniteMoeSharedForCausalLM, GraniteMoeSharedModel, GraniteMoeSharedPreTrainedModel
from .models.helium import (
    HeliumForCausalLM,
    HeliumForSequenceClassification,
    HeliumForTokenClassification,
    HeliumModel,
    HeliumPreTrainedModel,
)
from .models.hiera import (
    HieraBackbone,
    HieraForImageClassification,
    HieraForPreTraining,
    HieraModel,
    HieraPreTrainedModel,
)
from .models.hubert import HubertForCTC, HubertForSequenceClassification, HubertModel, HubertPreTrainedModel
from .models.idefics import (
    IdeficsForVisionText2Text,
    IdeficsImageProcessor,
    IdeficsModel,
    IdeficsPreTrainedModel,
    IdeficsProcessor,
)
from .models.idefics2 import Idefics2ForConditionalGeneration, Idefics2Model, Idefics2PreTrainedModel
from .models.idefics3 import (
    Idefics3ForConditionalGeneration,
    Idefics3Model,
    Idefics3PreTrainedModel,
    Idefics3VisionTransformer,
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
from .models.led import (
    LEDForConditionalGeneration,
    LEDForQuestionAnswering,
    LEDForSequenceClassification,
    LEDModel,
    LEDPreTrainedModel,
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
from .models.llava_next import (
    LlavaNextForConditionalGeneration,
    LlavaNextImageProcessor,
    LlavaNextPreTrainedModel,
    LlavaNextProcessor,
)
from .models.llava_next_video import (
    LlavaNextVideoForConditionalGeneration,
    LlavaNextVideoImageProcessor,
    LlavaNextVideoPreTrainedModel,
    LlavaNextVideoProcessor,
)
from .models.llava_onevision import (
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionImageProcessor,
    LlavaOnevisionPreTrainedModel,
    LlavaOnevisionProcessor,
    LlavaOnevisionVideoProcessor,
)
from .models.m2m_100 import M2M100ForConditionalGeneration, M2M100Model, M2M100PreTrainedModel
from .models.megatron_bert import (
    MegatronBertForCausalLM,
    MegatronBertForMaskedLM,
    MegatronBertForMultipleChoice,
    MegatronBertForNextSentencePrediction,
    MegatronBertForPreTraining,
    MegatronBertForQuestionAnswering,
    MegatronBertForSequenceClassification,
    MegatronBertForTokenClassification,
    MegatronBertModel,
    MegatronBertPreTrainedModel,
)
from .models.minicpm4 import MiniCPMForCausalLM, MiniCPMForSequenceClassification, MiniCPMModel
from .models.mistral import (
    MistralForCausalLM,
    MistralForQuestionAnswering,
    MistralForSequenceClassification,
    MistralForTokenClassification,
    MistralModel,
    MistralPreTrainedModel,
)
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
from .models.mpt import (
    MptForCausalLM,
    MptForQuestionAnswering,
    MptForSequenceClassification,
    MptForTokenClassification,
    MptModel,
    MptPreTrainedModel,
)
from .models.mt5 import (
    MT5_PRETRAINED_MODEL_ARCHIVE_LIST,
    MT5EncoderModel,
    MT5ForConditionalGeneration,
    MT5Model,
    MT5PreTrainedModel,
)
from .models.mvp import (
    MvpForCausalLM,
    MvpForConditionalGeneration,
    MvpForQuestionAnswering,
    MvpForSequenceClassification,
    MvpModel,
    MvpPreTrainedModel,
)
from .models.opt import (
    OPTForCausalLM,
    OPTForQuestionAnswering,
    OPTForSequenceClassification,
    OPTModel,
    OPTPreTrainedModel,
)
from .models.owlvit import (
    OwlViTForObjectDetection,
    OwlViTImageProcessor,
    OwlViTModel,
    OwlViTPreTrainedModel,
    OwlViTProcessor,
    OwlViTTextModel,
    OwlViTVisionModel,
)
from .models.paligemma import PaliGemmaForConditionalGeneration, PaliGemmaPreTrainedModel
from .models.persimmon import (
    PersimmonForCausalLM,
    PersimmonForSequenceClassification,
    PersimmonForTokenClassification,
    PersimmonModel,
    PersimmonPreTrainedModel,
)
from .models.phi import (
    PhiForCausalLM,
    PhiForSequenceClassification,
    PhiForTokenClassification,
    PhiModel,
    PhiPreTrainedModel,
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
from .models.recurrent_gemma import RecurrentGemmaForCausalLM, RecurrentGemmaModel, RecurrentGemmaPreTrainedModel
from .models.rembert import (
    RemBertForCausalLM,
    RemBertForMaskedLM,
    RemBertForMultipleChoice,
    RemBertForQuestionAnswering,
    RemBertForSequenceClassification,
    RemBertForTokenClassification,
    RemBertLayer,
    RemBertModel,
    RemBertPreTrainedModel,
)
from .models.roberta import (
    RobertaForCausalLM,
    RobertaForMaskedLM,
    RobertaForMultipleChoice,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
    RobertaPreTrainedModel,
)
from .models.segformer import (
    SegformerDecodeHead,
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    SegformerModel,
    SegformerPreTrainedModel,
)
from .models.siglip import (
    SiglipForImageClassification,
    SiglipImageProcessor,
    SiglipImageProcessorFast,
    SiglipModel,
    SiglipPreTrainedModel,
    SiglipProcessor,
    SiglipTextModel,
    SiglipVisionModel,
)
from .models.smolvlm import (
    SmolVLMForConditionalGeneration,
    SmolVLMModel,
    SmolVLMPreTrainedModel,
    SmolVLMVisionTransformer,
)
from .models.speecht5 import (
    SpeechT5ForSpeechToSpeech,
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Model,
    SpeechT5PreTrainedModel,
)
from .models.starcoder2 import (
    Starcoder2ForCausalLM,
    Starcoder2ForSequenceClassification,
    Starcoder2ForTokenClassification,
    Starcoder2Model,
    Starcoder2PreTrainedModel,
)
from .models.swin import (
    SwinBackbone,
    SwinForImageClassification,
    SwinForMaskedImageModeling,
    SwinModel,
    SwinPreTrainedModel,
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
from .models.vit import ViTForImageClassification, ViTForMaskedImageModeling, ViTModel, ViTPreTrainedModel
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
from .models.xlm_roberta_xl import (
    XLMRobertaXLForCausalLM,
    XLMRobertaXLForMaskedLM,
    XLMRobertaXLForMultipleChoice,
    XLMRobertaXLForQuestionAnswering,
    XLMRobertaXLForSequenceClassification,
    XLMRobertaXLForTokenClassification,
    XLMRobertaXLModel,
    XLMRobertaXLPreTrainedModel,
)
from .models.yolos import YolosForObjectDetection, YolosImageProcessor, YolosModel, YolosPreTrainedModel
from .pipelines import TextGenerationPipeline, pipeline
from .processing_utils import ProcessorMixin

if version.parse(transformers.__version__) >= version.parse("4.51.0"):
    from .models.qwen3 import Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel

if version.parse(transformers.__version__) >= version.parse("4.51.3"):
    from .models.glm4 import (
        Glm4ForCausalLM,
        Glm4ForSequenceClassification,
        Glm4ForTokenClassification,
        Glm4Model,
        Glm4PreTrainedModel,
    )

if version.parse(transformers.__version__) >= version.parse("4.53.0"):
    from .models.glm4v import (
        Glm4vForConditionalGeneration,
        Glm4vModel,
        Glm4vPreTrainedModel,
        Glm4vTextModel,
        Glm4vVisionModel,
    )
    from .models.minimax import (
        MiniMaxForCausalLM,
        MiniMaxForQuestionAnswering,
        MiniMaxForSequenceClassification,
        MiniMaxForTokenClassification,
        MiniMaxModel,
        MiniMaxPreTrainedModel,
    )
    from .models.vjepa2 import VJEPA2ForVideoClassification, VJEPA2Model, VJEPA2PreTrainedModel
