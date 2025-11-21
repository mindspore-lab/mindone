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

__version__ = "4.54.1"
import transformers
from packaging import version

from .cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    HybridCache,
    MambaCache,
    OffloadedStaticCache,
    SlidingWindowCache,
    StaticCache,
)
from .feature_extraction_sequence_utils import SequenceFeatureExtractor

# Feature Extractor
from .feature_extraction_utils import BatchFeature, FeatureExtractionMixin
from .image_processing_base import ImageProcessingMixin
from .image_processing_utils import BaseImageProcessor
from .image_processing_utils_fast import BaseImageProcessorFast
from .image_utils import ImageFeatureExtractionMixin
from .masking_utils import AttentionMaskInterface
from .modeling_utils import MSPreTrainedModel, PreTrainedModel
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
from .models.align import AlignModel, AlignPreTrainedModel, AlignTextModel, AlignVisionModel
from .models.altclip import AltCLIPModel, AltCLIPPreTrainedModel, AltCLIPTextModel, AltCLIPVisionModel
from .models.aria import (
    AriaForConditionalGeneration,
    AriaPreTrainedModel,
    AriaTextForCausalLM,
    AriaTextModel,
    AriaTextPreTrainedModel,
)
from .models.audio_spectrogram_transformer import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    ASTModel,
    ASTPreTrainedModel,
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
    AutoVideoProcessor,
)
from .models.aya_vision import AyaVisionForConditionalGeneration, AyaVisionPreTrainedModel
from .models.bamba import BambaForCausalLM, BambaModel, BambaPreTrainedModel
from .models.bark import BarkCausalModel, BarkCoarseModel, BarkFineModel, BarkModel, BarkSemanticModel
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
from .models.beit import (
    BeitBackbone,
    BeitForImageClassification,
    BeitForMaskedImageModeling,
    BeitForSemanticSegmentation,
    BeitModel,
    BeitPreTrainedModel,
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
from .models.bert_generation import BertGenerationDecoder, BertGenerationEncoder, BertGenerationPreTrainedModel
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
from .models.biogpt import (
    BioGptForCausalLM,
    BioGptForSequenceClassification,
    BioGptForTokenClassification,
    BioGptModel,
    BioGptPreTrainedModel,
)
from .models.bit import BitBackbone
from .models.blenderbot import (
    BlenderbotForCausalLM,
    BlenderbotForConditionalGeneration,
    BlenderbotModel,
    BlenderbotPreTrainedModel,
)
from .models.blenderbot_small import (
    BlenderbotSmallForCausalLM,
    BlenderbotSmallForConditionalGeneration,
    BlenderbotSmallModel,
    BlenderbotSmallPreTrainedModel,
)
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
from .models.bloom import (
    BloomForCausalLM,
    BloomForQuestionAnswering,
    BloomForSequenceClassification,
    BloomForTokenClassification,
    BloomModel,
    BloomPreTrainedModel,
)
from .models.bridgetower import (
    BridgeTowerForContrastiveLearning,
    BridgeTowerForImageAndTextRetrieval,
    BridgeTowerForMaskedLM,
    BridgeTowerModel,
    BridgeTowerPreTrainedModel,
)
from .models.bros import (
    BrosForTokenClassification,
    BrosModel,
    BrosPreTrainedModel,
    BrosSpadeEEForTokenClassification,
    BrosSpadeELForTokenClassification,
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
from .models.chinese_clip import (
    ChineseCLIPFeatureExtractor,
    ChineseCLIPImageProcessor,
    ChineseCLIPModel,
    ChineseCLIPPreTrainedModel,
    ChineseCLIPProcessor,
    ChineseCLIPTextModel,
    ChineseCLIPVisionModel,
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
from .models.clipseg import (
    CLIPSegForImageSegmentation,
    CLIPSegModel,
    CLIPSegPreTrainedModel,
    CLIPSegTextModel,
    CLIPSegVisionModel,
)
from .models.clvp import (
    ClvpDecoder,
    ClvpEncoder,
    ClvpForCausalLM,
    ClvpModel,
    ClvpModelForConditionalGeneration,
    ClvpPreTrainedModel,
)
from .models.codegen import CodeGenForCausalLM, CodeGenModel, CodeGenPreTrainedModel
from .models.cohere import CohereForCausalLM, CohereModel, CoherePreTrainedModel
from .models.cohere2 import Cohere2ForCausalLM, Cohere2Model, Cohere2PreTrainedModel
from .models.colpali import ColPaliForRetrieval, ColPaliPreTrainedModel, ColPaliProcessor
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
from .models.ctrl import CTRLForSequenceClassification, CTRLLMHeadModel, CTRLModel, CTRLPreTrainedModel
from .models.cvt import CvtForImageClassification, CvtModel, CvtPreTrainedModel
from .models.dac import DacModel, DacPreTrainedModel
from .models.data2vec import (
    Data2VecAudioForAudioFrameClassification,
    Data2VecAudioForCTC,
    Data2VecAudioForSequenceClassification,
    Data2VecAudioForXVector,
    Data2VecAudioModel,
    Data2VecAudioPreTrainedModel,
    Data2VecTextForCausalLM,
    Data2VecTextForMaskedLM,
    Data2VecTextForMultipleChoice,
    Data2VecTextForQuestionAnswering,
    Data2VecTextForSequenceClassification,
    Data2VecTextForTokenClassification,
    Data2VecTextModel,
    Data2VecTextPreTrainedModel,
    Data2VecVisionForImageClassification,
    Data2VecVisionForSemanticSegmentation,
    Data2VecVisionModel,
    Data2VecVisionPreTrainedModel,
)
from .models.dbrx import DbrxForCausalLM, DbrxModel, DbrxPreTrainedModel
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
from .models.deit import (
    DeiTForImageClassification,
    DeiTForImageClassificationWithTeacher,
    DeiTForMaskedImageModeling,
    DeiTModel,
    DeiTPreTrainedModel,
)
from .models.deprecated.xlm_prophetnet import (
    XLMProphetNetDecoder,
    XLMProphetNetEncoder,
    XLMProphetNetForCausalLM,
    XLMProphetNetForConditionalGeneration,
    XLMProphetNetModel,
    XLMProphetNetPreTrainedModel,
)
from .models.depth_anything import DepthAnythingForDepthEstimation, DepthAnythingPreTrainedModel
from .models.depth_pro import DepthProForDepthEstimation, DepthProImageProcessor, DepthProModel, DepthProPreTrainedModel
from .models.diffllama import (
    DiffLlamaForCausalLM,
    DiffLlamaForQuestionAnswering,
    DiffLlamaForSequenceClassification,
    DiffLlamaForTokenClassification,
    DiffLlamaModel,
    DiffLlamaPreTrainedModel,
)
from .models.dinov2 import Dinov2Backbone, Dinov2ForImageClassification, Dinov2Model, Dinov2PreTrainedModel
from .models.dinov2_with_registers import (
    Dinov2WithRegistersBackbone,
    Dinov2WithRegistersForImageClassification,
    Dinov2WithRegistersModel,
    Dinov2WithRegistersPreTrainedModel,
)
from .models.dinov3_vit import DINOv3ViTModel, DINOv3ViTPreTrainedModel, DINOv3ViTImageProcessorFast
from .models.distilbert import (
    DistilBertForMaskedLM,
    DistilBertForMultipleChoice,
    DistilBertForQuestionAnswering,
    DistilBertForSequenceClassification,
    DistilBertForTokenClassification,
    DistilBertModel,
    DistilBertPreTrainedModel,
)
from .models.dpr import (
    DPRContextEncoder,
    DPRPretrainedContextEncoder,
    DPRPreTrainedModel,
    DPRPretrainedQuestionEncoder,
    DPRPretrainedReader,
    DPRQuestionEncoder,
    DPRReader,
)
from .models.dpt import DPTForDepthEstimation, DPTImageProcessor, DPTModel, DPTPreTrainedModel
from .models.efficientnet import (
    EfficientNetForImageClassification,
    EfficientNetImageProcessor,
    EfficientNetModel,
    EfficientNetPreTrainedModel,
)
from .models.electra import (
    ElectraForCausalLM,
    ElectraForMaskedLM,
    ElectraForMultipleChoice,
    ElectraForPreTraining,
    ElectraForQuestionAnswering,
    ElectraForSequenceClassification,
    ElectraForTokenClassification,
    ElectraModel,
    ElectraPreTrainedModel,
)
from .models.emu3 import Emu3ForCausalLM, Emu3ForConditionalGeneration, Emu3PreTrainedModel, Emu3TextModel, Emu3VQVAE
from .models.encodec import EncodecModel, EncodecPreTrainedModel
from .models.encoder_decoder import EncoderDecoderModel
from .models.ernie import (
    ErnieForCausalLM,
    ErnieForMaskedLM,
    ErnieForMultipleChoice,
    ErnieForNextSentencePrediction,
    ErnieForPreTraining,
    ErnieForQuestionAnswering,
    ErnieForSequenceClassification,
    ErnieForTokenClassification,
    ErnieModel,
    ErniePreTrainedModel,
)
from .models.esm import (
    EsmForMaskedLM,
    EsmForSequenceClassification,
    EsmForTokenClassification,
    EsmModel,
    EsmPreTrainedModel,
)
from .models.falcon import (
    FalconForCausalLM,
    FalconForQuestionAnswering,
    FalconForSequenceClassification,
    FalconForTokenClassification,
    FalconModel,
    FalconPreTrainedModel,
)
from .models.falcon_mamba import FalconMambaForCausalLM, FalconMambaModel, FalconMambaPreTrainedModel
from .models.fastspeech2_conformer import (
    FastSpeech2ConformerHifiGan,
    FastSpeech2ConformerModel,
    FastSpeech2ConformerPreTrainedModel,
    FastSpeech2ConformerWithHifiGan,
)
from .models.flaubert import (
    FlaubertForMultipleChoice,
    FlaubertForQuestionAnswering,
    FlaubertForQuestionAnsweringSimple,
    FlaubertForSequenceClassification,
    FlaubertForTokenClassification,
    FlaubertModel,
    FlaubertPreTrainedModel,
    FlaubertWithLMHeadModel,
)
from .models.flava import (
    FlavaFeatureExtractor,
    FlavaForPreTraining,
    FlavaImageCodebook,
    FlavaImageModel,
    FlavaImageProcessor,
    FlavaModel,
    FlavaMultimodalModel,
    FlavaPreTrainedModel,
    FlavaProcessor,
    FlavaTextModel,
)
from .models.fnet import (
    FNetForMaskedLM,
    FNetForMultipleChoice,
    FNetForNextSentencePrediction,
    FNetForPreTraining,
    FNetForQuestionAnswering,
    FNetForSequenceClassification,
    FNetForTokenClassification,
    FNetLayer,
    FNetModel,
    FNetPreTrainedModel,
)
from .models.focalnet import (
    FocalNetBackbone,
    FocalNetForImageClassification,
    FocalNetForMaskedImageModeling,
    FocalNetModel,
    FocalNetPreTrainedModel,
)
from .models.fsmt import FSMTForConditionalGeneration, FSMTModel, PretrainedFSMTModel
from .models.funnel import (
    FunnelBaseModel,
    FunnelForMaskedLM,
    FunnelForMultipleChoice,
    FunnelForPreTraining,
    FunnelForQuestionAnswering,
    FunnelForSequenceClassification,
    FunnelForTokenClassification,
    FunnelModel,
    FunnelPreTrainedModel,
)
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
from .models.git import GitForCausalLM, GitModel, GitPreTrainedModel, GitVisionModel
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
from .models.got_ocr2 import GotOcr2ForConditionalGeneration, GotOcr2PreTrainedModel
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
from .models.gpt_neo import (
    GPTNeoForCausalLM,
    GPTNeoForQuestionAnswering,
    GPTNeoForSequenceClassification,
    GPTNeoForTokenClassification,
    GPTNeoModel,
    GPTNeoPreTrainedModel,
)
from .models.gpt_neox import (
    GPTNeoXForCausalLM,
    GPTNeoXForQuestionAnswering,
    GPTNeoXForSequenceClassification,
    GPTNeoXForTokenClassification,
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
from .models.grounding_dino import (
    GroundingDinoForObjectDetection,
    GroundingDinoImageProcessor,
    GroundingDinoModel,
    GroundingDinoPreTrainedModel,
    GroundingDinoProcessor,
)
from .models.groupvit import GroupViTModel, GroupViTPreTrainedModel, GroupViTTextModel, GroupViTVisionModel
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
from .models.ibert import (
    IBertForMaskedLM,
    IBertForMultipleChoice,
    IBertForQuestionAnswering,
    IBertForSequenceClassification,
    IBertForTokenClassification,
    IBertModel,
    IBertPreTrainedModel,
)
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
from .models.instructblip import (
    InstructBlipForConditionalGeneration,
    InstructBlipPreTrainedModel,
    InstructBlipProcessor,
    InstructBlipQFormerModel,
    InstructBlipVisionModel,
)
from .models.instructblipvideo import (
    InstructBlipVideoForConditionalGeneration,
    InstructBlipVideoPreTrainedModel,
    InstructBlipVideoQFormerModel,
    InstructBlipVideoVisionModel,
)
from .models.jamba import JambaForCausalLM, JambaForSequenceClassification, JambaModel, JambaPreTrainedModel
from .models.jetmoe import (
    JetMoeConfig,
    JetMoeForCausalLM,
    JetMoeForSequenceClassification,
    JetMoeModel,
    JetMoePreTrainedModel,
)
from .models.kosmos2 import Kosmos2ForConditionalGeneration, Kosmos2Model, Kosmos2PreTrainedModel
from .models.layoutlm import (
    LayoutLMForMaskedLM,
    LayoutLMForQuestionAnswering,
    LayoutLMForSequenceClassification,
    LayoutLMForTokenClassification,
    LayoutLMModel,
    LayoutLMPreTrainedModel,
)
from .models.layoutlmv3 import (
    LayoutLMv3ForQuestionAnswering,
    LayoutLMv3ForSequenceClassification,
    LayoutLMv3ForTokenClassification,
    LayoutLMv3ImageProcessor,
    LayoutLMv3Model,
    LayoutLMv3PreTrainedModel,
    LayoutLMv3Processor,
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
from .models.lilt import (
    LiltForQuestionAnswering,
    LiltForSequenceClassification,
    LiltForTokenClassification,
    LiltModel,
    LiltPreTrainedModel,
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
from .models.longformer import (
    LongformerForMaskedLM,
    LongformerForMultipleChoice,
    LongformerForQuestionAnswering,
    LongformerForSequenceClassification,
    LongformerForTokenClassification,
    LongformerModel,
    LongformerPreTrainedModel,
)
from .models.longt5 import LongT5EncoderModel, LongT5ForConditionalGeneration, LongT5Model, LongT5PreTrainedModel
from .models.luke import (
    LukeForEntityClassification,
    LukeForEntityPairClassification,
    LukeForEntitySpanClassification,
    LukeForMaskedLM,
    LukeForMultipleChoice,
    LukeForQuestionAnswering,
    LukeForSequenceClassification,
    LukeForTokenClassification,
    LukeModel,
    LukePreTrainedModel,
)
from .models.m2m_100 import M2M100ForConditionalGeneration, M2M100Model, M2M100PreTrainedModel
from .models.mamba import MambaForCausalLM, MambaModel, MambaPreTrainedModel
from .models.mamba2 import Mamba2ForCausalLM, Mamba2Model, Mamba2PreTrainedModel
from .models.marian import MarianForCausalLM, MarianModel, MarianMTModel, MarianPreTrainedModel
from .models.markuplm import (
    MarkupLMForQuestionAnswering,
    MarkupLMForSequenceClassification,
    MarkupLMForTokenClassification,
    MarkupLMModel,
    MarkupLMPreTrainedModel,
)
from .models.mask2former import Mask2FormerForUniversalSegmentation, Mask2FormerModel, Mask2FormerPreTrainedModel
from .models.maskformer import (
    MaskFormerForInstanceSegmentation,
    MaskFormerImageProcessor,
    MaskFormerModel,
    MaskFormerPreTrainedModel,
    MaskFormerSwinBackbone,
)
from .models.mbart import (
    MBartForCausalLM,
    MBartForConditionalGeneration,
    MBartForQuestionAnswering,
    MBartForSequenceClassification,
    MBartModel,
    MBartPreTrainedModel,
)
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
from .models.mgp_str import MgpstrForSceneTextRecognition, MgpstrModel, MgpstrPreTrainedModel, MgpstrProcessor
from .models.mimi import MimiModel, MimiPreTrainedModel
from .models.minicpm4 import MiniCPMForCausalLM, MiniCPMForSequenceClassification, MiniCPMModel
from .models.mistral import (
    MistralForCausalLM,
    MistralForQuestionAnswering,
    MistralForSequenceClassification,
    MistralForTokenClassification,
    MistralModel,
    MistralPreTrainedModel,
)
from .models.mistral3 import Mistral3ForConditionalGeneration, Mistral3PreTrainedModel
from .models.mixtral import (
    MixtralForCausalLM,
    MixtralForQuestionAnswering,
    MixtralForSequenceClassification,
    MixtralForTokenClassification,
    MixtralModel,
    MixtralPreTrainedModel,
)
from .models.mllama import (
    MllamaForCausalLM,
    MllamaForConditionalGeneration,
    MllamaPreTrainedModel,
    MllamaTextModel,
    MllamaVisionModel,
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
from .models.mobilenet_v1 import (
    MobileNetV1ForImageClassification,
    MobileNetV1ImageProcessor,
    MobileNetV1ImageProcessorFast,
    MobileNetV1Model,
    MobileNetV1PreTrainedModel,
)
from .models.mobilenet_v2 import (
    MobileNetV2ForImageClassification,
    MobileNetV2ForSemanticSegmentation,
    MobileNetV2ImageProcessor,
    MobileNetV2ImageProcessorFast,
    MobileNetV2Model,
    MobileNetV2PreTrainedModel,
)
from .models.mobilevit import (
    MobileViTForImageClassification,
    MobileViTForSemanticSegmentation,
    MobileViTModel,
    MobileViTPreTrainedModel,
)
from .models.mobilevitv2 import (
    MobileViTV2ForImageClassification,
    MobileViTV2ForSemanticSegmentation,
    MobileViTV2Model,
    MobileViTV2PreTrainedModel,
)
from .models.modernbert import (
    ModernBertForMaskedLM,
    ModernBertForSequenceClassification,
    ModernBertForTokenClassification,
    ModernBertModel,
    ModernBertPreTrainedModel,
)
from .models.moonshine import MoonshineForConditionalGeneration, MoonshineModel, MoonshinePreTrainedModel
from .models.moshi import MoshiForCausalLM, MoshiForConditionalGeneration, MoshiModel, MoshiPreTrainedModel
from .models.mpnet import (
    MPNetForMaskedLM,
    MPNetForMultipleChoice,
    MPNetForQuestionAnswering,
    MPNetForSequenceClassification,
    MPNetForTokenClassification,
    MPNetLayer,
    MPNetModel,
    MPNetPreTrainedModel,
)
from .models.mpt import (
    MptForCausalLM,
    MptForQuestionAnswering,
    MptForSequenceClassification,
    MptForTokenClassification,
    MptModel,
    MptPreTrainedModel,
)
from .models.mra import (
    MraForMaskedLM,
    MraForMultipleChoice,
    MraForQuestionAnswering,
    MraForSequenceClassification,
    MraForTokenClassification,
    MraLayer,
    MraModel,
    MraPreTrainedModel,
)
from .models.mt5 import (
    MT5_PRETRAINED_MODEL_ARCHIVE_LIST,
    MT5EncoderModel,
    MT5ForConditionalGeneration,
    MT5Model,
    MT5PreTrainedModel,
)
from .models.musicgen import (
    MusicgenForCausalLM,
    MusicgenForConditionalGeneration,
    MusicgenModel,
    MusicgenPreTrainedModel,
)
from .models.musicgen_melody import (
    MusicgenMelodyForCausalLM,
    MusicgenMelodyForConditionalGeneration,
    MusicgenMelodyModel,
    MusicgenMelodyPreTrainedModel,
)
from .models.mvp import (
    MvpForCausalLM,
    MvpForConditionalGeneration,
    MvpForQuestionAnswering,
    MvpForSequenceClassification,
    MvpModel,
    MvpPreTrainedModel,
)
from .models.nemotron import (
    NemotronForCausalLM,
    NemotronForQuestionAnswering,
    NemotronForSequenceClassification,
    NemotronForTokenClassification,
    NemotronModel,
    NemotronPreTrainedModel,
)
from .models.nllb_moe import (
    NllbMoeForConditionalGeneration,
    NllbMoeModel,
    NllbMoePreTrainedModel,
    NllbMoeSparseMLP,
    NllbMoeTop2Router,
)
from .models.nystromformer import (
    NystromformerEncoder,
    NystromformerForMaskedLM,
    NystromformerForMultipleChoice,
    NystromformerForQuestionAnswering,
    NystromformerForSequenceClassification,
    NystromformerForTokenClassification,
    NystromformerModel,
    NystromformerPreTrainedModel,
)
from .models.olmo import OlmoForCausalLM, OlmoModel, OlmoPreTrainedModel
from .models.olmo2 import Olmo2ForCausalLM, Olmo2Model, Olmo2PreTrainedModel
from .models.olmoe import OlmoeForCausalLM, OlmoeModel, OlmoePreTrainedModel
from .models.oneformer import (
    OneFormerForUniversalSegmentation,
    OneFormerImageProcessor,
    OneFormerModel,
    OneFormerPreTrainedModel,
    OneFormerProcessor,
)
from .models.opt import (
    OPTForCausalLM,
    OPTForQuestionAnswering,
    OPTForSequenceClassification,
    OPTModel,
    OPTPreTrainedModel,
)
from .models.owlv2 import (
    Owlv2ForObjectDetection,
    Owlv2ImageProcessor,
    Owlv2Model,
    Owlv2PreTrainedModel,
    Owlv2Processor,
    Owlv2TextModel,
    Owlv2VisionModel,
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
from .models.pegasus import PegasusForCausalLM, PegasusForConditionalGeneration, PegasusModel, PegasusPreTrainedModel
from .models.pegasus_x import PegasusXForConditionalGeneration, PegasusXModel, PegasusXPreTrainedModel
from .models.perceiver import (
    PerceiverForImageClassificationConvProcessing,
    PerceiverForImageClassificationFourier,
    PerceiverForImageClassificationLearned,
    PerceiverForMaskedLM,
    PerceiverForMultimodalAutoencoding,
    PerceiverForOpticalFlow,
    PerceiverForSequenceClassification,
    PerceiverModel,
    PerceiverPreTrainedModel,
)
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
from .models.phimoe import PhimoeForCausalLM, PhimoeForSequenceClassification, PhimoeModel, PhimoePreTrainedModel
from .models.pix2struct import (
    Pix2StructForConditionalGeneration,
    Pix2StructPreTrainedModel,
    Pix2StructTextModel,
    Pix2StructVisionModel,
)
from .models.pixtral import PixtralPreTrainedModel, PixtralVisionModel
from .models.plbart import (
    PLBartForCausalLM,
    PLBartForConditionalGeneration,
    PLBartForSequenceClassification,
    PLBartModel,
    PLBartPreTrainedModel,
)
from .models.poolformer import PoolFormerForImageClassification, PoolFormerModel, PoolFormerPreTrainedModel
from .models.pop2piano import Pop2PianoForConditionalGeneration, Pop2PianoPreTrainedModel
from .models.prompt_depth_anything import PromptDepthAnythingForDepthEstimation, PromptDepthAnythingPreTrainedModel
from .models.prophetnet import (
    ProphetNetDecoder,
    ProphetNetEncoder,
    ProphetNetForCausalLM,
    ProphetNetForConditionalGeneration,
    ProphetNetModel,
    ProphetNetPreTrainedModel,
)
from .models.pvt import PvtForImageClassification, PvtModel, PvtPreTrainedModel
from .models.pvt_v2 import PvtV2Backbone, PvtV2ForImageClassification, PvtV2Model, PvtV2PreTrainedModel
from .models.qwen2 import Qwen2ForCausalLM, Qwen2ForSequenceClassification, Qwen2ForTokenClassification, Qwen2Model
from .models.qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLPreTrainedModel,
    Qwen2_5_VLProcessor,
)
from .models.qwen2_audio import Qwen2AudioEncoder, Qwen2AudioForConditionalGeneration, Qwen2AudioPreTrainedModel
from .models.qwen2_moe import (
    Qwen2MoeForCausalLM,
    Qwen2MoeForQuestionAnswering,
    Qwen2MoeForSequenceClassification,
    Qwen2MoeForTokenClassification,
    Qwen2MoeModel,
    Qwen2MoePreTrainedModel,
)
from .models.qwen2_vl import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLImageProcessor,
    Qwen2VLImageProcessorFast,
    Qwen2VLModel,
    Qwen2VLPreTrainedModel,
)
from .models.rag import RagModel, RagPreTrainedModel, RagSequenceForGeneration, RagTokenForGeneration
from .models.recurrent_gemma import RecurrentGemmaForCausalLM, RecurrentGemmaModel, RecurrentGemmaPreTrainedModel
from .models.reformer import (
    ReformerAttention,
    ReformerForMaskedLM,
    ReformerForQuestionAnswering,
    ReformerForSequenceClassification,
    ReformerLayer,
    ReformerModel,
    ReformerModelWithLMHead,
    ReformerPreTrainedModel,
)
from .models.regnet import RegNetForImageClassification, RegNetModel, RegNetPreTrainedModel
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
from .models.resnet import ResNetBackbone, ResNetForImageClassification, ResNetModel, ResNetPreTrainedModel
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
from .models.roberta_prelayernorm import (
    RobertaPreLayerNormForCausalLM,
    RobertaPreLayerNormForMaskedLM,
    RobertaPreLayerNormForMultipleChoice,
    RobertaPreLayerNormForQuestionAnswering,
    RobertaPreLayerNormForSequenceClassification,
    RobertaPreLayerNormForTokenClassification,
    RobertaPreLayerNormModel,
    RobertaPreLayerNormPreTrainedModel,
)
from .models.roc_bert import (
    RoCBertForCausalLM,
    RoCBertForMaskedLM,
    RoCBertForMultipleChoice,
    RoCBertForPreTraining,
    RoCBertForQuestionAnswering,
    RoCBertForSequenceClassification,
    RoCBertForTokenClassification,
    RoCBertModel,
    RoCBertPreTrainedModel,
)
from .models.roformer import (
    RoFormerForCausalLM,
    RoFormerForMaskedLM,
    RoFormerForMultipleChoice,
    RoFormerForQuestionAnswering,
    RoFormerForSequenceClassification,
    RoFormerForTokenClassification,
    RoFormerLayer,
    RoFormerModel,
    RoFormerPreTrainedModel,
)
from .models.rt_detr import RTDetrForObjectDetection, RTDetrModel, RTDetrPreTrainedModel
from .models.rt_detr_v2 import RTDetrV2ForObjectDetection, RTDetrV2Model, RTDetrV2PreTrainedModel
from .models.rwkv import RwkvForCausalLM, RwkvModel, RwkvPreTrainedModel
from .models.sam import SamImageProcessor, SamModel, SamPreTrainedModel, SamProcessor
from .models.seamless_m4t import (
    SeamlessM4TConfig,
    SeamlessM4TFeatureExtractor,
    SeamlessM4TForSpeechToSpeech,
    SeamlessM4TForSpeechToText,
    SeamlessM4TForTextToSpeech,
    SeamlessM4TForTextToText,
    SeamlessM4TModel,
    SeamlessM4TProcessor,
)
from .models.seamless_m4t_v2 import (
    SeamlessM4Tv2ForSpeechToSpeech,
    SeamlessM4Tv2ForSpeechToText,
    SeamlessM4Tv2ForTextToSpeech,
    SeamlessM4Tv2ForTextToText,
    SeamlessM4Tv2Model,
    SeamlessM4Tv2PreTrainedModel,
)
from .models.segformer import (
    SegformerDecodeHead,
    SegformerForImageClassification,
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    SegformerModel,
    SegformerPreTrainedModel,
)
from .models.seggpt import SegGptForImageSegmentation, SegGptModel, SegGptPreTrainedModel
from .models.sew import SEWForCTC, SEWForSequenceClassification, SEWModel, SEWPreTrainedModel
from .models.sew_d import SEWDForCTC, SEWDForSequenceClassification, SEWDModel, SEWDPreTrainedModel
from .models.shieldgemma2 import ShieldGemma2ForImageClassification
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
from .models.siglip2 import (
    Siglip2ForImageClassification,
    Siglip2Model,
    Siglip2PreTrainedModel,
    Siglip2TextModel,
    Siglip2VisionModel,
)
from .models.smolvlm import (
    SmolVLMForConditionalGeneration,
    SmolVLMModel,
    SmolVLMPreTrainedModel,
    SmolVLMVisionTransformer,
)
from .models.speech_encoder_decoder import SpeechEncoderDecoderModel
from .models.speech_to_text import Speech2TextForConditionalGeneration, Speech2TextModel, Speech2TextPreTrainedModel
from .models.speecht5 import (
    SpeechT5ForSpeechToSpeech,
    SpeechT5ForSpeechToText,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    SpeechT5Model,
    SpeechT5PreTrainedModel,
)
from .models.splinter import (
    SplinterForPreTraining,
    SplinterForQuestionAnswering,
    SplinterLayer,
    SplinterModel,
    SplinterPreTrainedModel,
)
from .models.squeezebert import (
    SqueezeBertForMaskedLM,
    SqueezeBertForMultipleChoice,
    SqueezeBertForQuestionAnswering,
    SqueezeBertForSequenceClassification,
    SqueezeBertForTokenClassification,
    SqueezeBertModel,
    SqueezeBertPreTrainedModel,
)
from .models.stablelm import (
    StableLmForCausalLM,
    StableLmForSequenceClassification,
    StableLmForTokenClassification,
    StableLmModel,
    StableLmPreTrainedModel,
)
from .models.starcoder2 import (
    Starcoder2ForCausalLM,
    Starcoder2ForSequenceClassification,
    Starcoder2ForTokenClassification,
    Starcoder2Model,
    Starcoder2PreTrainedModel,
)
from .models.swiftformer import SwiftFormerForImageClassification, SwiftFormerModel, SwiftFormerPreTrainedModel
from .models.swin import (
    SwinBackbone,
    SwinForImageClassification,
    SwinForMaskedImageModeling,
    SwinModel,
    SwinPreTrainedModel,
)
from .models.swin2sr import Swin2SRForImageSuperResolution, Swin2SRModel, Swin2SRPreTrainedModel
from .models.swinv2 import (
    Swinv2Backbone,
    Swinv2ForImageClassification,
    Swinv2ForMaskedImageModeling,
    Swinv2Model,
    Swinv2PreTrainedModel,
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
from .models.table_transformer import (
    TableTransformerForObjectDetection,
    TableTransformerModel,
    TableTransformerPreTrainedModel,
)
from .models.tapas import (
    TapasForMaskedLM,
    TapasForQuestionAnswering,
    TapasForSequenceClassification,
    TapasModel,
    TapasPreTrainedModel,
)
from .models.textnet import TextNetBackbone, TextNetForImageClassification, TextNetModel, TextNetPreTrainedModel
from .models.timesformer import TimesformerForVideoClassification, TimesformerModel, TimesformerPreTrainedModel
from .models.trocr import TrOCRForCausalLM, TrOCRPreTrainedModel
from .models.tvp import TvpForVideoGrounding, TvpModel, TvpPreTrainedModel
from .models.udop import UdopEncoderModel, UdopForConditionalGeneration, UdopModel, UdopPreTrainedModel
from .models.umt5 import (
    UMT5EncoderModel,
    UMT5ForQuestionAnswering,
    UMT5ForSequenceClassification,
    UMT5ForTokenClassification,
    UMT5Model,
    UMT5PreTrainedModel,
)
from .models.unispeech import (
    UniSpeechForCTC,
    UniSpeechForPreTraining,
    UniSpeechForSequenceClassification,
    UniSpeechModel,
    UniSpeechPreTrainedModel,
)
from .models.unispeech_sat import (
    UniSpeechSatForAudioFrameClassification,
    UniSpeechSatForCTC,
    UniSpeechSatForPreTraining,
    UniSpeechSatForSequenceClassification,
    UniSpeechSatForXVector,
    UniSpeechSatModel,
    UniSpeechSatPreTrainedModel,
)
from .models.univnet import UnivNetModel
from .models.upernet import UperNetForSemanticSegmentation, UperNetPreTrainedModel
from .models.video_llava import VideoLlavaForConditionalGeneration, VideoLlavaPreTrainedModel
from .models.videomae import (
    VideoMAEForPreTraining,
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    VideoMAEModel,
    VideoMAEPreTrainedModel,
)
from .models.vilt import (
    ViltForImageAndTextRetrieval,
    ViltForImagesAndTextClassification,
    ViltForMaskedLM,
    ViltForQuestionAnswering,
    ViltForTokenClassification,
    ViltModel,
    ViltPreTrainedModel,
)
from .models.vipllava import VipLlavaForConditionalGeneration, VipLlavaPreTrainedModel
from .models.vision_encoder_decoder import VisionEncoderDecoderModel
from .models.vision_text_dual_encoder import VisionTextDualEncoderModel
from .models.visual_bert import (
    VisualBertForMultipleChoice,
    VisualBertForPreTraining,
    VisualBertForQuestionAnswering,
    VisualBertForRegionToPhraseAlignment,
    VisualBertForVisualReasoning,
    VisualBertModel,
    VisualBertPreTrainedModel,
)
from .models.vit import ViTForImageClassification, ViTForMaskedImageModeling, ViTModel, ViTPreTrainedModel
from .models.vit_mae import ViTMAEForPreTraining, ViTMAEModel, ViTMAEPreTrainedModel
from .models.vit_msn import ViTMSNForImageClassification, ViTMSNModel, ViTMSNPreTrainedModel
from .models.vitdet import VitDetBackbone, VitDetModel, VitDetPreTrainedModel
from .models.vitmatte import VitMatteForImageMatting, VitMattePreTrainedModel
from .models.vitpose import VitPoseForPoseEstimation, VitPosePreTrainedModel
from .models.vitpose_backbone import VitPoseBackbone, VitPoseBackbonePreTrainedModel
from .models.vits import VitsModel, VitsPreTrainedModel
from .models.vivit import VivitForVideoClassification, VivitModel, VivitPreTrainedModel
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
from .models.wav2vec2_bert import (
    Wav2Vec2BertForAudioFrameClassification,
    Wav2Vec2BertForCTC,
    Wav2Vec2BertForSequenceClassification,
    Wav2Vec2BertForXVector,
    Wav2Vec2BertModel,
    Wav2Vec2BertPreTrainedModel,
)
from .models.wav2vec2_conformer import (
    Wav2Vec2ConformerForAudioFrameClassification,
    Wav2Vec2ConformerForCTC,
    Wav2Vec2ConformerForPreTraining,
    Wav2Vec2ConformerForSequenceClassification,
    Wav2Vec2ConformerForXVector,
    Wav2Vec2ConformerModel,
    Wav2Vec2ConformerPreTrainedModel,
)
from .models.wavlm import (
    WavLMForAudioFrameClassification,
    WavLMForCTC,
    WavLMForSequenceClassification,
    WavLMForXVector,
    WavLMModel,
    WavLMPreTrainedModel,
)
from .models.whisper import (
    WhisperForAudioClassification,
    WhisperForCausalLM,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperPreTrainedModel,
    WhisperProcessor,
)
from .models.x_clip import XCLIPModel, XCLIPPreTrainedModel, XCLIPTextModel, XCLIPVisionModel
from .models.xglm import XGLMForCausalLM, XGLMModel, XGLMPreTrainedModel
from .models.xlm import (
    XLMForMultipleChoice,
    XLMForQuestionAnswering,
    XLMForQuestionAnsweringSimple,
    XLMForSequenceClassification,
    XLMForTokenClassification,
    XLMModel,
    XLMPreTrainedModel,
    XLMWithLMHeadModel,
)
from .models.xlm_roberta import (
    XLMRobertaForCausalLM,
    XLMRobertaForMaskedLM,
    XLMRobertaForMultipleChoice,
    XLMRobertaForQuestionAnswering,
    XLMRobertaForSequenceClassification,
    XLMRobertaForTokenClassification,
    XLMRobertaModel,
    XLMRobertaPreTrainedModel,
)
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
from .models.xlnet import (
    XLNetForMultipleChoice,
    XLNetForQuestionAnswering,
    XLNetForQuestionAnsweringSimple,
    XLNetForSequenceClassification,
    XLNetForTokenClassification,
    XLNetLMHeadModel,
    XLNetModel,
    XLNetPreTrainedModel,
)
from .models.xmod import (
    XmodForCausalLM,
    XmodForMaskedLM,
    XmodForMultipleChoice,
    XmodForQuestionAnswering,
    XmodForSequenceClassification,
    XmodForTokenClassification,
    XmodModel,
    XmodPreTrainedModel,
)
from .models.yolos import YolosForObjectDetection, YolosImageProcessor, YolosModel, YolosPreTrainedModel
from .models.yoso import (
    YosoForMaskedLM,
    YosoForMultipleChoice,
    YosoForQuestionAnswering,
    YosoForSequenceClassification,
    YosoForTokenClassification,
    YosoLayer,
    YosoModel,
    YosoPreTrainedModel,
)
from .models.zamba import ZambaForCausalLM, ZambaForSequenceClassification, ZambaModel, ZambaPreTrainedModel
from .models.zamba2 import Zamba2ForCausalLM, Zamba2ForSequenceClassification, Zamba2Model, Zamba2PreTrainedModel
from .models.zoedepth import ZoeDepthForDepthEstimation, ZoeDepthPreTrainedModel
from .pipelines import (
    ImageToImagePipeline,
    ImageToTextPipeline,
    TextGenerationPipeline,
    VisualQuestionAnsweringPipeline,
    pipeline,
)
from .processing_utils import ProcessorMixin
from .trainer import Trainer
from .training_args import TrainingArguments
from .utils import logging
from .video_processing_utils import BaseVideoProcessor

if version.parse(transformers.__version__) >= version.parse("4.51.0"):
    from .models.qwen3 import Qwen3ForCausalLM, Qwen3Model, Qwen3PreTrainedModel
    from .models.qwen3_moe import (
        Qwen3MoeForCausalLM,
        Qwen3MoeForQuestionAnswering,
        Qwen3MoeForSequenceClassification,
        Qwen3MoeForTokenClassification,
        Qwen3MoeModel,
        Qwen3MoePreTrainedModel,
    )

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
        Glm4vImageProcessor,
        Glm4vModel,
        Glm4vPreTrainedModel,
        Glm4vProcessor,
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
    from .models.vjepa2 import VJEPA2ForVideoClassification, VJEPA2Model, VJEPA2PreTrainedModel

if version.parse(transformers.__version__) >= version.parse("4.57.0"):
    from .models.qwen3_vl import (
        Qwen3VLForConditionalGeneration,
        Qwen3VLModel,
        Qwen3VLPreTrainedModel,
        Qwen3VLProcessor,
        Qwen3VLTextModel,
        Qwen3VLVideoProcessor,
        Qwen3VLVisionModel,
    )
    from .models.qwen3_vl_moe import (
        Qwen3VLMoeForConditionalGeneration,
        Qwen3VLMoeModel,
        Qwen3VLMoePreTrainedModel,
        Qwen3VLMoeTextModel,
        Qwen3VLMoeVisionModel,
    )
