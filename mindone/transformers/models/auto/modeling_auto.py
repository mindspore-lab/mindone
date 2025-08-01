# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
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
"""Auto Model class."""

import warnings
from collections import OrderedDict

import transformers
from packaging import version
from transformers.utils import logging

from .auto_factory import _BaseAutoBackboneClass, _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from .configuration_auto import CONFIG_MAPPING_NAMES

logger = logging.get_logger(__name__)

MODEL_MAPPING_NAMES = OrderedDict(
    [
        # Base model mapping
        ("albert", "AlbertModel"),
        ("aria", "AriaForConditionalGeneration"),
        ("aria_text", "AriaTextModel"),
        ("bert", "BertModel"),
        ("bart", "BartModel"),
        ("camembert", "CamembertModel"),
        ("mvp", "MvpModel"),
        ("convbert", "ConvBertModel"),
        ("bit", "BitModel"),
        ("blip", "BlipModel"),
        ("blip-2", "Blip2Model"),
        ("starcoder2", "Starcoder2Model"),
        ("canine", "CanineModel"),
        ("chameleon", "ChameleonModel"),
        ("clap", "ClapModel"),
        ("clip", "CLIPModel"),
        ("clip_text_model", "CLIPTextModel"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("deberta", "DebertaModel"),
        ("opt", "OPTModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("dpt", "DPTModel"),
        ("gemma", "GemmaModel"),
        ("m2m_100", "M2M100Model"),
        ("gemma2", "Gemma2Model"),
        ("persimmon", "PersimmonModel"),
        ("glm", "GlmModel"),
        ("glpn", "GLPNModel"),
        ("gpt2", "GPT2Model"),
        ("granite", "GraniteModel"),
        ("granitemoe", "GraniteMoeModel"),
        ("granitemoeshared", "GraniteMoeSharedModel"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("recurrent_gemma", "RecurrentGemmaModel"),
        ("gemma3_text", "Gemma3TextModel"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("siglip", "SiglipModel"),
        ("helium", "HeliumModel"),
        ("hiera", "HieraModel"),
        ("hubert", "HubertModel"),
        ("idefics", "IdeficsModel"),
        ("idefics2", "Idefics2Model"),
        ("idefics3", "Idefics3Model"),
        ("idefics3_vision", "Idefics3VisionTransformer"),
        ("ijepa", "IJepaModel"),
        ("imagegpt", "ImageGPTModel"),
        ("led", "LEDModel"),
        ("levit", "LevitModel"),
        ("llama", "LlamaModel"),
        ("mistral", "MistralModel"),
        ("mobilebert", "MobileBertModel"),
        ("mpt", "MptModel"),
        ("mt5", "MT5Model"),
        ("megatron-bert", "MegatronBertModel"),
        ("mixtral", "MixtralModel"),
        ("phi", "PhiModel"),
        ("phi3", "Phi3Model"),
        ("qwen2", "Qwen2Model"),
        ("qwen2_5_vl", "Qwen2_5_VLModel"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("qwen2_vl", "Qwen2VLModel"),
        ("roberta", "RobertaModel"),
        ("rembert", "RemBertModel"),
        ("siglip", "SiglipModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("smolvlm", "SmolVLMModel"),
        ("smolvlm_vision", "SmolVLMVisionTransformer"),
        ("speecht5", "SpeechT5Model"),
        ("t5", "T5Model"),
        ("umt5", "UMT5Model"),
        ("vit", "ViTModel"),
        ("wav2vec2", "Wav2Vec2Model"),
        ("whisper", "WhisperModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
        ("cohere2", "Cohere2Model"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "AlbertForPreTraining"),
        ("bart", "BartForConditionalGeneration"),
        ("camembert", "CamembertForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("bert", "BertForPreTraining"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gemma3", "Gemma3ForConditionalGeneration"),
        ("hiera", "HieraForPreTraining"),
        ("hubert", "HubertForPreTraining"),
        ("idefics", "IdeficsForVisionText2Text"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("mobilebert", "MobileBertForPreTraining"),
        ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
        ("roberta", "RobertaForMaskedLM"),
        ("megatron-bert", "MegatronBertForPreTraining"),
        ("mpt", "MptForCausalLM"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("t5", "T5ForConditionalGeneration"),
        ("wav2vec2", "Wav2Vec2ForPreTraining"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("albert", "AlbertForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("bart", "BartForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("bert", "BertForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("convbert", "ConvBertForMaskedLM"),
        ("gpt2", "GPT2LMHeadModel"),
        ("led", "LEDForConditionalGeneration"),
        ("camembert", "CamembertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("mpt", "MptForCausalLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("t5", "T5ForConditionalGeneration"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("whisper", "WhisperForConditionalGeneration"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("aria_text", "AriaTextForCausalLM"),
        ("bart", "BartForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("opt", "OPTForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("gemma", "GemmaForCausalLM"),
        ("gemma2", "Gemma2ForCausalLM"),
        ("starcoder2", "Starcoder2ForCausalLM"),
        ("gemma3", "Gemma3ForCausalLM"),
        ("gemma3_text", "Gemma3ForCausalLM"),
        ("granite", "GraniteForCausalLM"),
        ("glm", "GlmForCausalLM"),
        ("gpt2", "GPT2LMHeadModel"),
        ("persimmon", "PersimmonForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("granitemoe", "GraniteMoeForCausalLM"),
        ("granitemoeshared", "GraniteMoeSharedForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mpt", "MptForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("phi3", "Phi3ForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("recurrent_gemma", "RecurrentGemmaForCausalLM"),
        ("rembert", "RemBertForCausalLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("cohere2", "Cohere2ForCausalLM"),
    ]
)

MODEL_FOR_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image mapping
        ("bit", "BitModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("dpt", "DPTModel"),
        ("glpn", "GLPNModel"),
        ("hiera", "HieraModel"),
        ("hubert", "HubertModel"),
        ("ijepa", "IJepaModel"),
        ("imagegpt", "ImageGPTModel"),
        ("levit", "LevitModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("vit", "ViTModel"),
    ]
)

MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("vit", "ViTForMaskedImageModeling"),
    ]
)


MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("imagegpt", "ImageGPTForCausalImageModeling"),
    ]
)

MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image Classification mapping
        ("bit", "BitForImageClassification"),
        ("clip", "CLIPForImageClassification"),
        ("hiera", "HieraForImageClassification"),
        ("ijepa", "IJepaForImageClassification"),
        ("imagegpt", "ImageGPTForImageClassification"),
        (
            "levit",
            ("LevitForImageClassification", "LevitForImageClassificationWithTeacher"),
        ),
        ("siglip", "SiglipForImageClassification"),
        ("vit", "ViTForImageClassification"),
    ]
)

MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForConditionalGeneration"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
    ]
)
MODEL_FOR_RETRIEVAL_MAPPING_NAMES = OrderedDict(
    [
        # ("colpali", "ColPaliForRetrieval"),
    ]
)
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = OrderedDict(
    [
        ("aria", "AriaForConditionalGeneration"),
        ("blip", "BlipForConditionalGeneration"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("gemma3", "Gemma3ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("idefics", "IdeficsForVisionText2Text"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("fuyu", "FuyuForCausalLM"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
        ("smolvlm", "SmolVLMForConditionalGeneration"),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("mvp", "MvpForConditionalGeneration"),
        ("albert", "AlbertForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("convbert", "ConvBertForMaskedLM"),
        ("bert", "BertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
    ]
)

MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Object Detection mapping
        ("conditional_detr", "ConditionalDetrForObjectDetection"),
        ("deformable_detr", "DeformableDetrForObjectDetection"),
        ("deta", "DetaForObjectDetection"),
        ("detr", "DetrForObjectDetection"),
        ("rt_detr", "RTDetrForObjectDetection"),
        ("table-transformer", "TableTransformerForObjectDetection"),
        ("yolos", "YolosForObjectDetection"),
    ]
)

MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Object Detection mapping
        ("grounding-dino", "GroundingDinoForObjectDetection"),
        ("omdet-turbo", "OmDetTurboForObjectDetection"),
        ("owlv2", "Owlv2ForObjectDetection"),
        ("owlvit", "OwlViTForObjectDetection"),
    ]
)

MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for depth estimation mapping
        ("depth_anything", "DepthAnythingForDepthEstimation"),
        ("dpt", "DPTForDepthEstimation"),
        ("glpn", "GLPNForDepthEstimation"),
        ("zoedepth", "ZoeDepthForDepthEstimation"),
    ]
)
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "BartForConditionalGeneration"),
        ("led", "LEDForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("mvp", "MvpForConditionalGeneration"),
        ("mt5", "MT5ForConditionalGeneration"),
        ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
        ("t5", "T5ForConditionalGeneration"),
        ("umt5", "UMT5ForConditionalGeneration"),
    ]
)

MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("speecht5", "SpeechT5ForSpeechToText"),
        ("whisper", "WhisperForConditionalGeneration"),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "AlbertForSequenceClassification"),
        ("bart", "BartForSequenceClassification"),
        ("camembert", "CamembertForSequenceClassification"),
        ("opt", "OPTForSequenceClassification"),
        ("bert", "BertForSequenceClassification"),
        ("mvp", "MvpForSequenceClassification"),
        ("roberta", "RobertaForSequenceClassification"),
        ("deberta", "DebertaForSequenceClassification"),
        ("deberta-v2", "DebertaV2ForSequenceClassification"),
        ("gemma", "GemmaForSequenceClassification"),
        ("gemma2", "Gemma2ForSequenceClassification"),
        ("glm", "GlmForSequenceClassification"),
        ("helium", "HeliumForSequenceClassification"),
        ("hubert", "HubertForSequenceClassification"),
        ("led", "LEDForSequenceClassification"),
        ("starcoder2", "Starcoder2ForSequenceClassification"),
        ("canine", "CanineForSequenceClassification"),
        ("llama", "LlamaForSequenceClassification"),
        ("persimmon", "PersimmonForSequenceClassification"),
        ("mobilebert", "MobileBertForSequenceClassification"),
        ("convbert", "ConvBertForSequenceClassification"),
        ("mt5", "MT5ForSequenceClassification"),
        ("megatron-bert", "MegatronBertForSequenceClassification"),
        ("mistral", "MistralForSequenceClassification"),
        ("mixtral", "MixtralForSequenceClassification"),
        ("mpt", "MptForSequenceClassification"),
        ("phi", "PhiForSequenceClassification"),
        ("phi3", "Phi3ForSequenceClassification"),
        ("qwen2", "Qwen2ForSequenceClassification"),
        ("rembert", "RemBertForSequenceClassification"),
        ("t5", "T5ForSequenceClassification"),
        ("umt5", "UMT5ForSequenceClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForSequenceClassification"),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "AlbertForQuestionAnswering"),
        ("bart", "BartForQuestionAnswering"),
        ("opt", "OPTForQuestionAnswering"),
        ("bert", "BertForQuestionAnswering"),
        ("camembert", "CamembertForQuestionAnswering"),
        ("mvp", "MvpForQuestionAnswering"),
        ("roberta", "RobertaForQuestionAnswering"),
        ("deberta", "DebertaForQuestionAnswering"),
        ("deberta-v2", "DebertaV2ForQuestionAnswering"),
        ("led", "LEDForQuestionAnswering"),
        ("convbert", "ConvBertForQuestionAnswering"),
        ("llama", "LlamaForQuestionAnswering"),
        ("mistral", "MistralForQuestionAnswering"),
        ("mobilebert", "MobileBertForQuestionAnswering"),
        ("megatron-bert", "MegatronBertForQuestionAnswering"),
        ("mistral", "MistralForQuestionAnswering"),
        ("qwen2", "Qwen2ForQuestionAnswering"),
        ("rembert", "RemBertForQuestionAnswering"),
        ("t5", "T5ForQuestionAnswering"),
        ("mixtral", "MixtralForQuestionAnswering"),
        ("mpt", "MptForQuestionAnswering"),
        ("canine", "CanineForQuestionAnswering"),
        ("umt5", "UMT5ForQuestionAnswering"),
        ("xlm-roberta", "XLMRobertaForQuestionAnswering"),
        ("xlm-roberta-xl", "XLMRobertaXLForQuestionAnswering"),
    ]
)

MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict()

MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForQuestionAnswering"),
    ]
)

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict()

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "AlbertForTokenClassification"),
        ("bert", "BertForTokenClassification"),
        ("camembert", "CamembertForTokenClassification"),
        ("deberta", "DebertaForTokenClassification"),
        ("deberta-v2", "DebertaV2ForTokenClassification"),
        ("starcoder2", "Starcoder2ForTokenClassification"),
        ("glm", "GlmForTokenClassification"),
        ("helium", "HeliumForTokenClassification"),
        ("mistral", "MistralForTokenClassification"),
        ("mobilebert", "MobileBertForTokenClassification"),
        ("mt5", "MT5ForTokenClassification"),
        ("persimmon", "PersimmonForTokenClassification"),
        ("megatron-bert", "MegatronBertForTokenClassification"),
        ("mixtral", "MixtralForTokenClassification"),
        ("mpt", "MptForTokenClassification"),
        ("phi", "PhiForTokenClassification"),
        ("phi3", "Phi3ForTokenClassification"),
        ("qwen2", "Qwen2ForTokenClassification"),
        ("roberta", "RobertaForTokenClassification"),
        ("rembert", "RemBertForTokenClassification"),
        ("convbert", "ConvBertForTokenClassification"),
        ("canine", "CanineForTokenClassification"),
        ("t5", "T5ForTokenClassification"),
        ("umt5", "UMT5ForTokenClassification"),
        ("xlm-roberta", "XLMRobertaForTokenClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForTokenClassification"),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("camembert", "CamembertForMultipleChoice"),
        ("albert", "AlbertForMultipleChoice"),
        ("convbert", "ConvBertForMultipleChoice"),
        ("canine", "CanineForMultipleChoice"),
        ("bert", "BertForMultipleChoice"),
        ("roberta", "RobertaForMultipleChoice"),
        ("deberta-v2", "DebertaV2ForMultipleChoice"),
        ("megatron-bert", "MegatronBertForMultipleChoice"),
        ("mobilebert", "MobileBertForMultipleChoice"),
        ("rembert", "RemBertForMultipleChoice"),
        ("xlm-roberta", "XLMRobertaForMultipleChoice"),
        ("xlm-roberta-xl", "XLMRobertaXLForMultipleChoice"),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertForNextSentencePrediction"),
        ("megatron-bert", "MegatronBertForNextSentencePrediction"),
        ("mobilebert", "MobileBertForNextSentencePrediction"),
    ]
)

MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("wav2vec2", "Wav2Vec2ForSequenceClassification"),
        ("whisper", "WhisperForAudioClassification"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        ("wav2vec2", "Wav2Vec2ForCTC"),
    ]
)

MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("wav2vec2", "Wav2Vec2ForAudioFrameClassification"),
    ]
)

MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict(
    [
        ("wav2vec2", "Wav2Vec2ForXVector"),
    ]
)

MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = OrderedDict(
    [
        ("siglip", "SiglipModel"),
        ("speecht5", "SpeechT5ForTextToSpeech"),
    ]
)

MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = OrderedDict()

MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Image Classification mapping
        ("blip", "BlipModel"),
        ("siglip", "SiglipModel"),
    ]
)

MODEL_FOR_BACKBONE_MAPPING_NAMES = OrderedDict(
    [
        ("hiera", "HieraBackbone"),
        ("swin", "SwinBackbone"),
    ]
)

MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict()


MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES = OrderedDict()


MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        ("albert", "AlbertModel"),
        ("bert", "BertModel"),
        ("roberta", "RobertaModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("mobilebert", "MobileBertModel"),
        ("mt5", "MT5EncoderModel"),
        ("rembert", "RemBertModel"),
        ("t5", "T5EncoderModel"),
        ("umt5", "UMT5EncoderModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
    ]
)

MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = OrderedDict()


if version.parse(transformers.__version__) >= version.parse("4.51.0"):
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.update({"qwen3": "Qwen3Model"})
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.update({"qwen3": "Qwen3ForCausalLM"})
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.update({"qwen3": "Qwen3ForSequenceClassification"})
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.update({"qwen3": "Qwen3ForQuestionAnswering"})
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.update({"qwen3": "Qwen3ForTokenClassification"})

if version.parse(transformers.__version__) >= version.parse("4.51.3"):
    MODEL_MAPPING_NAMES.update({"glm4": "Glm4Model"})
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.update({"glm4": "Glm4ForCausalLM"})
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.update({"glm4": "Glm4ForSequenceClassification"})
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.update({"glm4": "Glm4ForTokenClassification"})

if version.parse(transformers.__version__) >= version.parse("4.53.0"):
    MODEL_MAPPING_NAMES.update({"minimax": "MiniMaxModel", "vjepa2": "VJEPA2Model"})
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.update({"minimax": "MiniMaxForCausalLM"})
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES.update({"vjepa2": "VJEPA2ForVideoClassification"})
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.update({"minimax": "MiniMaxForSequenceClassification"})
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.update({"minimax": "MiniMaxForQuestionAnswering"})
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.update({"minimax": "MiniMaxForTokenClassification"})

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)
MODEL_FOR_PRETRAINING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_PRETRAINING_MAPPING_NAMES)
MODEL_WITH_LM_HEAD_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_WITH_LM_HEAD_MAPPING_NAMES)
MODEL_FOR_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES)
MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_CAUSAL_IMAGE_MODELING_MAPPING_NAMES
)
MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_IMAGE_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES
)
MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_VISION_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES)
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
)
MODEL_FOR_RETRIEVAL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_RETRIEVAL_MAPPING_NAMES)
MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_MASKED_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_LM_MAPPING_NAMES)
MODEL_FOR_IMAGE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_MAPPING_NAMES)
MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES
)
MODEL_FOR_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES)
MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING_NAMES
)
MODEL_FOR_DEPTH_ESTIMATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES)
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES
)
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES
)
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES)
MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_CTC_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_CTC_MAPPING_NAMES)
MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES)
MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES
)
MODEL_FOR_AUDIO_XVECTOR_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES)

MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES
)

MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES)

MODEL_FOR_BACKBONE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_BACKBONE_MAPPING_NAMES)

MODEL_FOR_MASK_GENERATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_MASK_GENERATION_MAPPING_NAMES)

MODEL_FOR_KEYPOINT_DETECTION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES
)

MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES)

MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES
)

MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES
)

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)


class AutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASK_GENERATION_MAPPING


class AutoModelForKeypointDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_KEYPOINT_DETECTION_MAPPING


class AutoModelForTextEncoding(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_ENCODING_MAPPING


class AutoModelForImageToImage(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TO_IMAGE_MAPPING


class AutoModel(_BaseAutoModelClass):
    _model_mapping = MODEL_MAPPING


AutoModel = auto_class_update(AutoModel)


class AutoModelForPreTraining(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_PRETRAINING_MAPPING


AutoModelForPreTraining = auto_class_update(AutoModelForPreTraining, head_doc="pretraining")


# Private on purpose, the public class will add the deprecation warnings.
class _AutoModelWithLMHead(_BaseAutoModelClass):
    _model_mapping = MODEL_WITH_LM_HEAD_MAPPING


_AutoModelWithLMHead = auto_class_update(_AutoModelWithLMHead, head_doc="language modeling")


class AutoModelForCausalLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CAUSAL_LM_MAPPING


AutoModelForCausalLM = auto_class_update(AutoModelForCausalLM, head_doc="causal language modeling")


class AutoModelForMaskedLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_LM_MAPPING


AutoModelForMaskedLM = auto_class_update(AutoModelForMaskedLM, head_doc="masked language modeling")


class AutoModelForSeq2SeqLM(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING


AutoModelForSeq2SeqLM = auto_class_update(
    AutoModelForSeq2SeqLM,
    head_doc="sequence-to-sequence language modeling",
    checkpoint_for_example="google-t5/t5-base",
)


class AutoModelForSequenceClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING


AutoModelForSequenceClassification = auto_class_update(
    AutoModelForSequenceClassification, head_doc="sequence classification"
)


class AutoModelForQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_QUESTION_ANSWERING_MAPPING


AutoModelForQuestionAnswering = auto_class_update(AutoModelForQuestionAnswering, head_doc="question answering")


class AutoModelForTableQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING


AutoModelForTableQuestionAnswering = auto_class_update(
    AutoModelForTableQuestionAnswering,
    head_doc="table question answering",
    checkpoint_for_example="google/tapas-base-finetuned-wtq",
)


class AutoModelForVisualQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING


AutoModelForVisualQuestionAnswering = auto_class_update(
    AutoModelForVisualQuestionAnswering,
    head_doc="visual question answering",
    checkpoint_for_example="dandelin/vilt-b32-finetuned-vqa",
)


class AutoModelForDocumentQuestionAnswering(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING


AutoModelForDocumentQuestionAnswering = auto_class_update(
    AutoModelForDocumentQuestionAnswering,
    head_doc="document question answering",
    checkpoint_for_example='impira/layoutlm-document-qa", revision="52e01b3',
)


class AutoModelForTokenClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING


AutoModelForTokenClassification = auto_class_update(AutoModelForTokenClassification, head_doc="token classification")


class AutoModelForMultipleChoice(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MULTIPLE_CHOICE_MAPPING


AutoModelForMultipleChoice = auto_class_update(AutoModelForMultipleChoice, head_doc="multiple choice")


class AutoModelForNextSentencePrediction(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING


AutoModelForNextSentencePrediction = auto_class_update(
    AutoModelForNextSentencePrediction, head_doc="next sentence prediction"
)


class AutoModelForImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING


AutoModelForImageClassification = auto_class_update(AutoModelForImageClassification, head_doc="image classification")


class AutoModelForZeroShotImageClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING


AutoModelForZeroShotImageClassification = auto_class_update(
    AutoModelForZeroShotImageClassification, head_doc="zero-shot image classification"
)


class AutoModelForImageSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_SEGMENTATION_MAPPING


AutoModelForImageSegmentation = auto_class_update(AutoModelForImageSegmentation, head_doc="image segmentation")


class AutoModelForSemanticSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING


AutoModelForSemanticSegmentation = auto_class_update(AutoModelForSemanticSegmentation, head_doc="semantic segmentation")


class AutoModelForUniversalSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING


AutoModelForUniversalSegmentation = auto_class_update(
    AutoModelForUniversalSegmentation, head_doc="universal image segmentation"
)


class AutoModelForInstanceSegmentation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING


AutoModelForInstanceSegmentation = auto_class_update(AutoModelForInstanceSegmentation, head_doc="instance segmentation")


class AutoModelForObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_OBJECT_DETECTION_MAPPING


AutoModelForObjectDetection = auto_class_update(AutoModelForObjectDetection, head_doc="object detection")


class AutoModelForZeroShotObjectDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_ZERO_SHOT_OBJECT_DETECTION_MAPPING


AutoModelForZeroShotObjectDetection = auto_class_update(
    AutoModelForZeroShotObjectDetection, head_doc="zero-shot object detection"
)


class AutoModelForDepthEstimation(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_DEPTH_ESTIMATION_MAPPING


AutoModelForDepthEstimation = auto_class_update(AutoModelForDepthEstimation, head_doc="depth estimation")


class AutoModelForVideoClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING


AutoModelForVideoClassification = auto_class_update(AutoModelForVideoClassification, head_doc="video classification")


class AutoModelForVision2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_VISION_2_SEQ_MAPPING


AutoModelForVision2Seq = auto_class_update(AutoModelForVision2Seq, head_doc="vision-to-text modeling")


class AutoModelForImageTextToText(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING


AutoModelForImageTextToText = auto_class_update(AutoModelForImageTextToText, head_doc="image-text-to-text modeling")


class AutoModelForAudioClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING


AutoModelForAudioClassification = auto_class_update(AutoModelForAudioClassification, head_doc="audio classification")


class AutoModelForCTC(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_CTC_MAPPING


AutoModelForCTC = auto_class_update(AutoModelForCTC, head_doc="connectionist temporal classification")


class AutoModelForSpeechSeq2Seq(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING


AutoModelForSpeechSeq2Seq = auto_class_update(
    AutoModelForSpeechSeq2Seq, head_doc="sequence-to-sequence speech-to-text modeling"
)


class AutoModelForAudioFrameClassification(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING


AutoModelForAudioFrameClassification = auto_class_update(
    AutoModelForAudioFrameClassification, head_doc="audio frame (token) classification"
)


class AutoModelForAudioXVector(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_XVECTOR_MAPPING


class AutoModelForTextToSpectrogram(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING


class AutoModelForTextToWaveform(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING


class AutoBackbone(_BaseAutoBackboneClass):
    _model_mapping = MODEL_FOR_BACKBONE_MAPPING


AutoModelForAudioXVector = auto_class_update(AutoModelForAudioXVector, head_doc="audio retrieval via x-vector")


class AutoModelForMaskedImageModeling(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING


AutoModelForMaskedImageModeling = auto_class_update(AutoModelForMaskedImageModeling, head_doc="masked image modeling")


class AutoModelWithLMHead(_AutoModelWithLMHead):
    @classmethod
    def from_config(cls, config):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_config(config)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        warnings.warn(
            "The class `AutoModelWithLMHead` is deprecated and will be removed in a future version. Please use "
            "`AutoModelForCausalLM` for causal language models, `AutoModelForMaskedLM` for masked language models and "
            "`AutoModelForSeq2SeqLM` for encoder-decoder models.",
            FutureWarning,
        )
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
