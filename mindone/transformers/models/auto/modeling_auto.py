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
        ("align", "AlignModel"),
        ("altclip", "AltCLIPModel"),
        ("aria", "AriaForConditionalGeneration"),
        ("aria_text", "AriaTextModel"),
        ("bamba", "BambaModel"),
        ("bert", "BertModel"),
        ("bert-generation", "BertGenerationEncoder"),
        ("bart", "BartModel"),
        ("camembert", "CamembertModel"),
        ("mvp", "MvpModel"),
        ("nllb-moe", "NllbMoeModel"),
        ("convbert", "ConvBertModel"),
        ("bert", "BertModel"),
        ("bit", "BitModel"),
        ("blenderbot", "BlenderbotModel"),
        ("blenderbot-small", "BlenderbotSmallModel"),
        ("blip", "BlipModel"),
        ("blip-2", "Blip2Model"),
        ("bloom", "BloomModel"),
        ("camembert", "CamembertModel"),
        ("canine", "CanineModel"),
        ("chameleon", "ChameleonModel"),
        ("chinese_clip", "ChineseCLIPModel"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionModel"),
        ("clap", "ClapModel"),
        ("clip", "CLIPModel"),
        ("clip_text_model", "CLIPTextModel"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("clipseg", "CLIPSegModel"),
        ("clvp", "ClvpModelForConditionalGeneration"),
        ("codegen", "CodeGenModel"),
        ("cohere2", "Cohere2Model"),
        ("convbert", "ConvBertModel"),
        ("convnext", "ConvNextModel"),
        ("convnextv2", "ConvNextV2Model"),
        ("ctrl", "CTRLModel"),
        ("cvt", "CvtModel"),
        ("dac", "DacModel"),
        ("data2vec-audio", "Data2VecAudioModel"),
        ("data2vec-text", "Data2VecTextModel"),
        ("data2vec-vision", "Data2VecVisionModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("deit", "DeiTModel"),
        ("depth_pro", "DepthProModel"),
        ("detr", "DetrModel"),
        ("diffllama", "DiffLlamaModel"),
        ("dinov2", "Dinov2Model"),
        ("distilbert", "DistilBertModel"),
        ("dpr", "DPRQuestionEncoder"),
        ("dpt", "DPTModel"),
        ("efficientnet", "EfficientNetModel"),
        ("electra", "ElectraModel"),
        ("encodec", "EncodecModel"),
        ("falcon", "FalconModel"),
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),
        ("fsmt", "FSMTModel"),
        ("funnel", ("FunnelModel", "FunnelBaseModel")),
        ("gemma", "GemmaModel"),
        ("gemma2", "Gemma2Model"),
        ("gemma3_text", "Gemma3TextModel"),
        ("git", "GitModel"),
        ("glm", "GlmModel"),
        ("glpn", "GLPNModel"),
        ("got_ocr2", "GotOcr2ForConditionalGeneration"),
        ("gpt_neo", "GPTNeoModel"),
        ("gpt2", "GPT2Model"),
        ("granite", "GraniteModel"),
        ("granitemoe", "GraniteMoeModel"),
        ("granitemoeshared", "GraniteMoeSharedModel"),
        ("groupvit", "GroupViTModel"),
        ("helium", "HeliumModel"),
        ("grounding-dino", "GroundingDinoModel"),
        ("hiera", "HieraModel"),
        ("hubert", "HubertModel"),
        ("ibert", "IBertModel"),
        ("idefics", "IdeficsModel"),
        ("idefics2", "Idefics2Model"),
        ("idefics3", "Idefics3Model"),
        ("idefics3_vision", "Idefics3VisionTransformer"),
        ("ijepa", "IJepaModel"),
        ("imagegpt", "ImageGPTModel"),
        ("jetmoe", "JetMoeModel"),
        ("kosmos-2", "Kosmos2Model"),
        ("led", "LEDModel"),
        ("levit", "LevitModel"),
        ("lilt", "LiltModel"),
        ("llama", "LlamaModel"),
        ("longformer", "LongformerModel"),
        ("longt5", "LongT5Model"),
        ("luke", "LukeModel"),
        ("m2m_100", "M2M100Model"),
        ("mamba", "MambaModel"),
        ("mamba2", "Mamba2Model"),
        ("mbart", "MBartModel"),
        ("megatron-bert", "MegatronBertModel"),
        ("mgp-str", "MgpstrForSceneTextRecognition"),
        ("mimi", "MimiModel"),
        ("mistral", "MistralModel"),
        ("mixtral", "MixtralModel"),
        ("mobilebert", "MobileBertModel"),
        ("moonshine", "MoonshineModel"),
        ("moshi", "MoshiModel"),
        ("mpnet", "MPNetModel"),
        ("mpt", "MptModel"),
        ("mt5", "MT5Model"),
        ("mvp", "MvpModel"),
        ("nystromformer", "NystromformerModel"),
        ("olmoe", "OlmoeModel"),
        ("opt", "OPTModel"),
        ("owlv2", "Owlv2Model"),
        ("owlvit", "OwlViTModel"),
        ("pegasus", "PegasusModel"),
        ("pegasus_x", "PegasusXModel"),
        ("persimmon", "PersimmonModel"),
        ("olmo", "OlmoModel"),
        ("olmo2", "Olmo2Model"),
        ("phi", "PhiModel"),
        ("phi3", "Phi3Model"),
        ("pixtral", "PixtralVisionModel"),
        ("poolformer", "PoolFormerModel"),
        ("prophetnet", "ProphetNetModel"),
        ("qwen2", "Qwen2Model"),
        ("qwen2_5_vl", "Qwen2_5_VLModel"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("qwen2_vl", "Qwen2VLModel"),
        ("recurrent_gemma", "RecurrentGemmaModel"),
        ("rembert", "RemBertModel"),
        ("resnet", "ResNetModel"),
        ("roberta", "RobertaModel"),
        ("rwkv", "RwkvModel"),
        ("sam", "SamModel"),
        ("seamless_m4t_v2", "SeamlessM4Tv2Model"),
        ("segformer", "SegformerModel"),
        ("sew", "SEWModel"),
        ("sew-d", "SEWDModel"),
        ("siglip", "SiglipModel"),
        ("siglip2", "Siglip2Model"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("smolvlm", "SmolVLMModel"),
        ("smolvlm_vision", "SmolVLMVisionTransformer"),
        ("speech_to_text", "Speech2TextModel"),
        ("speecht5", "SpeechT5Model"),
        ("squeezebert", "SqueezeBertModel"),
        ("stablelm", "StableLmModel"),
        ("starcoder2", "Starcoder2Model"),
        ("swin2sr", "Swin2SRModel"),
        ("swinv2", "Swinv2Model"),
        ("t5", "T5Model"),
        ("tapas", "TapasModel"),
        ("timesformer", "TimesformerModel"),
        ("tvp", "TvpModel"),
        ("umt5", "UMT5Model"),
        ("unispeech", "UniSpeechModel"),
        ("unispeech-sat", "UniSpeechSatModel"),
        ("univnet", "UnivNetModel"),
        ("vilt", "ViltModel"),
        ("visual_bert", "VisualBertModel"),
        ("vit", "ViTModel"),
        ("vit_msn", "ViTMSNModel"),
        ("vitdet", "VitDetModel"),
        ("vivit", "VivitModel"),
        ("wav2vec2", "Wav2Vec2Model"),
        ("whisper", "WhisperModel"),
        ("xclip", "XCLIPModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("gpt_bigcode", "GPTBigCodeModel"),
        ("gpt_neox", "GPTNeoXModel"),
        ("gptj", "GPTJModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
        ("mobilenet_v1", "MobileNetV1Model"),
        ("mobilenet_v2", "MobileNetV2Model"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseModel"),
        ("yolos", "YolosModel"),
        ("zamba", "ZambaModel"),
        ("zamba2", "Zamba2Model"),
    ]
)

MODEL_FOR_PRETRAINING_MAPPING_NAMES = OrderedDict(
    [
        # Model for pre-training mapping
        ("albert", "AlbertForPreTraining"),
        ("bart", "BartForConditionalGeneration"),
        ("bert", "BertForPreTraining"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("colpali", "ColPaliForRetrieval"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForPreTraining"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("funnel", "FunnelForPreTraining"),
        ("gpt2", "GPT2LMHeadModel"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("mbart", "MBartForConditionalGeneration"),
        ("gemma3", "Gemma3ForConditionalGeneration"),
        ("hiera", "HieraForPreTraining"),
        ("hubert", "HubertForPreTraining"),
        ("ibert", "IBertForMaskedLM"),
        ("idefics", "IdeficsForVisionText2Text"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("luke", "LukeForMaskedLM"),
        ("mobilebert", "MobileBertForPreTraining"),
        ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
        ("roberta", "RobertaForMaskedLM"),
        ("megatron-bert", "MegatronBertForPreTraining"),
        ("mistral3", "Mistral3ForConditionalGeneration"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("mobilebert", "MobileBertForPreTraining"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mpt", "MptForCausalLM"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("mvp", "MvpForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
        ("roberta", "RobertaForMaskedLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("t5", "T5ForConditionalGeneration"),
        ("tapas", "TapasForMaskedLM"),
        ("unispeech", "UniSpeechForPreTraining"),
        ("unispeec-sat", "UniSpeechSatForPreTraining"),
        ("video_llava", "VideoLlavaForConditionalGeneration"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        ("visual_bert", "VisualBertForPreTraining"),
        ("wav2vec2", "Wav2Vec2ForPreTraining"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("gpt_neox_japanese", "GPTNeoXJapanesePreTrainedModel"),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("albert", "AlbertForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("bart", "BartForConditionalGeneration"),
        ("bloom", "BloomForCausalLM"),
        ("bert", "BertForMaskedLM"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("camembert", "CamembertForMaskedLM"),
        ("convbert", "ConvBertForMaskedLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForMaskedLM"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("funnel", "FunnelForMaskedLM"),
        ("git", "GitForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt2", "GPT2LMHeadModel"),
        ("ibert", "IBertForMaskedLM"),
        ("led", "LEDForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("luke", "LukeForMaskedLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("mbart", "MBartForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("moonshine", "MoonshineForConditionalGeneration"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mpt", "MptForCausalLM"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("pegasus_x", "PegasusXForConditionalGeneration"),
        ("pop2piano", "Pop2PianoForConditionalGeneration"),
        ("rembert", "RemBertForMaskedLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("t5", "T5ForConditionalGeneration"),
        ("tapas", "TapasForMaskedLM"),
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
        ("bamba", "BambaForCausalLM"),
        ("bart", "BartForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("cohere2", "Cohere2ForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForCausalLM"),
        ("diffllama", "DiffLlamaForCausalLM"),
        ("emu3", "Emu3ForCausalLM"),
        ("falcon", "FalconForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("gemma2", "Gemma2ForCausalLM"),
        ("gemma3", "Gemma3ForCausalLM"),
        ("gemma3_text", "Gemma3ForCausalLM"),
        ("git", "GitForCausalLM"),
        ("glm", "GlmForCausalLM"),
        ("got_ocr2", "GotOcr2ForConditionalGeneration"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt2", "GPT2LMHeadModel"),
        ("granite", "GraniteForCausalLM"),
        ("jetmoe", "JetMoeForCausalLM"),
        ("persimmon", "PersimmonForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("granitemoe", "GraniteMoeForCausalLM"),
        ("granitemoeshared", "GraniteMoeSharedForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("mllama", "MllamaForCausalLM"),
        ("moshi", "MoshiForCausalLM"),
        ("mpt", "MptForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("opt", "OPTForCausalLM"),
        ("persimmon", "PersimmonForCausalLM"),
        ("olmo", "OlmoForCausalLM"),
        ("olmo2", "Olmo2ForCausalLM"),
        ("olmoe", "OlmoeForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("phi3", "Phi3ForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("recurrent_gemma", "RecurrentGemmaForCausalLM"),
        ("rembert", "RemBertForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("stablelm", "StableLmForCausalLM"),
        ("starcoder2", "Starcoder2ForCausalLM"),
        ("trocr", "TrOCRForCausalLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("cohere2", "Cohere2ForCausalLM"),
        ("zamba", "ZambaForCausalLM"),
    ]
)

MODEL_FOR_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image mapping
        ("bit", "BitModel"),
        ("convnext", "ConvNextModel"),
        ("convnextv2", "ConvNextV2Model"),
        ("data2vec-vision", "Data2VecVisionModel"),
        ("deit", "DeiTModel"),
        ("detr", "DetrModel"),
        ("depth_pro", "DepthProModel"),
        ("dinov2", "Dinov2Model"),
        ("dpt", "DPTModel"),
        ("efficientnet", "EfficientNetModel"),
        ("glpn", "GLPNModel"),
        ("hiera", "HieraModel"),
        ("hubert", "HubertModel"),
        ("ijepa", "IJepaModel"),
        ("imagegpt", "ImageGPTModel"),
        ("levit", "LevitModel"),
        ("poolformer", "PoolFormerModel"),
        ("mllama", "MllamaVisionModel"),
        ("resnet", "ResNetModel"),
        ("segformer", "SegformerModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("swin2sr", "Swin2SRModel"),
        ("swinv2", "Swinv2Model"),
        ("timesformer", "TimesformerModel"),
        ("vit", "ViTModel"),
        ("vit_msn", "ViTMSNModel"),
        ("vitdet", "VitDetModel"),
        ("vivit", "VivitModel"),
        ("yolos", "YolosModel"),
        ("zamba2", "Zamba2ForCausalLM"),
    ]
)

MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES = OrderedDict(
    [
        ("deit", "DeiTForMaskedImageModeling"),
        ("swinv2", "Swinv2ForMaskedImageModeling"),
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
        ("convnext", "ConvNextForImageClassification"),
        ("convnextv2", "ConvNextV2ForImageClassification"),
        ("cvt", "CvtForImageClassification"),
        ("data2vec-vision", "Data2VecVisionForImageClassification"),
        (
            "deit",
            ("DeiTForImageClassification", "DeiTForImageClassificationWithTeacher"),
        ),
        ("dinov2", "Dinov2ForImageClassification"),
        ("efficientnet", "EfficientNetForImageClassification"),
        ("hiera", "HieraForImageClassification"),
        ("ijepa", "IJepaForImageClassification"),
        ("imagegpt", "ImageGPTForImageClassification"),
        (
            "levit",
            ("LevitForImageClassification", "LevitForImageClassificationWithTeacher"),
        ),
        ("poolformer", "PoolFormerForImageClassification"),
        ("resnet", "ResNetForImageClassification"),
        ("segformer", "SegformerForImageClassification"),
        ("siglip", "SiglipForImageClassification"),
        ("siglip2", "Siglip2ForImageClassification"),
        ("swinv2", "Swinv2ForImageClassification"),
        ("mobilenet_v1", "MobileNetV1ForImageClassification"),
        ("mobilenet_v2", "MobileNetV2ForImageClassification"),
        ("vit", "ViTForImageClassification"),
        ("vit_msn", "ViTMSNForImageClassification"),
    ]
)

MODEL_FOR_IMAGE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Do not add new models here, this class will be deprecated in the future.
        # Model for Image Segmentation mapping
        ("detr", "DetrForSegmentation"),
    ]
)

MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Semantic Segmentation mapping
        ("beit", "BeitForSemanticSegmentation"),
        ("data2vec-vision", "Data2VecVisionForSemanticSegmentation"),
        ("dpt", "DPTForSemanticSegmentation"),
        ("mobilenet_v2", "MobileNetV2ForSemanticSegmentation"),
        ("mobilevit", "MobileViTForSemanticSegmentation"),
        ("mobilevitv2", "MobileViTV2ForSemanticSegmentation"),
        ("segformer", "SegformerForSemanticSegmentation"),
        ("upernet", "UperNetForSemanticSegmentation"),
    ]
)

MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Universal Segmentation mapping
        ("detr", "DetrForSegmentation"),
        ("mask2former", "Mask2FormerForUniversalSegmentation"),
        ("maskformer", "MaskFormerForInstanceSegmentation"),
        ("oneformer", "OneFormerForUniversalSegmentation"),
    ]
)

MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("timesformer", "TimesformerForVideoClassification"),
        ("vivit", "VivitForVideoClassification"),
    ]
)

MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForConditionalGeneration"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("git", "GitForCausalLM"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("instructblip", "InstructBlipForConditionalGeneration"),
        ("instructblipvideo", "InstructBlipVideoForConditionalGeneration"),
        ("kosmos-2", "Kosmos2ForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("mistral3", "Mistral3ForConditionalGeneration"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("prophetnet", "ProphetNetForConditionalGeneration"),
        ("pix2struct", "Pix2StructForConditionalGeneration"),
        ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
        ("video_llava", "VideoLlavaForConditionalGeneration"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        ("vision-encoder-decoder", "VisionEncoderDecoderModel"),
    ]
)
MODEL_FOR_RETRIEVAL_MAPPING_NAMES = OrderedDict(
    [
        ("colpali", "ColPaliForRetrieval"),
    ]
)
MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES = OrderedDict(
    [
        ("aria", "AriaForConditionalGeneration"),
        ("blip", "BlipForConditionalGeneration"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("gemma3", "Gemma3ForConditionalGeneration"),
        ("got_ocr2", "GotOcr2ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("emu3", "Emu3ForConditionalGeneration"),
        ("idefics", "IdeficsForVisionText2Text"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("instructblip", "InstructBlipForConditionalGeneration"),
        ("kosmos-2", "Kosmos2ForConditionalGeneration"),
        ("fuyu", "FuyuForCausalLM"),
        ("git", "GitForCausalLM"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("mistral3", "Mistral3ForConditionalGeneration"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("pix2struct", "Pix2StructForConditionalGeneration"),
        ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
        ("smolvlm", "SmolVLMForConditionalGeneration"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        ("vision-encoder-decoder", "VisionEncoderDecoderModel"),
    ]
)

MODEL_FOR_MASKED_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Masked LM mapping
        ("albert", "AlbertForMaskedLM"),
        ("bart", "BartForConditionalGeneration"),
        ("bert", "BertForMaskedLM"),
        ("camembert", "CamembertForMaskedLM"),
        ("convbert", "ConvBertForMaskedLM"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("deberta", "DebertaForMaskedLM"),
        ("deberta-v2", "DebertaV2ForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForMaskedLM"),
        ("funnel", "FunnelForMaskedLM"),
        ("ibert", "IBertForMaskedLM"),
        ("longformer", "LongformerForMaskedLM"),
        ("luke", "LukeForMaskedLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("tapas", "TapasForMaskedLM"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
    ]
)

MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Object Detection mapping
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
        ("siglip2", "Siglip2Model"),
    ]
)

MODEL_FOR_DEPTH_ESTIMATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for depth estimation mapping
        ("depth_anything", "DepthAnythingForDepthEstimation"),
        ("depth_pro", "DepthProForDepthEstimation"),
        ("dpt", "DPTForDepthEstimation"),
        ("glpn", "GLPNForDepthEstimation"),
        ("prompt_depth_anything", "PromptDepthAnythingForDepthEstimation"),
        ("zoedepth", "ZoeDepthForDepthEstimation"),
    ]
)
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Seq2Seq Causal LM mapping
        ("bart", "BartForConditionalGeneration"),
        ("blenderbot", "BlenderbotForConditionalGeneration"),
        ("blenderbot-small", "BlenderbotSmallForConditionalGeneration"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("led", "LEDForConditionalGeneration"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("mt5", "MT5ForConditionalGeneration"),
        ("pegasus", "PegasusForConditionalGeneration"),
        ("pegasus_x", "PegasusXForConditionalGeneration"),
        ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForTextToText"),
        ("squeezebert", "SqueezeBertForSequenceClassification"),
        ("t5", "T5ForConditionalGeneration"),
        ("umt5", "UMT5ForConditionalGeneration"),
    ]
)

MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("moonshine", "MoonshineForConditionalGeneration"),
        ("pop2piano", "Pop2PianoForConditionalGeneration"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForSpeechToText"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderModel"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("speecht5", "SpeechT5ForSpeechToText"),
        ("whisper", "WhisperForConditionalGeneration"),
    ]
)

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Sequence Classification mapping
        ("albert", "AlbertForSequenceClassification"),
        ("bart", "BartForSequenceClassification"),
        ("bloom", "BloomForSequenceClassification"),
        ("camembert", "CamembertForSequenceClassification"),
        ("electra", "ElectraForSequenceClassification"),
        ("opt", "OPTForSequenceClassification"),
        ("bert", "BertForSequenceClassification"),
        ("ctrl", "CTRLForSequenceClassification"),
        ("mvp", "MvpForSequenceClassification"),
        ("roberta", "RobertaForSequenceClassification"),
        ("deberta", "DebertaForSequenceClassification"),
        ("deberta-v2", "DebertaV2ForSequenceClassification"),
        ("diffllama", "DiffLlamaForSequenceClassification"),
        ("distilbert", "DistilBertForSequenceClassification"),
        ("falcon", "FalconForSequenceClassification"),
        ("funnel", "FunnelForSequenceClassification"),
        ("gemma", "GemmaForSequenceClassification"),
        ("gemma2", "Gemma2ForSequenceClassification"),
        ("glm", "GlmForSequenceClassification"),
        ("gpt_neo", "GPTNeoForSequenceClassification"),
        ("jetmoe", "JetMoeForSequenceClassification"),
        ("helium", "HeliumForSequenceClassification"),
        ("hubert", "HubertForSequenceClassification"),
        ("ibert", "IBertForSequenceClassification"),
        ("led", "LEDForSequenceClassification"),
        ("luke", "LukeForSequenceClassification"),
        ("starcoder2", "Starcoder2ForSequenceClassification"),
        ("canine", "CanineForSequenceClassification"),
        ("lilt", "LiltForSequenceClassification"),
        ("llama", "LlamaForSequenceClassification"),
        ("longformer", "LongformerForSequenceClassification"),
        ("opt", "OPTForSequenceClassification"),
        ("persimmon", "PersimmonForSequenceClassification"),
        ("mbart", "MBartForSequenceClassification"),
        ("mobilebert", "MobileBertForSequenceClassification"),
        ("mpnet", "MPNetForSequenceClassification"),
        ("convbert", "ConvBertForSequenceClassification"),
        ("mt5", "MT5ForSequenceClassification"),
        ("megatron-bert", "MegatronBertForSequenceClassification"),
        ("mistral", "MistralForSequenceClassification"),
        ("mixtral", "MixtralForSequenceClassification"),
        ("mpt", "MptForSequenceClassification"),
        ("nystromformer", "NystromformerForSequenceClassification"),
        ("phi", "PhiForSequenceClassification"),
        ("phi3", "Phi3ForSequenceClassification"),
        ("qwen2", "Qwen2ForSequenceClassification"),
        ("rembert", "RemBertForSequenceClassification"),
        ("stablelm", "StableLmForSequenceClassification"),
        ("t5", "T5ForSequenceClassification"),
        ("gpt_bigcode", "GPTBigCodeForSequenceClassification"),
        ("gptj", "GPTJForSequenceClassification"),
        ("tapas", "TapasForSequenceClassification"),
        ("umt5", "UMT5ForSequenceClassification"),
        ("gpt_neox", "SequenceClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForSequenceClassification"),
        ("zamba", "ZambaForSequenceClassification"),
        ("zamba2", "Zamba2ForSequenceClassification"),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "AlbertForQuestionAnswering"),
        ("bart", "BartForQuestionAnswering"),
        ("mbart", "MBartForQuestionAnswering"),
        ("bloom", "BloomForQuestionAnswering"),
        ("opt", "OPTForQuestionAnswering"),
        ("bert", "BertForQuestionAnswering"),
        ("camembert", "CamembertForQuestionAnswering"),
        ("electra", "ElectraForQuestionAnswering"),
        ("mvp", "MvpForQuestionAnswering"),
        ("roberta", "RobertaForQuestionAnswering"),
        ("deberta", "DebertaForQuestionAnswering"),
        ("deberta-v2", "DebertaV2ForQuestionAnswering"),
        ("diffllama", "DiffLlamaForQuestionAnswering"),
        ("distilbert", "DistilBertForQuestionAnswering"),
        ("falcon", "FalconForQuestionAnswering"),
        ("funnel", "FunnelForQuestionAnswering"),
        ("gpt_neo", "GPTNeoForQuestionAnswering"),
        ("ibert", "IBertForQuestionAnswering"),
        ("led", "LEDForQuestionAnswering"),
        ("lilt", "LiltForQuestionAnswering"),
        ("luke", "LukeForQuestionAnswering"),
        ("convbert", "ConvBertForQuestionAnswering"),
        ("llama", "LlamaForQuestionAnswering"),
        ("longformer", "LongformerForQuestionAnswering"),
        ("mistral", "MistralForQuestionAnswering"),
        ("mobilebert", "MobileBertForQuestionAnswering"),
        ("mpnet", "MPNetForQuestionAnswering"),
        ("megatron-bert", "MegatronBertForQuestionAnswering"),
        ("mistral", "MistralForQuestionAnswering"),
        ("nystromformer", "NystromformerForQuestionAnswering"),
        ("opt", "OPTForQuestionAnswering"),
        ("qwen2", "Qwen2ForQuestionAnswering"),
        ("rembert", "RemBertForQuestionAnswering"),
        ("squeezebert", "SqueezeBertForQuestionAnswering"),
        ("t5", "T5ForQuestionAnswering"),
        ("mixtral", "MixtralForQuestionAnswering"),
        ("mpt", "MptForQuestionAnswering"),
        ("canine", "CanineForQuestionAnswering"),
        ("umt5", "UMT5ForQuestionAnswering"),
        ("xlm-roberta", "XLMRobertaForQuestionAnswering"),
        ("gpt_neox", "GPTNeoXForQuestionAnswering"),
        ("xlm-roberta-xl", "XLMRobertaXLForQuestionAnswering"),
        ("gptj", "GPTJForQuestionAnswering"),
    ]
)

MODEL_FOR_TABLE_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Table Question Answering mapping
        ("tapas", "TapasForQuestionAnswering"),
    ]
)

MODEL_FOR_VISUAL_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("blip", "BlipForQuestionAnswering"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("vilt", "ViltForQuestionAnswering"),
    ]
)

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict()

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "AlbertForTokenClassification"),
        ("bloom", "BloomForTokenClassification"),
        ("bert", "BertForTokenClassification"),
        ("camembert", "CamembertForTokenClassification"),
        ("canine", "CanineForTokenClassification"),
        ("convbert", "ConvBertForTokenClassification"),
        ("deberta", "DebertaForTokenClassification"),
        ("deberta-v2", "DebertaV2ForTokenClassification"),
        ("diffllama", "DiffLlamaForTokenClassification"),
        ("distilbert", "DistilBertForTokenClassification"),
        ("starcoder2", "Starcoder2ForTokenClassification"),
        ("funnel", "FunnelForTokenClassification"),
        ("falcon", "FalconForTokenClassification"),
        ("gpt_neo", "GPTNeoForTokenClassification"),
        ("electra", "ElectraForTokenClassification"),
        ("glm", "GlmForTokenClassification"),
        ("helium", "HeliumForTokenClassification"),
        ("ibert", "IBertForTokenClassification"),
        ("lilt", "LiltForTokenClassification"),
        ("longformer", "LongformerForTokenClassification"),
        ("luke", "LukeForTokenClassification"),
        ("mistral", "MistralForTokenClassification"),
        ("mobilebert", "MobileBertForTokenClassification"),
        ("mpnet", "MPNetForTokenClassification"),
        ("mt5", "MT5ForTokenClassification"),
        ("persimmon", "PersimmonForTokenClassification"),
        ("megatron-bert", "MegatronBertForTokenClassification"),
        ("mixtral", "MixtralForTokenClassification"),
        ("mpt", "MptForTokenClassification"),
        ("nystromformer", "NystromformerForTokenClassification"),
        ("phi", "PhiForTokenClassification"),
        ("phi3", "Phi3ForTokenClassification"),
        ("qwen2", "Qwen2ForTokenClassification"),
        ("roberta", "RobertaForTokenClassification"),
        ("rembert", "RemBertForTokenClassification"),
        ("squeezebert", "SqueezeBertForTokenClassification"),
        ("convbert", "ConvBertForTokenClassification"),
        ("canine", "CanineForTokenClassification"),
        ("stablelm", "StableLmForTokenClassification"),
        ("t5", "T5ForTokenClassification"),
        ("umt5", "UMT5ForTokenClassification"),
        ("xlm-roberta", "XLMRobertaForTokenClassification"),
        ("gpt_neox", "GPTNeoXForTokenClassification"),
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
        ("electra", "ElectraForMultipleChoice"),
        ("deberta-v2", "DebertaV2ForMultipleChoice"),
        ("distilbert", "DistilBertForMultipleChoice"),
        ("funnel", "FunnelForMultipleChoice"),
        ("ibert", "IBertForMultipleChoice"),
        ("longformer", "LongformerForMultipleChoice"),
        ("luke", "LukeForMultipleChoice"),
        ("megatron-bert", "MegatronBertForMultipleChoice"),
        ("mobilebert", "MobileBertForMultipleChoice"),
        ("mpnet", "MPNetForMultipleChoice"),
        ("nystromformer", "NystromformerForMultipleChoice"),
        ("rembert", "RemBertForMultipleChoice"),
        ("roberta", "RobertaForMultipleChoice"),
        ("squeezebert", "SqueezeBertForMultipleChoice"),
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
        ("unispeech", "UniSpeechForSequenceClassification"),
        ("unispeech-sat", "UniSpeechSatForSequenceClassification"),
        ("sew", "SEWForSequenceClassification"),
        ("sew-d", "SEWDForSequenceClassification"),
        ("wav2vec2", "Wav2Vec2ForSequenceClassification"),
        ("whisper", "WhisperForAudioClassification"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        ("unispeech", "UniSpeechForCTC"),
        ("unispeech-sat", "UniSpeechSatForCTC"),
        ("sew", "SEWForCTC"),
        ("sew-d", "SEWDForCTC"),
        ("wav2vec2", "Wav2Vec2ForCTC"),
    ]
)

MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("unispeech-sat", "UniSpeechSatForAudioFrameClassification"),
        ("wav2vec2", "Wav2Vec2ForAudioFrameClassification"),
    ]
)

MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict(
    [
        ("unispeech-sat", "UniSpeechSatForXVector"),
        ("wav2vec2", "Wav2Vec2ForXVector"),
    ]
)

MODEL_FOR_TEXT_TO_SPECTROGRAM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Text-To-Spectrogram mapping
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),
        ("speecht5", "SpeechT5ForTextToSpeech"),
    ]
)

MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Text-To-Waveform mapping
        ("fastspeech2_conformer", "FastSpeech2ConformerWithHifiGan"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForTextToSpeech"),
    ]
)

MODEL_FOR_ZERO_SHOT_IMAGE_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Zero Shot Image Classification mapping
        ("align", "AlignModel"),
        ("altclip", "AltCLIPModel"),
        ("blip", "BlipModel"),
        ("blip-2", "Blip2ForImageTextRetrieval"),
        ("chinese_clip", "ChineseCLIPModel"),
        ("clipseg", "CLIPSegModel"),
        ("siglip", "SiglipModel"),
    ]
)

MODEL_FOR_BACKBONE_MAPPING_NAMES = OrderedDict(
    [
        ("convnext", "ConvNextBackbone"),
        ("convnextv2", "ConvNextV2Backbone"),
        ("dinov2", "Dinov2Backbone"),
        ("hiera", "HieraBackbone"),
        ("resnet", "ResNetBackbone"),
        ("swin", "SwinBackbone"),
        ("swinv2", "Swinv2Backbone"),
        ("vitdet", "VitDetBackbone"),
        ("vitpose_backbone", "VitPoseBackbone"),
    ]
)

MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "SamModel"),
    ]
)


MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES = OrderedDict()


MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        ("albert", "AlbertModel"),
        ("bert", "BertModel"),
        ("electra", "ElectraModel"),
        ("roberta", "RobertaModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("distilbert", "DistilBertModel"),
        ("emu3", "Emu3TextModel"),
        ("ibert", "IBertModel"),
        ("longformer", "LongformerModel"),
        ("mllama", "MllamaTextModel"),
        ("mobilebert", "MobileBertModel"),
        ("mt5", "MT5EncoderModel"),
        ("nystromformer", "NystromformerModel"),
        ("rembert", "RemBertModel"),
        ("squeezebert", "SqueezeBertModel"),
        ("t5", "T5EncoderModel"),
        ("umt5", "UMT5EncoderModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
    ]
)

MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        ("swin2sr", "Swin2SRForImageSuperResolution"),
    ]
)


if version.parse(transformers.__version__) >= version.parse("4.51.0"):
    MODEL_MAPPING_NAMES.update({"qwen3": "Qwen3Model"})
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
