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
"""Auto Config class."""
import importlib
import os
import re
import warnings
from collections import OrderedDict
from typing import List, Union

import transformers
from packaging import version
from transformers.configuration_utils import PretrainedConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module, resolve_trust_remote_code
from transformers.utils import CONFIG_NAME, logging

logger = logging.get_logger(__name__)


CONFIG_MAPPING_NAMES = OrderedDict(
    [
        # Add configs here
        ("albert", "AlbertConfig"),
        ("align", "AlignConfig"),
        ("altclip", "AltCLIPConfig"),
        ("aria", "AriaConfig"),
        ("aria_text", "AriaTextConfig"),
        ("audio-spectrogram-transformer", "ASTConfig"),
        ("aya_vision", "AyaVisionConfig"),
        ("bamba", "BambaConfig"),
        ("bart", "BartConfig"),
        ("beit", "BeitConfig"),
        ("bert", "BertConfig"),
        ("bert-generation", "BertGenerationConfig"),
        ("biogpt", "BioGptConfig"),
        ("bit", "BitConfig"),
        ("blenderbot", "BlenderbotConfig"),
        ("blenderbot-small", "BlenderbotSmallConfig"),
        ("blip", "BlipConfig"),
        ("blip-2", "Blip2Config"),
        ("bloom", "BloomConfig"),
        ("bridgetower", "BridgeTowerConfig"),
        ("bros", "BrosConfig"),
        ("camembert", "CamembertConfig"),
        ("canine", "CanineConfig"),
        ("chameleon", "ChameleonConfig"),
        ("chinese_clip", "ChineseCLIPConfig"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionConfig"),
        ("clip", "CLIPConfig"),
        ("clip_vision_model", "CLIPVisionConfig"),
        ("clipseg", "CLIPSegConfig"),
        ("clvp", "ClvpConfig"),
        ("codegen", "CodeGenConfig"),
        ("cohere", "CohereConfig"),
        ("cohere2", "Cohere2Config"),
        ("colpali", "ColPaliConfig"),
        ("convbert", "ConvBertConfig"),
        ("convnext", "ConvNextConfig"),
        ("convnextv2", "ConvNextV2Config"),
        ("ctrl", "CTRLConfig"),
        ("cvt", "CvtConfig"),
        ("dac", "DacConfig"),
        ("data2vec-audio", "Data2VecAudioConfig"),
        ("data2vec-text", "Data2VecTextConfig"),
        ("data2vec-vision", "Data2VecVisionConfig"),
        ("deberta", "DebertaConfig"),
        ("deberta-v2", "DebertaV2Config"),
        ("depth_anything", "DepthAnythingConfig"),
        ("depth_pro", "DepthProConfig"),
        ("detr", "DetrConfig"),
        ("diffllama", "DiffLlamaConfig"),
        ("dinov2", "Dinov2Config"),
        ("dinov2_with_registers", "Dinov2WithRegistersConfig"),
        ("dinov3_vit", "DINOv3ViTConfig"),
        ("deit", "DeiTConfig"),
        ("distilbert", "DistilBertConfig"),
        ("dpr", "DPRConfig"),
        ("dpt", "DPTConfig"),
        ("efficientnet", "EfficientNetConfig"),
        ("electra", "ElectraConfig"),
        ("emu3", "Emu3Config"),
        ("encodec", "EncodecConfig"),
        ("encoder-decoder", "EncoderDecoderConfig"),
        ("esm", "EsmConfig"),
        ("falcon", "FalconConfig"),
        ("falcon_mamba", "FalconMambaConfig"),
        ("fastspeech2_conformer", "FastSpeech2ConformerConfig"),
        ("flava", "FlavaConfig"),
        ("fnet", "FNetConfig"),
        ("focalnet", "FocalNetConfig"),
        ("fsmt", "FSMTConfig"),
        ("fuyu", "FuyuConfig"),
        ("funnel", "FunnelConfig"),
        ("gemma", "GemmaConfig"),
        ("gemma2", "Gemma2Config"),
        ("gemma3", "Gemma3Config"),
        ("gemma3_text", "Gemma3TextConfig"),
        ("git", "GitConfig"),
        ("glm", "GlmConfig"),
        ("glpn", "GLPNConfig"),
        ("got_ocr2", "GotOcr2Config"),
        ("gpt2", "GPT2Config"),
        ("gpt_bigcode", "GPTBigCodeConfig"),
        ("gpt_neo", "GPTNeoConfig"),
        ("gpt_neox", "GPTNeoXConfig"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseConfig"),
        ("gptj", "GPTJConfig"),
        ("granite", "GraniteConfig"),
        ("granitemoe", "GraniteMoeConfig"),
        ("granitemoeshared", "GraniteMoeSharedConfig"),
        ("groupvit", "GroupViTConfig"),
        ("helium", "HeliumConfig"),
        ("hiera", "HieraConfig"),
        ("hubert", "HubertConfig"),
        ("ibert", "IBertConfig"),
        ("idefics", "IdeficsConfig"),
        ("idefics2", "Idefics2Config"),
        ("idefics3", "Idefics3Config"),
        ("idefics3_vision", "Idefics3VisionConfig"),
        ("ijepa", "IJepaConfig"),
        ("imagegpt", "ImageGPTConfig"),
        ("instructblip", "InstructBlipConfig"),
        ("instructblipvideo", "InstructBlipVideoConfig"),
        ("jamba", "JambaConfig"),
        ("jetmoe", "JetMoeConfig"),
        ("kosmos-2", "Kosmos2Config"),
        ("layoutlm", "LayoutLMConfig"),
        ("layoutlmv3", "LayoutLMv3Config"),
        ("led", "LEDConfig"),
        ("levit", "LevitConfig"),
        ("lilt", "LiltConfig"),
        ("llama", "LlamaConfig"),
        ("persimmon", "PersimmonConfig"),
        ("llava", "LlavaConfig"),
        ("llava_next", "LlavaNextConfig"),
        ("llava_next_video", "LlavaNextVideoConfig"),
        ("llava_onevision", "LlavaOnevisionConfig"),
        ("longformer", "LongformerConfig"),
        ("longt5", "LongT5Config"),
        ("luke", "LukeConfig"),
        ("m2m_100", "M2M100Config"),
        ("mamba", "MambaConfig"),
        ("mamba2", "Mamba2Config"),
        ("markuplm", "MarkupLMConfig"),
        ("marian", "MarianConfig"),
        ("mask2former", "Mask2FormerConfig"),
        ("maskformer", "MaskFormerConfig"),
        ("maskformer-swin", "MaskFormerSwinConfig"),
        ("mbart", "MBartConfig"),
        ("megatron-bert", "MegatronBertConfig"),
        ("mimi", "MimiConfig"),
        ("mistral", "MistralConfig"),
        ("mistral3", "Mistral3Config"),
        ("mixtral", "MixtralConfig"),
        ("mgp-str", "MgpstrConfig"),
        ("mllama", "MllamaConfig"),
        ("mobilebert", "MobileBertConfig"),
        ("mobilevit", "MobileViTConfig"),
        ("mobilevitv2", "MobileViTV2Config"),
        ("mobilenet_v1", "MobileNetV1Config"),
        ("mobilenet_v2", "MobileNetV2Config"),
        ("moonshine", "MoonshineConfig"),
        ("moshi", "MoshiConfig"),
        ("mpnet", "MPNetConfig"),
        ("mpt", "MptConfig"),
        ("mra", "MraConfig"),
        ("mt5", "MT5Config"),
        ("musicgen_melody", "MusicgenMelodyConfig"),
        ("musicgen", "MusicgenConfig"),
        ("mvp", "MvpConfig"),
        ("nemotron", "NemotronConfig"),
        ("nllb-moe", "NllbMoeConfig"),
        ("nystromformer", "NystromformerConfig"),
        ("olmoe", "OlmoeConfig"),
        ("olmo", "OlmoConfig"),
        ("olmo2", "Olmo2Config"),
        ("oneformer", "OneFormerConfig"),
        ("opt", "OPTConfig"),
        ("owlv2", "Owlv2Config"),
        ("owlvit", "OwlViTConfig"),
        ("paligemma", "PaliGemmaConfig"),
        ("pegasus", "PegasusConfig"),
        ("pegasus_x", "PegasusXConfig"),
        ("perceiver", "PerceiverConfig"),
        ("phi", "PhiConfig"),
        ("phi3", "Phi3Config"),
        ("phimoe", "PhimoeConfig"),
        ("pix2struct", "Pix2StructConfig"),
        ("pixtral", "PixtralVisionConfig"),
        ("plbart", "PLBartConfig"),
        ("poolformer", "PoolFormerConfig"),
        ("pop2piano", "Pop2PianoConfig"),
        ("prompt_depth_anything", "PromptDepthAnythingConfig"),
        ("prophetnet", "ProphetNetConfig"),
        ("pvt", "PvtConfig"),
        ("pvt_v2", "PvtV2Config"),
        ("qwen2", "Qwen2Config"),
        ("qwen2_5_vl", "Qwen2_5_VLConfig"),
        ("qwen2_audio", "Qwen2AudioConfig"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoderConfig"),
        ("qwen2_vl", "Qwen2VLConfig"),
        ("rag", "RagConfig"),
        ("recurrent_gemma", "RecurrentGemmaConfig"),
        ("regnet", "RegNetConfig"),
        ("resnet", "ResNetConfig"),
        ("rembert", "RemBertConfig"),
        ("roberta", "RobertaConfig"),
        ("roc_bert", "RoCBertConfig"),
        ("roberta-prelayernorm", "RobertaPreLayerNormConfig"),
        ("roformer", "RoFormerConfig"),
        ("rt_detr", "RTDetrConfig"),
        ("rt_detr_resnet", "RTDetrResNetConfig"),
        ("rt_detr_v2", "RTDetrV2Config"),
        ("rwkv", "RwkvConfig"),
        ("sam", "SamConfig"),
        ("seamless_m4t", "SeamlessM4TConfig"),
        ("seamless_m4t_v2", "SeamlessM4Tv2Config"),
        ("segformer", "SegformerConfig"),
        ("seggpt", "SegGptConfig"),
        ("sew", "SEWConfig"),
        ("sew-d", "SEWDConfig"),
        ("shieldgemma2", "ShieldGemma2Config"),
        ("siglip", "SiglipConfig"),
        ("siglip2", "Siglip2Config"),
        ("siglip_vision_model", "SiglipVisionConfig"),
        ("smolvlm", "SmolVLMConfig"),
        ("smolvlm_vision", "SmolVLMVisionConfig"),
        ("speech-encoder-decoder", "SpeechEncoderDecoderConfig"),
        ("speech_to_text", "Speech2TextConfig"),
        ("speecht5", "SpeechT5Config"),
        ("squeezebert", "SqueezeBertConfig"),
        ("stablelm", "StableLmConfig"),
        ("starcoder2", "Starcoder2Config"),
        ("swiftformer", "SwiftFormerConfig"),
        ("swin", "SwinConfig"),
        ("swin2sr", "Swin2SRConfig"),
        ("swinv2", "Swinv2Config"),
        ("t5", "T5Config"),
        ("table-transformer", "TableTransformerConfig"),
        ("tapas", "TapasConfig"),
        ("textnet", "TextNetConfig"),
        ("timesformer", "TimesformerConfig"),
        ("trocr", "TrOCRConfig"),
        ("tvp", "TvpConfig"),
        ("udop", "UdopConfig"),
        ("umt5", "UMT5Config"),
        ("unispeech", "UniSpeechConfig"),
        ("unispeech-sat", "UniSpeechSatConfig"),
        ("univnet", "UnivNetConfig"),
        ("upernet", "UperNetConfig"),
        ("video_llava", "VideoLlavaConfig"),
        ("videomae", "VideoMAEConfig"),
        ("vilt", "ViltConfig"),
        ("vipllava", "VipLlavaConfig"),
        ("vision-encoder-decoder", "VisionEncoderDecoderConfig"),
        ("visual_bert", "VisualBertConfig"),
        ("vit", "ViTConfig"),
        ("vit_mae", "ViTMAEConfig"),
        ("vit_msn", "ViTMSNConfig"),
        ("vitdet", "VitDetConfig"),
        ("vitpose", "VitPoseConfig"),
        ("vitpose_backbone", "VitPoseBackboneConfig"),
        ("vivit", "VivitConfig"),
        ("wav2vec2", "Wav2Vec2Config"),
        ("wav2vec2-bert", "Wav2Vec2BertConfig"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerConfig"),
        ("wavlm", "WavLMConfig"),
        ("whisper", "WhisperConfig"),
        ("xclip", "XCLIPConfig"),
        ("xglm", "XGLMConfig"),
        ("xlm", "XLMConfig"),
        ("xlm-prophetnet", "XLMProphetNetConfig"),
        ("xlm-roberta", "XLMRobertaConfig"),
        ("xlm-roberta-xl", "XLMRobertaXLConfig"),
        ("xlnet", "XLNetConfig"),
        ("xmod", "XmodConfig"),
        ("yolos", "YolosConfig"),
        ("yoso", "YosoConfig"),
        ("zamba", "ZambaConfig"),
        ("zamba2", "Zamba2Config"),
        ("zoedepth", "ZoeDepthConfig"),
    ]
)


MODEL_NAMES_MAPPING = OrderedDict(
    [
        # Add full (and cased) model names here
        ("albert", "ALBERT"),
        ("align", "ALIGN"),
        ("altclip", "AltCLIP"),
        ("aria", "Aria"),
        ("aria_text", "AriaText"),
        ("audio-spectrogram-transformer", "Audio Spectrogram Transformer"),
        ("aya_vision", "AyaVision"),
        ("bamba", "Bamba"),
        ("bart", "BART"),
        ("barthez", "BARThez"),
        ("bartpho", "BARTpho"),
        ("beit", "BEiT"),
        ("bert", "BERT"),
        ("bert-generation", "Bert Generation"),
        ("biogpt", "BioGpt"),
        ("bit", "BiT"),
        ("blenderbot", "Blenderbot"),
        ("blenderbot-small", "BlenderbotSmall"),
        ("blip", "BLIP"),
        ("blip-2", "BLIP-2"),
        ("bloom", "BLOOM"),
        ("bridgetower", "BridgeTower"),
        ("bros", "BROS"),
        ("camembert", "CamemBERT"),
        ("canine", "CANINE"),
        ("chameleon", "Chameleon"),
        ("chinese_clip", "Chinese-CLIP"),
        ("chinese_clip_vision_model", "ChineseCLIPVisionModel"),
        ("clap", "CLAP"),
        ("clip", "CLIP"),
        ("clip_vision_model", "CLIPVisionModel"),
        ("clipseg", "CLIPSeg"),
        ("clvp", "CLVP"),
        ("codegen", "CodeGen"),
        ("cohere", "Cohere"),
        ("cohere2", "Cohere2"),
        ("colpali", "ColPali"),
        ("convbert", "ConvBERT"),
        ("convnext", "ConvNeXT"),
        ("convnextv2", "ConvNeXTV2"),
        ("ctrl", "CTRL"),
        ("cvt", "CvT"),
        ("dac", "Dac"),
        ("data2vec-audio", "Data2VecAudio"),
        ("data2vec-text", "Data2VecText"),
        ("data2vec-vision", "Data2VecVision"),
        ("deberta", "DeBERTa"),
        ("deberta-v2", "DeBERTa-v2"),
        ("deit", "DeiT"),
        ("depth_anything", "Depth Anything"),
        ("depth_pro", "DepthPro"),
        ("detr", "DETR"),
        ("diffllama", "DiffLlama"),
        ("dinov2", "DINOv2"),
        ("dinov2_with_registers", "DINOv2 with Registers"),
        ("dinov3_vit", "DINOv3 ViT"),
        ("distilbert", "DistilBERT"),
        ("dpr", "DPR"),
        ("dpt", "DPT"),
        ("efficientnet", "EfficientNet"),
        ("electra", "ELECTRA"),
        ("emu3", "Emu3"),
        ("encodec", "Encodec"),
        ("encoder-decoder", "Encoder decoder"),
        ("esm", "ESM"),
        ("falcon", "Falcon"),
        ("falcon_mamba", "FalconMamba"),
        ("fastspeech2_conformer", "FastSpeech2Conformer"),
        ("flava", "FLAVA"),
        ("fnet", "FNet"),
        ("focalnet", "FocalNet"),
        ("fsmt", "FairSeq Machine-Translation"),
        ("funnel", "Funnel Transformer"),
        ("fuyu", "Fuyu"),
        ("gemma", "Gemma"),
        ("gemma2", "Gemma2"),
        ("gemma3", "Gemma3ForConditionalGeneration"),
        ("gemma3_text", "Gemma3ForCausalLM"),
        ("git", "GIT"),
        ("glm", "GLM"),
        ("glpn", "GLPN"),
        ("got_ocr2", "GOT-OCR2"),
        ("gpt2", "OpenAI GPT-2"),
        ("gpt_bigcode", "GPTBigCode"),
        ("gpt_neo", "GPT Neo"),
        ("gpt_neox", "GPTNeoX"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseModel"),
        ("gptj", "GPTJ"),
        ("granite", "Granite"),
        ("granitemoe", "GraniteMoeMoe"),
        ("granitemoeshared", "GraniteMoeSharedMoe"),
        ("groupvit", "GroupViT"),
        ("helium", "Helium"),
        ("hiera", "Hiera"),
        ("hubert", "Hubert"),
        ("ibert", "I-BERT"),
        ("idefics", "IDEFICS"),
        ("idefics2", "Idefics2"),
        ("idefics3", "Idefics3"),
        ("idefics3_vision", "Idefics3VisionTransformer"),
        ("ijepa", "I-JEPA"),
        ("imagegpt", "ImageGPT"),
        ("instructblip", "InstructBLIP"),
        ("instructblipvideo", "InstructBlipVideo"),
        ("jamba", "Jamba"),
        ("jetmoe", "JetMoe"),
        ("kosmos-2", "KOSMOS-2"),
        ("led", "LED"),
        ("levit", "LeViT"),
        ("lilt", "LiLT"),
        ("layoutlm", "LayoutLM"),
        ("layoutlmv3", "LayoutLMv3"),
        ("llama", "LLaMA"),
        ("llama2", "Llama2"),
        ("llama3", "Llama3"),
        ("llava", "Llava"),
        ("llava_next", "LLaVA-NeXT"),
        ("llava_next_video", "LLaVa-NeXT-Video"),
        ("llava_onevision", "LLaVA-Onevision"),
        ("longformer", "Longformer"),
        ("longt5", "LongT5"),
        ("luke", "LUKE"),
        ("m2m_100", "M2M100"),
        ("mamba", "Mamba"),
        ("mamba2", "mamba2"),
        ("marian", "Marian"),
        ("markuplm", "MarkupLM"),
        ("mask2former", "Mask2Former"),
        ("maskformer", "MaskFormer"),
        ("maskformer-swin", "MaskFormerSwin"),
        ("mbart", "MBart"),
        ("megatron-bert", "Megatron-BERT"),
        ("mgp-str", "MGP-STR"),
        ("mimi", "Mimi"),
        ("mistral", "Mistral"),
        ("mistral3", "Mistral3"),
        ("mixtral", "Mixtral"),
        ("mllama", "Mllama"),
        ("mobilebert", "MobileBERT"),
        ("mobilevit", "MobileViT"),
        ("mobilevitv2", "MobileViTV2"),
        ("mobilenet_v1", "MobileNetV1"),
        ("mobilenet_v2", "MobileNetV2"),
        ("moonshine", "Moonshine"),
        ("moshi", "Moshi"),
        ("mpnet", "MPNet"),
        ("mpt", "MPT"),
        ("mra", "MRA"),
        ("mt5", "MT5"),
        ("musicgen", "MusicGen"),
        ("mvp", "MVP"),
        ("musicgen_melody", "MusicGen Melody"),
        ("nemotron", "Nemotron"),
        ("nllb-moe", "NLLB-MOE"),
        ("nystromformer", "Nyströmformer"),
        ("olmo", "OLMo"),
        ("olmo2", "OLMo2"),
        ("olmoe", "OLMoE"),
        ("oneformer", "OneFormer"),
        ("opt", "OPT"),
        ("owlv2", "OWLv2"),
        ("owlvit", "OWL-ViT"),
        ("paligemma", "PaliGemma"),
        ("perceiver", "Perceiver"),
        ("persimmon", "Persimmon"),
        ("phi", "Phi"),
        ("phi3", "Phi3"),
        ("phimoe", "Phimoe"),
        ("pegasus", "Pegasus"),
        ("pegasus_x", "PEGASUS-X"),
        ("pix2struct", "Pix2Struct"),
        ("pixtral", "Pixtral"),
        ("plbart", "PLBart"),
        ("poolformer", "PoolFormer"),
        ("pop2piano", "Pop2Piano"),
        ("prompt_depth_anything", "PromptDepthAnything"),
        ("prophetnet", "ProphetNet"),
        ("pvt", "PVT"),
        ("pvt_v2", "PVTv2"),
        ("qwen2", "Qwen2"),
        ("qwen2_5_vl", "Qwen2_5_VL"),
        ("qwen2_audio", "Qwen2Audio"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("qwen2_vl", "Qwen2VL"),
        ("rag", "RAG"),
        ("recurrent_gemma", "RecurrentGemma"),
        ("regnet", "RegNet"),
        ("rembert", "RemBERT"),
        ("roformer", "RoFormer"),
        ("resnet", "ResNet"),
        ("roberta", "RoBERTa"),
        ("roc_bert", "RoCBert"),
        ("roberta-prelayernorm", "RoBERTa-PreLayerNorm"),
        ("rt_detr", "RT-DETR"),
        ("rt_detr_resnet", "RT-DETR-ResNet"),
        ("rt_detr_v2", "RT-DETRv2"),
        ("rwkv", "RWKV"),
        ("sam", "SAM"),
        ("seamless_m4t", "SeamlessM4T"),
        ("seamless_m4t_v2", "SeamlessM4Tv2"),
        ("segformer", "SegFormer"),
        ("seggpt", "SegGPT"),
        ("sew", "SEW"),
        ("sew-d", "SEW_D"),
        ("shieldgemma2", "Shieldgemma2"),
        ("siglip", "SigLIP"),
        ("siglip2", "SigLIP2"),
        ("siglip2_vision_model", "Siglip2VisionModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("smolvlm", "SmolVLM"),
        ("smolvlm_vision", "SmolVLMVisionTransformer"),
        ("speech-encoder-decoder", "Speech Encoder decoder"),
        ("speech_to_text", "Speech2Text"),
        ("speecht5", "SpeechT5"),
        ("squeezebert", "SqueezeBERT"),
        ("stablelm", "StableLm"),
        ("starcoder2", "Starcoder2"),
        ("swiftformer", "SwiftFormer"),
        ("swin", "Swin Transformer"),
        ("swinv2", "Swin Transformer V2"),
        ("swin2sr", "Swin2SR"),
        ("t5", "T5"),
        ("t5v1.1", "T5v1.1"),
        ("table-transformer", "Table Transformer"),
        ("tapas", "TAPAS"),
        ("textnet", "TextNet"),
        ("timesformer", "TimeSformer"),
        ("trocr", "TrOCR"),
        ("tvp", "TVP"),
        ("udop", "UDOP"),
        ("umt5", "UMT5"),
        ("unispeech", "UniSpeech"),
        ("unispeech-sat", "UniSpeechSat"),
        ("univnet", "UnivNet"),
        ("upernet", "UPerNet"),
        ("video_llava", "VideoLlava"),
        ("videomae", "VideoMAE"),
        ("vilt", "ViLT"),
        ("vipllava", "VipLlava"),
        ("vision-encoder-decoder", "Vision Encoder decoder"),
        ("visual_bert", "VisualBERT"),
        ("vit", "ViT"),
        ("vit_mae", "ViTMAE"),
        ("vit_msn", "ViTMSN"),
        ("vitdet", "VitDet"),
        ("vitpose", "ViTPose"),
        ("vitpose_backbone", "ViTPoseBackbone"),
        ("vivit", "ViViT"),
        ("wav2vec2", "Wav2Vec2"),
        ("wav2vec2-bert", "Wav2Vec2-BERT"),
        ("wav2vec2-conformer", "Wav2Vec2-Conformer"),
        ("wavlm", "WavLM"),
        ("whisper", "Whisper"),
        ("xclip", "X-CLIP"),
        ("xglm", "XGLM"),
        ("xlm", "XLM"),
        ("xlm-prophetnet", "XLM-ProphetNet"),
        ("xlm-roberta", "XLM-RoBERTa"),
        ("xlm-roberta-xl", "XLM-RoBERTa-XL"),
        ("xlnet", "XLNet"),
        ("xmod", "X-MOD"),
        ("yolos", "YOLOS"),
        ("yoso", "YOSO"),
        ("zamba", "Zamba"),
        ("zamba2", "Zamba2"),
        ("zoedepth", "ZoeDepth"),
    ]
)

# This is tied to the processing `-` -> `_` in `model_type_to_module_name`. For example, instead of putting
# `transfo-xl` (as in `CONFIG_MAPPING_NAMES`), we should use `transfo_xl`.
DEPRECATED_MODELS = [
    "bort",
    "deta",
    "efficientformer",
    "ernie_m",
    "gptsan_japanese",
    "graphormer",
    "jukebox",
    "mctct",
    "mega",
    "mmbt",
    "nat",
    "nezha",
    "open_llama",
    "qdqbert",
    "realm",
    "retribert",
    "speech_to_text_2",
    "tapex",
    "trajectory_transformer",
    "transfo_xl",
    "tvlt",
    "van",
    "vit_hybrid",
    "xlm_prophetnet",
]

SPECIAL_MODEL_TYPE_TO_MODULE_NAME = OrderedDict(
    [
        ("aria_text", "aria"),
        ("chinese_clip_vision_model", "chinese_clip"),
        ("clip_vision_model", "clip"),
        ("clip_text_model", "clip"),
        ("data2vec-audio", "data2vec"),
        ("data2vec-text", "data2vec"),
        ("data2vec-vision", "data2vec"),
        ("donut-swin", "donut"),
        ("gemma3_text", "gemma3"),
        ("idefics3_vision", "idefics3"),
        ("kosmos-2", "kosmos2"),
        ("maskformer-swin", "maskformer"),
        ("openai-gpt", "openai"),
        ("qwen2_audio_encoder", "qwen2_audio"),
        ("rt_detr_resnet", "rt_detr"),
        ("siglip_vision_model", "siglip"),
        ("smolvlm_vision", "smolvlm"),
        ("xclip", "x_clip"),
    ]
)

if version.parse(transformers.__version__) >= version.parse("4.51.0"):
    CONFIG_MAPPING_NAMES.update({"qwen3": "Qwen3Config"})
    MODEL_NAMES_MAPPING.update({"qwen3": "Qwen3Model"})

if version.parse(transformers.__version__) >= version.parse("4.51.3"):
    CONFIG_MAPPING_NAMES.update({"glm4": "Glm4Config"})
    MODEL_NAMES_MAPPING.update({"glm4": "glm4"})

if version.parse(transformers.__version__) >= version.parse("4.53.0"):
    CONFIG_MAPPING_NAMES.update({"minimax": "MiniMaxConfig", "vjepa2": "VJEPA2Model"})
    MODEL_NAMES_MAPPING.update({"minimax": "MiniMax", "vjepa2": "VJEPA2Model"})

if version.parse(transformers.__version__) >= version.parse("4.57.0"):
    CONFIG_MAPPING_NAMES.update(
        {
            ("qwen3_vl", "Qwen3VLConfig"),
            ("qwen3_vl_moe", "Qwen3VLMoeConfig"),
            ("qwen3_vl_moe_text", "Qwen3VLMoeTextConfig"),
            ("qwen3_vl_text", "Qwen3VLTextConfig"),
        }
    )
    MODEL_NAMES_MAPPING.update(
        {
            ("qwen3_vl", "Qwen3VL"),
            ("qwen3_vl_moe", "Qwen3VLMoe"),
            ("qwen3_vl_moe_text", "Qwen3VLMoe"),
            ("qwen3_vl_text", "Qwen3VL"),
        }
    )


def model_type_to_module_name(key):
    """Converts a config key to the corresponding module."""
    # Special treatment
    if key in SPECIAL_MODEL_TYPE_TO_MODULE_NAME:
        key = SPECIAL_MODEL_TYPE_TO_MODULE_NAME[key]

        if key in DEPRECATED_MODELS:
            key = f"deprecated.{key}"
        return key

    key = key.replace("-", "_")
    if key in DEPRECATED_MODELS:
        key = f"deprecated.{key}"

    return key


def config_class_to_model_type(config):
    """Converts a config class name to the corresponding model type"""
    for key, cls in CONFIG_MAPPING_NAMES.items():
        if cls == config:
            return key
    # if key not found check in extra content
    for key, cls in CONFIG_MAPPING._extra_content.items():
        if cls.__name__ == config:
            return key
    return None


class _LazyConfigMapping(OrderedDict):
    """
    A dictionary that lazily load its values when they are requested.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._extra_content = {}
        self._modules = {}

    def __getitem__(self, key):
        if key in self._extra_content:
            return self._extra_content[key]
        if key not in self._mapping:
            raise KeyError(key)
        value = self._mapping[key]
        module_name = model_type_to_module_name(key)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "transformers.models")
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        transformers_module = importlib.import_module("transformers")
        return getattr(transformers_module, value)

    def keys(self):
        return list(self._mapping.keys()) + list(self._extra_content.keys())

    def values(self):
        return [self[k] for k in self._mapping.keys()] + list(self._extra_content.values())

    def items(self):
        return [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item):
        return item in self._mapping or item in self._extra_content

    def register(self, key, value, exist_ok=False):
        """
        Register a new configuration in this mapping.
        """
        if key in self._mapping.keys() and not exist_ok:
            raise ValueError(f"'{key}' is already used by a Transformers config, pick another name.")
        self._extra_content[key] = value


CONFIG_MAPPING = _LazyConfigMapping(CONFIG_MAPPING_NAMES)


class _LazyLoadAllMappings(OrderedDict):
    """
    A mapping that will load all pairs of key values at the first access (either by indexing, requestions keys, values,
    etc.)

    Args:
        mapping: The mapping to load.
    """

    def __init__(self, mapping):
        self._mapping = mapping
        self._initialized = False
        self._data = {}

    def _initialize(self):
        if self._initialized:
            return

        for model_type, map_name in self._mapping.items():
            module_name = model_type_to_module_name(model_type)
            module = importlib.import_module(f".{module_name}", "transformers.models")
            mapping = getattr(module, map_name)
            self._data.update(mapping)

        self._initialized = True

    def __getitem__(self, key):
        self._initialize()
        return self._data[key]

    def keys(self):
        self._initialize()
        return self._data.keys()

    def values(self):
        self._initialize()
        return self._data.values()

    def items(self):
        self._initialize()
        return self._data.keys()

    def __iter__(self):
        self._initialize()
        return iter(self._data)

    def __contains__(self, item):
        self._initialize()
        return item in self._data


def _get_class_name(model_class: Union[str, List[str]]):
    if isinstance(model_class, (list, tuple)):
        return " or ".join([f"[`{c}`]" for c in model_class if c is not None])
    return f"[`{model_class}`]"


def _list_model_options(indent, config_to_class=None, use_model_types=True):
    if config_to_class is None and not use_model_types:
        raise ValueError("Using `use_model_types=False` requires a `config_to_class` dictionary.")
    if use_model_types:
        if config_to_class is None:
            model_type_to_name = {model_type: f"[`{config}`]" for model_type, config in CONFIG_MAPPING_NAMES.items()}
        else:
            model_type_to_name = {
                model_type: _get_class_name(model_class)
                for model_type, model_class in config_to_class.items()
                if model_type in MODEL_NAMES_MAPPING
            }
        lines = [
            f"{indent}- **{model_type}** -- {model_type_to_name[model_type]} ({MODEL_NAMES_MAPPING[model_type]} model)"
            for model_type in sorted(model_type_to_name.keys())
        ]
    else:
        config_to_name = {
            CONFIG_MAPPING_NAMES[config]: _get_class_name(clas)
            for config, clas in config_to_class.items()
            if config in CONFIG_MAPPING_NAMES
        }
        config_to_model_name = {
            config: MODEL_NAMES_MAPPING[model_type] for model_type, config in CONFIG_MAPPING_NAMES.items()
        }
        lines = [
            f"{indent}- [`{config_name}`] configuration class:"
            f" {config_to_name[config_name]} ({config_to_model_name[config_name]} model)"
            for config_name in sorted(config_to_name.keys())
        ]
    return "\n".join(lines)


def replace_list_option_in_docstrings(config_to_class=None, use_model_types=True):
    def docstring_decorator(fn):
        docstrings = fn.__doc__
        if docstrings is None:
            # Example: -OO
            return fn
        lines = docstrings.split("\n")
        i = 0
        while i < len(lines) and re.search(r"^(\s*)List options\s*$", lines[i]) is None:
            i += 1
        if i < len(lines):
            indent = re.search(r"^(\s*)List options\s*$", lines[i]).groups()[0]
            if use_model_types:
                indent = f"{indent}    "
            lines[i] = _list_model_options(indent, config_to_class=config_to_class, use_model_types=use_model_types)
            docstrings = "\n".join(lines)
        else:
            raise ValueError(
                f"The function {fn} should have an empty 'List options' in its docstring as placeholder, current"
                f" docstring is:\n{docstrings}"
            )
        fn.__doc__ = docstrings
        return fn

    return docstring_decorator


class AutoConfig:
    r"""
    This is a generic configuration class that will be instantiated as one of the configuration classes of the library
    when created with the [`~AutoConfig.from_pretrained`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """

    def __init__(self):
        raise EnvironmentError(
            "AutoConfig is designed to be instantiated "
            "using the `AutoConfig.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def for_model(cls, model_type: str, *args, **kwargs):
        if model_type in CONFIG_MAPPING:
            config_class = CONFIG_MAPPING[model_type]
            return config_class(*args, **kwargs)
        raise ValueError(
            f"Unrecognized model identifier: {model_type}. Should contain one of {', '.join(CONFIG_MAPPING.keys())}"
        )

    @classmethod
    @replace_list_option_in_docstrings()
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r"""
        Instantiate one of the configuration classes of the library from a pretrained model configuration.

        The configuration class to instantiate is selected based on the `model_type` property of the config object that
        is loaded, or when it's missing, by falling back to using pattern matching on `pretrained_model_name_or_path`:

        List options

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                Can be either:

                    - A string, the *model id* of a pretrained model configuration hosted inside a model repo on
                      huggingface.co.
                    - A path to a *directory* containing a configuration file saved using the
                      [`~PretrainedConfig.save_pretrained`] method, or the [`~PreTrainedModel.save_pretrained`] method,
                      e.g., `./my_model_directory/`.
                    - A path or url to a saved configuration JSON *file*, e.g.,
                      `./my_model_directory/configuration.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download the model weights and configuration files and override the
                cached versions if they exist.
            resume_download:
                Deprecated and ignored. All downloads are now resumed by default when possible.
                Will be removed in v5 of Transformers.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final configuration object.

                If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where *unused_kwargs* is a
                dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the
                part of `kwargs` which has not been used to update `config` and is otherwise ignored.
            trust_remote_code (`bool`, *optional*, defaults to `False`):
                Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
                should only be set to `True` for repositories you trust and in which you have read the code, as it will
                execute code present on the Hub on your local machine.
            kwargs(additional keyword arguments, *optional*):
                The values in kwargs of any keys which are configuration attributes will be used to override the loaded
                values. Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Examples:

        ```python
        >>> from transformers import AutoConfig

        >>> # Download configuration from huggingface.co and cache.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased")

        >>> # Download configuration from huggingface.co (user-uploaded) and cache.
        >>> config = AutoConfig.from_pretrained("dbmdz/bert-base-german-cased")

        >>> # If configuration file is in a directory (e.g., was saved using *save_pretrained('./test/saved_model/')*).
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/")

        >>> # Load a specific configuration file.
        >>> config = AutoConfig.from_pretrained("./test/bert_saved_model/my_configuration.json")

        >>> # Change some config attributes when loading a pretrained config.
        >>> config = AutoConfig.from_pretrained("google-bert/bert-base-uncased", output_attentions=True, foo=False)
        >>> config.output_attentions
        True

        >>> config, unused_kwargs = AutoConfig.from_pretrained(
        ...     "google-bert/bert-base-uncased", output_attentions=True, foo=False, return_unused_kwargs=True
        ... )
        >>> config.output_attentions
        True

        >>> unused_kwargs
        {'foo': False}
        ```"""
        use_auth_token = kwargs.pop("use_auth_token", None)
        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if kwargs.get("token", None) is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            kwargs["token"] = use_auth_token

        kwargs["_from_auto"] = True
        kwargs["name_or_path"] = pretrained_model_name_or_path
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        code_revision = kwargs.pop("code_revision", None)

        config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
        has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING
        trust_remote_code = resolve_trust_remote_code(
            trust_remote_code, pretrained_model_name_or_path, has_local_code, has_remote_code
        )

        if has_remote_code and trust_remote_code:
            class_ref = config_dict["auto_map"]["AutoConfig"]
            config_class = get_class_from_dynamic_module(
                class_ref, pretrained_model_name_or_path, code_revision=code_revision, **kwargs
            )
            if os.path.isdir(pretrained_model_name_or_path):
                config_class.register_for_auto_class()
            return config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        elif "model_type" in config_dict:
            try:
                config_class = CONFIG_MAPPING[config_dict["model_type"]]
            except KeyError:
                raise ValueError(
                    f"The checkpoint you are trying to load has model type `{config_dict['model_type']}` "
                    "but Transformers does not recognize this architecture. This could be because of an "
                    "issue with the checkpoint, or because your version of Transformers is out of date."
                )
            return config_class.from_dict(config_dict, **unused_kwargs)
        else:
            # Fallback: use pattern matching on the string.
            # We go from longer names to shorter names to catch roberta before bert (for instance)
            for pattern in sorted(CONFIG_MAPPING.keys(), key=len, reverse=True):
                if pattern in str(pretrained_model_name_or_path):
                    return CONFIG_MAPPING[pattern].from_dict(config_dict, **unused_kwargs)

        raise ValueError(
            f"Unrecognized model in {pretrained_model_name_or_path}. "
            f"Should have a `model_type` key in its {CONFIG_NAME}, or contain one of the following strings "
            f"in its name: {', '.join(CONFIG_MAPPING.keys())}"
        )

    @staticmethod
    def register(model_type, config, exist_ok=False):
        """
        Register a new configuration for this class.

        Args:
            model_type (`str`): The model type like "bert" or "gpt".
            config ([`PretrainedConfig`]): The config to register.
        """
        if issubclass(config, PretrainedConfig) and config.model_type != model_type:
            raise ValueError(
                "The config you are passing has a `model_type` attribute that is not consistent with the model type "
                f"you passed (config has {config.model_type} and you passed {model_type}. Fix one of those so they "
                "match!"
            )
        CONFIG_MAPPING.register(model_type, config, exist_ok=exist_ok)
