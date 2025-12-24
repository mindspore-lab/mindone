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
        ("aimv2", "Aimv2Model"),
        ("albert", "AlbertModel"),
        ("align", "AlignModel"),
        ("altclip", "AltCLIPModel"),
        ("apertus", "ApertusModel"),
        ("arcee", "ArceeModel"),
        ("aria", "AriaForConditionalGeneration"),
        ("aria_text", "AriaTextModel"),
        ("audio-spectrogram-transformer", "ASTModel"),
        ("bamba", "BambaModel"),
        ("bart", "BartModel"),
        ("beit", "BeitModel"),
        ("bert", "BertModel"),
        ("bert-generation", "BertGenerationEncoder"),
        ("bert", "BertModel"),
        ("bit", "BitModel"),
        ("blenderbot", "BlenderbotModel"),
        ("blenderbot-small", "BlenderbotSmallModel"),
        ("blip", "BlipModel"),
        ("blip-2", "Blip2Model"),
        ("blip_2_qformer", "Blip2QFormerModel"),
        ("biogpt", "BioGptModel"),
        ("bloom", "BloomModel"),
        ("blt", "BltModel"),
        ("bridgetower", "BridgeTowerModel"),
        ("bros", "BrosModel"),
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
        ("cohere", "CohereModel"),
        ("cohere2", "Cohere2Model"),
        ("cohere2_vision", "Cohere2VisionModel"),
        ("convbert", "ConvBertModel"),
        ("convnext", "ConvNextModel"),
        ("convnextv2", "ConvNextV2Model"),
        ("csm", "CsmForConditionalGeneration"),
        ("ctrl", "CTRLModel"),
        ("cvt", "CvtModel"),
        ("d_fine", "DFineModel"),
        ("dac", "DacModel"),
        ("data2vec-audio", "Data2VecAudioModel"),
        ("data2vec-text", "Data2VecTextModel"),
        ("data2vec-vision", "Data2VecVisionModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("deepseek_v2", "DeepseekV2Model"),
        ("deepseek_vl", "DeepseekVLModel"),
        ("deepseek_vl_hybrid", "DeepseekVLHybridModel"),
        ("deit", "DeiTModel"),
        ("depth_pro", "DepthProModel"),
        ("detr", "DetrModel"),
        ("dia", "DiaModel"),
        ("diffllama", "DiffLlamaModel"),
        ("dinov2", "Dinov2Model"),
        ("dinov2_with_registers", "Dinov2WithRegistersModel"),
        ("dinov3_convnext", "DINOv3ConvNextModel"),
        ("dinov3_vit", "DINOv3ViTModel"),
        ("dots1", "Dots1Model"),
        ("distilbert", "DistilBertModel"),
        ("doge", "DogeModel"),
        ("dpr", "DPRQuestionEncoder"),
        ("dpt", "DPTModel"),
        ("efficientloftr", "EfficientLoFTRModel"),
        ("efficientnet", "EfficientNetModel"),
        ("electra", "ElectraModel"),
        ("encodec", "EncodecModel"),
        ("ernie4_5", "Ernie4_5Model"),
        ("ernie4_5_moe", "Ernie4_5_MoeModel"),
        ("esm", "EsmModel"),
        ("exaone4", "Exaone4Model"),
        ("falcon", "FalconModel"),
        ("falcon_h1", "FalconH1Model"),
        ("falcon_mamba", "FalconMambaModel"),
        ("fastspeech2_conformer", "FastSpeech2ConformerModel"),
        ("flava", "FlavaModel"),
        ("flex_olmo", "FlexOlmoModel"),
        ("florence2", "Florence2Model"),
        ("fnet", "FNetModel"),
        ("focalnet", "FocalNetModel"),
        ("fsmt", "FSMTModel"),
        ("funnel", ("FunnelModel", "FunnelBaseModel")),
        ("gemma", "GemmaModel"),
        ("gemma2", "Gemma2Model"),
        ("gemma3_text", "Gemma3TextModel"),
        ("git", "GitModel"),
        ("glm", "GlmModel"),
        ("glm4_moe", "Glm4MoeModel"),
        ("glpn", "GLPNModel"),
        ("got_ocr2", "GotOcr2ForConditionalGeneration"),
        ("gpt2", "GPT2Model"),
        ("gpt_bigcode", "GPTBigCodeModel"),
        ("gpt_neo", "GPTNeoModel"),
        ("gpt_neox", "GPTNeoXModel"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseModel"),
        ("gptj", "GPTJModel"),
        ("granite", "GraniteModel"),
        ("granitemoe", "GraniteMoeModel"),
        ("granitemoehybrid", "GraniteMoeHybridModel"),
        ("granitemoeshared", "GraniteMoeSharedModel"),
        ("groupvit", "GroupViTModel"),
        ("grounding-dino", "GroundingDinoModel"),
        ("mm-grounding-dino", "MMGroundingDinoModel"),
        ("helium", "HeliumModel"),
        ("hgnet_v2", "HGNetV2Backbone"),
        ("hiera", "HieraModel"),
        ("hubert", "HubertModel"),
        ("hunyuan_v1_dense", "HunYuanDenseV1Model"),
        ("hunyuan_v1_moe", "HunYuanMoEV1Model"),
        ("ibert", "IBertModel"),
        ("idefics", "IdeficsModel"),
        ("idefics2", "Idefics2Model"),
        ("idefics3", "Idefics3Model"),
        ("idefics3_vision", "Idefics3VisionTransformer"),
        ("ijepa", "IJepaModel"),
        ("imagegpt", "ImageGPTModel"),
        ("internvl", "InternVLModel"),
        ("internvl_vision", "InternVLVisionModel"),
        ("janus", "JanusModel"),
        ("jamba", "JambaModel"),
        ("jetmoe", "JetMoeModel"),
        ("kosmos-2", "Kosmos2Model"),
        ("kosmos-2.5", "Kosmos2_5Model"),
        ("kyutai_speech_to_text", "KyutaiSpeechToTextModel"),
        ("layoutlm", "LayoutLMModel"),
        ("layoutlmv3", "LayoutLMv3Model"),
        ("led", "LEDModel"),
        ("levit", "LevitModel"),
        ("lfm2", "Lfm2Model"),
        ("lfm2_vl", "Lfm2VlModel"),
        ("lilt", "LiltModel"),
        ("llama", "LlamaModel"),
        ("llama4", "Llama4ForConditionalGeneration"),
        ("llama4_text", "Llama4TextModel"),
        ("longformer", "LongformerModel"),
        ("longt5", "LongT5Model"),
        ("luke", "LukeModel"),
        ("m2m_100", "M2M100Model"),
        ("marian", "MarianModel"),
        ("mamba", "MambaModel"),
        ("mamba2", "Mamba2Model"),
        ("markuplm", "MarkupLMModel"),
        ("mask2former", "Mask2FormerModel"),
        ("maskformer", "MaskFormerModel"),
        ("maskformer-swin", "MaskFormerSwinModel"),
        ("mbart", "MBartModel"),
        ("megatron-bert", "MegatronBertModel"),
        ("metaclip_2", "MetaClip2Model"),
        ("mgp-str", "MgpstrForSceneTextRecognition"),
        ("mimi", "MimiModel"),
        ("ministral", "MinistralModel"),
        ("mistral", "MistralModel"),
        ("mixtral", "MixtralModel"),
        ("mobilebert", "MobileBertModel"),
        ("mobilenet_v1", "MobileNetV1Model"),
        ("mobilenet_v2", "MobileNetV2Model"),
        ("mobilevit", "MobileViTModel"),
        ("mobilevitv2", "MobileViTV2Model"),
        ("modernbert-decoder", "ModernBertDecoderModel"),
        ("moonshine", "MoonshineModel"),
        ("moshi", "MoshiModel"),
        ("mpnet", "MPNetModel"),
        ("mlcd", "MLCDVisionModel"),
        ("mpt", "MptModel"),
        ("mra", "MraModel"),
        ("mt5", "MT5Model"),
        ("musicgen", "MusicgenModel"),
        ("musicgen_melody", "MusicgenMelodyModel"),
        ("mvp", "MvpModel"),
        ("nemotron", "NemotronModel"),
        ("nllb-moe", "NllbMoeModel"),
        ("nystromformer", "NystromformerModel"),
        ("olmo", "OlmoModel"),
        ("olmo2", "Olmo2Model"),
        ("olmo3", "Olmo3Model"),
        ("olmoe", "OlmoeModel"),
        ("oneformer", "OneFormerModel"),
        ("opt", "OPTModel"),
        ("ovis2", "Ovis2Model"),
        ("owlv2", "Owlv2Model"),
        ("owlvit", "OwlViTModel"),
        ("parakeet_ctc", "ParakeetForCTC"),
        ("parakeet_encoder", "ParakeetEncoder"),
        ("pegasus", "PegasusModel"),
        ("pegasus_x", "PegasusXModel"),
        ("perceiver", "PerceiverModel"),
        ("persimmon", "PersimmonModel"),
        ("phi", "PhiModel"),
        ("phi3", "Phi3Model"),
        ("pixtral", "PixtralVisionModel"),
        ("plbart", "PLBartModel"),
        ("poolformer", "PoolFormerModel"),
        ("prophetnet", "ProphetNetModel"),
        ("pvt", "PvtModel"),
        ("pvt_v2", "PvtV2Model"),
        ("qwen2", "Qwen2Model"),
        ("qwen2_5_vl", "Qwen2_5_VLModel"),
        ("qwen2_audio_encoder", "Qwen2AudioEncoder"),
        ("qwen2_moe", "Qwen2MoeModel"),
        ("qwen2_vl", "Qwen2VLModel"),
        ("qwen2_vl_text", "Qwen2VLTextModel"),
        ("qwen3_next", "Qwen3NextModel"),
        ("recurrent_gemma", "RecurrentGemmaModel"),
        ("regnet", "RegNetModel"),
        ("roberta", "RobertaModel"),
        ("rembert", "RemBertModel"),
        ("resnet", "ResNetModel"),
        ("rt_detr", "RTDetrModel"),
        ("rt_detr_v2", "RTDetrV2Model"),
        ("roberta", "RobertaModel"),
        ("roformer", "RoFormerModel"),
        ("roc_bert", "RoCBertModel"),
        ("roberta-prelayernorm", "RobertaPreLayerNormModel"),
        ("rwkv", "RwkvModel"),
        ("sam", "SamModel"),
        ("sam2", "Sam2Model"),
        ("sam2_hiera_det_model", "Sam2HieraDetModel"),
        ("sam2_vision_model", "Sam2VisionModel"),
        ("sam2_video", "Sam2VideoModel"),
        ("sam_hq", "SamHQModel"),
        ("sam_hq_vision_model", "SamHQVisionModel"),
        ("sam_vision_model", "SamVisionModel"),
        ("seamless_m4t", "SeamlessM4TModel"),
        ("seamless_m4t_v2", "SeamlessM4Tv2Model"),
        ("seed_oss", "SeedOssModel"),
        ("segformer", "SegformerModel"),
        ("seggpt", "SegGptModel"),
        ("sew", "SEWModel"),
        ("sew-d", "SEWDModel"),
        ("siglip", "SiglipModel"),
        ("siglip2", "Siglip2Model"),
        ("siglip2_vision_model", "Siglip2VisionModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("smollm3", "SmolLM3Model"),
        ("smolvlm", "SmolVLMModel"),
        ("smolvlm_vision", "SmolVLMVisionTransformer"),
        ("speech_to_text", "Speech2TextModel"),
        ("speecht5", "SpeechT5Model"),
        ("squeezebert", "SqueezeBertModel"),
        ("stablelm", "StableLmModel"),
        ("starcoder2", "Starcoder2Model"),
        ("swiftformer", "SwiftFormerModel"),
        ("swin2sr", "Swin2SRModel"),
        ("swinv2", "Swinv2Model"),
        ("t5", "T5Model"),
        ("table-transformer", "TableTransformerModel"),
        ("tapas", "TapasModel"),
        ("textnet", "TextNetModel"),
        ("timesfm", "TimesFmModel"),
        ("timesformer", "TimesformerModel"),
        ("tvp", "TvpModel"),
        ("udop", "UdopModel"),
        ("t5gemma", "T5GemmaModel"),
        ("umt5", "UMT5Model"),
        ("unispeech", "UniSpeechModel"),
        ("unispeech-sat", "UniSpeechSatModel"),
        ("univnet", "UnivNetModel"),
        ("vaultgemma", "VaultGemmaModel"),
        ("videomae", "VideoMAEModel"),
        ("vilt", "ViltModel"),
        ("visual_bert", "VisualBertModel"),
        ("vit", "ViTModel"),
        ("vit_mae", "ViTMAEModel"),
        ("vit_msn", "ViTMSNModel"),
        ("vitdet", "VitDetModel"),
        ("vivit", "VivitModel"),
        ("voxtral", "VoxtralForConditionalGeneration"),
        ("voxtral_encoder", "VoxtralEncoder"),
        ("wav2vec2", "Wav2Vec2Model"),
        ("wav2vec2-bert", "Wav2Vec2BertModel"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerModel"),
        ("wavlm", "WavLMModel"),
        ("whisper", "WhisperModel"),
        ("xlstm", "xLSTMModel"),
        ("xglm", "XGLMModel"),
        ("xclip", "XCLIPModel"),
        ("xcodec", "XcodecModel"),
        ("xlm", "XLMModel"),
        ("xlm-prophetnet", "XLMProphetNetModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
        ("xlnet", "XLNetModel"),
        ("xmod", "XmodModel"),
        ("yolos", "YolosModel"),
        ("yoso", "YosoModel"),
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
        ("colqwen2", "ColQwen2ForRetrieval"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForMaskedLM"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("electra", "ElectraForPreTraining"),
        ("exaone4", "Exaone4ForCausalLM"),
        ("falcon_mamba", "FalconMambaForCausalLM"),
        ("flava", "FlavaForPreTraining"),
        ("florence2", "Florence2ForConditionalGeneration"),
        ("fnet", "FNetForPreTraining"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("funnel", "FunnelForPreTraining"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_neox_japanese", "GPTNeoXJapanesePreTrainedModel"),
        ("gemma3", "Gemma3ForConditionalGeneration"),
        ("hiera", "HieraForPreTraining"),
        ("hubert", "HubertForPreTraining"),
        ("ibert", "IBertForMaskedLM"),
        ("idefics", "IdeficsForVisionText2Text"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("internvl", "InternVLForConditionalGeneration"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("luke", "LukeForMaskedLM"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("mbart", "MBartForConditionalGeneration"),
        ("megatron-bert", "MegatronBertForPreTraining"),
        ("mistral3", "Mistral3ForConditionalGeneration"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("mobilebert", "MobileBertForPreTraining"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mpt", "MptForCausalLM"),
        ("mra", "MraForMaskedLM"),
        ("musicgen", "MusicgenForCausalLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
        ("roberta", "RobertaForMaskedLM"),
        ("roc_bert", "RoCBertForPreTraining"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMaskedLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("smollm3", "SmolLM3ForCausalLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("t5", "T5ForConditionalGeneration"),
        ("tapas", "TapasForMaskedLM"),
        ("unispeech", "UniSpeechForPreTraining"),
        ("unispeec-sat", "UniSpeechSatForPreTraining"),
        ("video_llava", "VideoLlavaForConditionalGeneration"),
        ("videomae", "VideoMAEForPreTraining"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        ("visual_bert", "VisualBertForPreTraining"),
        ("vit_mae", "ViTMAEForPreTraining"),
        ("t5gemma", "T5GemmaForConditionalGeneration"),
        ("voxtral", "VoxtralForConditionalGeneration"),
        ("wav2vec2", "Wav2Vec2ForPreTraining"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForPreTraining"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForMaskedLM"),
    ]
)

MODEL_WITH_LM_HEAD_MAPPING_NAMES = OrderedDict(
    [
        # Model with LM heads mapping
        ("albert", "AlbertForMaskedLM"),
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
        ("dia", "DiaForConditionalGeneration"),
        ("distilbert", "DistilBertForMaskedLM"),
        ("esm", "EsmForMaskedLM"),
        ("electra", "ElectraForMaskedLM"),
        ("encoder-decoder", "EncoderDecoderModel"),
        ("exaone4", "Exaone4ForCausalLM"),
        ("falcon_mamba", "FalconMambaForCausalLM"),
        ("fnet", "FNetForMaskedLM"),
        ("fsmt", "FSMTForConditionalGeneration"),
        ("funnel", "FunnelForMaskedLM"),
        ("git", "GitForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt2", "GPT2LMHeadModel"),
        ("ibert", "IBertForMaskedLM"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("led", "LEDForConditionalGeneration"),
        ("longformer", "LongformerForMaskedLM"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("luke", "LukeForMaskedLM"),
        ("marian", "MarianMTModel"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("mbart", "MBartForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("moonshine", "MoonshineForConditionalGeneration"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mpt", "MptForCausalLM"),
        ("mra", "MraForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("plbart", "PLBartForConditionalGeneration"),
        ("pegasus_x", "PegasusXForConditionalGeneration"),
        ("pop2piano", "Pop2PianoForConditionalGeneration"),
        ("rembert", "RemBertForMaskedLM"),
        ("roc_bert", "RoCBertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMaskedLM"),
        ("roformer", "RoFormerForMaskedLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("speech_to_text", "Speech2TextForConditionalGeneration"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("t5", "T5ForConditionalGeneration"),
        ("tapas", "TapasForMaskedLM"),
        ("t5gemma", "T5GemmaForConditionalGeneration"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("whisper", "WhisperForConditionalGeneration"),
        ("xlstm", "xLSTMForCausalLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForMaskedLM"),
        ("yoso", "YosoForMaskedLM"),
    ]
)

MODEL_FOR_CAUSAL_LM_MAPPING_NAMES = OrderedDict(
    [
        # Model for Causal LM mapping
        ("apertus", "ApertusForCausalLM"),
        ("arcee", "ArceeForCausalLM"),
        ("aria_text", "AriaTextForCausalLM"),
        ("bamba", "BambaForCausalLM"),
        ("bart", "BartForCausalLM"),
        ("biogpt", "BioGptForCausalLM"),
        ("camembert", "CamembertForCausalLM"),
        ("bert", "BertLMHeadModel"),
        ("bert-generation", "BertGenerationDecoder"),
        ("blenderbot", "BlenderbotForCausalLM"),
        ("blenderbot-small", "BlenderbotSmallForCausalLM"),
        ("bloom", "BloomForCausalLM"),
        ("blt", "BltForCausalLM"),
        ("bert-generation", "BertGenerationDecoder"),
        ("camembert", "CamembertForCausalLM"),
        ("codegen", "CodeGenForCausalLM"),
        ("cohere", "CohereForCausalLM"),
        ("cohere2", "Cohere2ForCausalLM"),
        ("ctrl", "CTRLLMHeadModel"),
        ("data2vec-text", "Data2VecTextForCausalLM"),
        ("deepseek_v2", "DeepseekV2ForCausalLM"),
        ("diffllama", "DiffLlamaForCausalLM"),
        ("doge", "DogeForCausalLM"),
        ("dots1", "Dots1ForCausalLM"),
        ("emu3", "Emu3ForCausalLM"),
        ("ernie4_5", "Ernie4_5ForCausalLM"),
        ("ernie4_5_moe", "Ernie4_5_MoeForCausalLM"),
        ("exaone4", "Exaone4ForCausalLM"),
        ("falcon", "FalconForCausalLM"),
        ("falcon_h1", "FalconH1ForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("falcon_mamba", "FalconMambaForCausalLM"),
        ("flex_olmo", "FlexOlmoForCausalLM"),
        ("fuyu", "FuyuForCausalLM"),
        ("gemma", "GemmaForCausalLM"),
        ("gemma2", "Gemma2ForCausalLM"),
        ("gemma3", "Gemma3ForCausalLM"),
        ("gemma3_text", "Gemma3ForCausalLM"),
        ("git", "GitForCausalLM"),
        ("glm", "GlmForCausalLM"),
        ("glm4_moe", "Glm4MoeForCausalLM"),
        ("got_ocr2", "GotOcr2ForConditionalGeneration"),
        ("gpt2", "GPT2LMHeadModel"),
        ("gpt_bigcode", "GPTBigCodeForCausalLM"),
        ("gpt_neo", "GPTNeoForCausalLM"),
        ("gpt_neox", "GPTNeoXForCausalLM"),
        ("gpt_neox_japanese", "GPTNeoXJapaneseForCausalLM"),
        ("gptj", "GPTJForCausalLM"),
        ("granite", "GraniteForCausalLM"),
        ("granitemoe", "GraniteMoeForCausalLM"),
        ("granitemoehybrid", "GraniteMoeHybridForCausalLM"),
        ("granitemoeshared", "GraniteMoeSharedForCausalLM"),
        ("helium", "HeliumForCausalLM"),
        ("hunyuan_v1_dense", "HunYuanDenseV1ForCausalLM"),
        ("hunyuan_v1_moe", "HunYuanMoEV1ForCausalLM"),
        ("jamba", "JambaForCausalLM"),
        ("janus", "JanusForConditionalGeneration"),
        ("jetmoe", "JetMoeForCausalLM"),
        ("layoutlm", "LayoutLMForMaskedLM"),
        ("lfm2", "Lfm2ForCausalLM"),
        ("llama", "LlamaForCausalLM"),
        ("llama4", "Llama4ForCausalLM"),
        ("llama4_text", "Llama4ForCausalLM"),
        ("mamba", "MambaForCausalLM"),
        ("mamba2", "Mamba2ForCausalLM"),
        ("marian", "MarianForCausalLM"),
        ("mbart", "MBartForCausalLM"),
        ("megatron-bert", "MegatronBertForCausalLM"),
        ("ministral", "MinistralForCausalLM"),
        ("mistral", "MistralForCausalLM"),
        ("mixtral", "MixtralForCausalLM"),
        ("mllama", "MllamaForCausalLM"),
        ("modernbert-decoder", "ModernBertDecoderForCausalLM"),
        ("moshi", "MoshiForCausalLM"),
        ("mpt", "MptForCausalLM"),
        ("musicgen_melody", "MusicgenMelodyForCausalLM"),
        ("mvp", "MvpForCausalLM"),
        ("nemotron", "NemotronForCausalLM"),
        ("olmo", "OlmoForCausalLM"),
        ("olmo2", "Olmo2ForCausalLM"),
        ("olmo3", "Olmo3ForCausalLM"),
        ("olmoe", "OlmoeForCausalLM"),
        ("opt", "OPTForCausalLM"),
        ("persimmon", "PersimmonForCausalLM"),
        ("phi", "PhiForCausalLM"),
        ("phi3", "Phi3ForCausalLM"),
        ("pegasus", "PegasusForCausalLM"),
        ("plbart", "PLBartForCausalLM"),
        ("prophetnet", "ProphetNetForCausalLM"),
        ("qwen2", "Qwen2ForCausalLM"),
        ("qwen2_moe", "Qwen2MoeForCausalLM"),
        ("qwen3_next", "Qwen3NextForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForCausalLM"),
        ("recurrent_gemma", "RecurrentGemmaForCausalLM"),
        ("rembert", "RemBertForCausalLM"),
        ("roformer", "RoFormerForCausalLM"),
        ("roc_bert", "RoCBertForCausalLM"),
        ("roberta", "RobertaForCausalLM"),
        ("rwkv", "RwkvForCausalLM"),
        ("seed_oss", "SeedOssForCausalLM"),
        ("stablelm", "StableLmForCausalLM"),
        ("starcoder2", "Starcoder2ForCausalLM"),
        ("trocr", "TrOCRForCausalLM"),
        ("vaultgemma", "VaultGemmaForCausalLM"),
        ("whisper", "WhisperForCausalLM"),
        ("xlstm", "xLSTMForCausalLM"),
        ("xglm", "XGLMForCausalLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-prophetnet", "XLMProphetNetForCausalLM"),
        ("xlm-roberta", "XLMRobertaForCausalLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForCausalLM"),
        ("xlnet", "XLNetLMHeadModel"),
        ("xmod", "XmodForCausalLM"),
        ("zamba", "ZambaForCausalLM"),
    ]
)

MODEL_FOR_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Image mapping
        ("beit", "BeitModel"),
        ("bit", "BitModel"),
        ("cohere2_vision", "Cohere2VisionModel"),
        ("convnext", "ConvNextModel"),
        ("convnextv2", "ConvNextV2Model"),
        ("data2vec-vision", "Data2VecVisionModel"),
        ("deit", "DeiTModel"),
        ("detr", "DetrModel"),
        ("depth_pro", "DepthProModel"),
        ("dinov2", "Dinov2Model"),
        ("dinov2_with_registers", "Dinov2WithRegistersModel"),
        ("dinov3_convnext", "DINOv3ConvNextModel"),
        ("dinov3_vit", "DINOv3ViTModel"),
        ("dpt", "DPTModel"),
        ("efficientnet", "EfficientNetModel"),
        ("focalnet", "FocalNetModel"),
        ("glpn", "GLPNModel"),
        ("hiera", "HieraModel"),
        ("hubert", "HubertModel"),
        ("ijepa", "IJepaModel"),
        ("imagegpt", "ImageGPTModel"),
        ("levit", "LevitModel"),
        ("llama4", "Llama4VisionModel"),
        ("mobilevit", "MobileViTModel"),
        ("mobilevitv2", "MobileViTV2Model"),
        ("mllama", "MllamaVisionModel"),
        ("poolformer", "PoolFormerModel"),
        ("pvt", "PvtModel"),
        ("regnet", "RegNetModel"),
        ("resnet", "ResNetModel"),
        ("mlcd", "MLCDVisionModel"),
        ("segformer", "SegformerModel"),
        ("seggpt", "SegGptModel"),
        ("siglip_vision_model", "SiglipVisionModel"),
        ("swiftformer", "SwiftFormerModel"),
        ("swin2sr", "Swin2SRModel"),
        ("swinv2", "Swinv2Model"),
        ("table-transformer", "TableTransformerModel"),
        ("timesformer", "TimesformerModel"),
        ("videomae", "VideoMAEModel"),
        ("vit", "ViTModel"),
        ("vit_mae", "ViTMAEModel"),
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
        ("focalnet", "FocalNetForMaskedImageModeling"),
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
        ("beit", "BeitForImageClassification"),
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
        ("dinov2_with_registers", "Dinov2WithRegistersForImageClassification"),
        ("efficientnet", "EfficientNetForImageClassification"),
        ("focalnet", "FocalNetForImageClassification"),
        ("hgnet_v2", "HGNetV2ForImageClassification"),
        ("hiera", "HieraForImageClassification"),
        ("ijepa", "IJepaForImageClassification"),
        ("imagegpt", "ImageGPTForImageClassification"),
        (
            "levit",
            ("LevitForImageClassification", "LevitForImageClassificationWithTeacher"),
        ),
        ("metaclip_2", "MetaClip2ForImageClassification"),
        ("mobilenet_v1", "MobileNetV1ForImageClassification"),
        ("mobilenet_v2", "MobileNetV2ForImageClassification"),
        ("mobilevit", "MobileViTForImageClassification"),
        ("mobilevitv2", "MobileViTV2ForImageClassification"),
        (
            "perceiver",
            (
                "PerceiverForImageClassificationLearned",
                "PerceiverForImageClassificationFourier",
                "PerceiverForImageClassificationConvProcessing",
            ),
        ),
        ("poolformer", "PoolFormerForImageClassification"),
        ("pvt", "PvtForImageClassification"),
        ("pvt_v2", "PvtV2ForImageClassification"),
        ("regnet", "RegNetForImageClassification"),
        ("resnet", "ResNetForImageClassification"),
        ("segformer", "SegformerForImageClassification"),
        ("shieldgemma2", "ShieldGemma2ForImageClassification"),
        ("siglip", "SiglipForImageClassification"),
        ("siglip2", "Siglip2ForImageClassification"),
        ("swiftformer", "SwiftFormerForImageClassification"),
        ("swinv2", "Swinv2ForImageClassification"),
        ("textnet", "TextNetForImageClassification"),
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

MODEL_FOR_INSTANCE_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Instance Segmentation mapping
        # MaskFormerForInstanceSegmentation can be removed from this mapping in v5
        ("maskformer", "MaskFormerForInstanceSegmentation"),
    ]
)

MODEL_FOR_UNIVERSAL_SEGMENTATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Universal Segmentation mapping
        ("detr", "DetrForSegmentation"),
        ("eomt", "EomtForUniversalSegmentation"),
        ("mask2former", "Mask2FormerForUniversalSegmentation"),
        ("maskformer", "MaskFormerForInstanceSegmentation"),
        ("oneformer", "OneFormerForUniversalSegmentation"),
    ]
)

MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("timesformer", "TimesformerForVideoClassification"),
        ("videomae", "VideoMAEForVideoClassification"),
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
        ("kosmos-2.5", "Kosmos2_5ForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_next_video", "LlavaNextVideoForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("mistral3", "Mistral3ForConditionalGeneration"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("ovis2", "Ovis2ForConditionalGeneration"),
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
        ("aya_vision", "AyaVisionForConditionalGeneration"),
        ("blip", "BlipForConditionalGeneration"),
        ("blip-2", "Blip2ForConditionalGeneration"),
        ("chameleon", "ChameleonForConditionalGeneration"),
        ("cohere2_vision", "Cohere2VisionForConditionalGeneration"),
        ("deepseek_vl", "DeepseekVLForConditionalGeneration"),
        ("deepseek_vl_hybrid", "DeepseekVLHybridForConditionalGeneration"),
        ("fuyu", "FuyuForCausalLM"),
        ("emu3", "Emu3ForConditionalGeneration"),
        ("florence2", "Florence2ForConditionalGeneration"),
        ("gemma3", "Gemma3ForConditionalGeneration"),
        ("git", "GitForCausalLM"),
        ("got_ocr2", "GotOcr2ForConditionalGeneration"),
        ("idefics", "IdeficsForVisionText2Text"),
        ("idefics2", "Idefics2ForConditionalGeneration"),
        ("idefics3", "Idefics3ForConditionalGeneration"),
        ("instructblip", "InstructBlipForConditionalGeneration"),
        ("kosmos-2", "Kosmos2ForConditionalGeneration"),
        ("kosmos-2.5", "Kosmos2_5ForConditionalGeneration"),
        ("llama4", "Llama4ForConditionalGeneration"),
        ("lfm2_vl", "Lfm2VlForConditionalGeneration"),
        ("llava", "LlavaForConditionalGeneration"),
        ("llava_next", "LlavaNextForConditionalGeneration"),
        ("llava_onevision", "LlavaOnevisionForConditionalGeneration"),
        ("mistral3", "Mistral3ForConditionalGeneration"),
        ("mllama", "MllamaForConditionalGeneration"),
        ("ovis2", "Ovis2ForConditionalGeneration"),
        ("paligemma", "PaliGemmaForConditionalGeneration"),
        ("pix2struct", "Pix2StructForConditionalGeneration"),
        ("qwen2_5_vl", "Qwen2_5_VLForConditionalGeneration"),
        ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
        ("shieldgemma2", "Gemma3ForConditionalGeneration"),
        ("smolvlm", "SmolVLMForConditionalGeneration"),
        ("vipllava", "VipLlavaForConditionalGeneration"),
        ("vision-encoder-decoder", "VisionEncoderDecoderModel"),
        ("udop", "UdopForConditionalGeneration"),
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
        ("esm", "EsmForMaskedLM"),
        ("fnet", "FNetForMaskedLM"),
        ("funnel", "FunnelForMaskedLM"),
        ("ibert", "IBertForMaskedLM"),
        ("longformer", "LongformerForMaskedLM"),
        ("luke", "LukeForMaskedLM"),
        ("mobilebert", "MobileBertForMaskedLM"),
        ("mra", "MraForMaskedLM"),
        ("mpnet", "MPNetForMaskedLM"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nystromformer", "NystromformerForMaskedLM"),
        ("perceiver", "PerceiverForMaskedLM"),
        ("rembert", "RemBertForMaskedLM"),
        ("roformer", "RoFormerForMaskedLM"),
        ("roc_bert", "RoCBertForMaskedLM"),
        ("roberta", "RobertaForMaskedLM"),
        ("squeezebert", "SqueezeBertForMaskedLM"),
        ("tapas", "TapasForMaskedLM"),
        ("wav2vec2", "Wav2Vec2ForMaskedLM"),
        ("xlm", "XLMWithLMHeadModel"),
        ("xlm-roberta", "XLMRobertaForMaskedLM"),
        ("xlm-roberta-xl", "XLMRobertaXLForMaskedLM"),
        ("xmod", "XmodForMaskedLM"),
        ("yoso", "YosoForMaskedLM"),
    ]
)

MODEL_FOR_OBJECT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Object Detection mapping
        ("d_fine", "DFineForObjectDetection"),
        ("deformable_detr", "DeformableDetrForObjectDetection"),
        ("deta", "DetaForObjectDetection"),
        ("detr", "DetrForObjectDetection"),
        ("rt_detr", "RTDetrForObjectDetection"),
        ("rt_detr_v2", "RTDetrV2ForObjectDetection"),
        ("table-transformer", "TableTransformerForObjectDetection"),
        ("yolos", "YolosForObjectDetection"),
        ("mm-grounding-dino", "MMGroundingDinoForObjectDetection"),
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
        ("granite_speech", "GraniteSpeechForConditionalGeneration"),
        ("led", "LEDForConditionalGeneration"),
        ("longt5", "LongT5ForConditionalGeneration"),
        ("m2m_100", "M2M100ForConditionalGeneration"),
        ("marian", "MarianMTModel"),
        ("mvp", "MvpForConditionalGeneration"),
        ("nllb-moe", "NllbMoeForConditionalGeneration"),
        ("mt5", "MT5ForConditionalGeneration"),
        ("plbart", "PLBartForConditionalGeneration"),
        ("pegasus", "PegasusForConditionalGeneration"),
        ("pegasus_x", "PegasusXForConditionalGeneration"),
        ("qwen2_audio", "Qwen2AudioForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForTextToText"),
        ("seamless_m4t_v2", "SeamlessM4Tv2ForTextToText"),
        ("squeezebert", "SqueezeBertForSequenceClassification"),
        ("t5", "T5ForConditionalGeneration"),
        ("t5gemma", "T5GemmaForConditionalGeneration"),
        ("umt5", "UMT5ForConditionalGeneration"),
        ("voxtral", "VoxtralForConditionalGeneration"),
        ("xlm-prophetnet", "XLMProphetNetForConditionalGeneration"),
    ]
)

MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES = OrderedDict(
    [
        ("dia", "DiaForConditionalGeneration"),
        ("granite_speech", "GraniteSpeechForConditionalGeneration"),
        ("kyutai_speech_to_text", "KyutaiSpeechToTextForConditionalGeneration"),
        ("moonshine", "MoonshineForConditionalGeneration"),
        ("pop2piano", "Pop2PianoForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForSpeechToText"),
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
        ("arcee", "ArceeForSequenceClassification"),
        ("bart", "BartForSequenceClassification"),
        ("bert", "BertForSequenceClassification"),
        ("biogpt", "BioGptForSequenceClassification"),
        ("bloom", "BloomForSequenceClassification"),
        ("camembert", "CamembertForSequenceClassification"),
        ("canine", "CanineForSequenceClassification"),
        ("convbert", "ConvBertForSequenceClassification"),
        ("ctrl", "CTRLForSequenceClassification"),
        ("deberta", "DebertaForSequenceClassification"),
        ("deberta-v2", "DebertaV2ForSequenceClassification"),
        ("diffllama", "DiffLlamaForSequenceClassification"),
        ("distilbert", "DistilBertForSequenceClassification"),
        ("doge", "DogeForSequenceClassification"),
        ("electra", "ElectraForSequenceClassification"),
        ("esm", "EsmForSequenceClassification"),
        ("exaone4", "Exaone4ForSequenceClassification"),
        ("falcon", "FalconForSequenceClassification"),
        ("fnet", "FNetForSequenceClassification"),
        ("funnel", "FunnelForSequenceClassification"),
        ("gemma", "GemmaForSequenceClassification"),
        ("gemma2", "Gemma2ForSequenceClassification"),
        ("glm", "GlmForSequenceClassification"),
        ("gpt_bigcode", "GPTBigCodeForSequenceClassification"),
        ("gptj", "GPTJForSequenceClassification"),
        ("gpt_neo", "GPTNeoForSequenceClassification"),
        ("gpt_neox", "SequenceClassification"),
        ("helium", "HeliumForSequenceClassification"),
        ("hubert", "HubertForSequenceClassification"),
        ("hunyuan_v1_dense", "HunYuanDenseV1ForSequenceClassification"),
        ("hunyuan_v1_moe", "HunYuanMoEV1ForSequenceClassification"),
        ("ibert", "IBertForSequenceClassification"),
        ("jamba", "JambaForSequenceClassification"),
        ("jetmoe", "JetMoeForSequenceClassification"),
        ("layoutlm", "LayoutLMForSequenceClassification"),
        ("layoutlmv3", "LayoutLMv3ForSequenceClassification"),
        ("led", "LEDForSequenceClassification"),
        ("lilt", "LiltForSequenceClassification"),
        ("llama", "LlamaForSequenceClassification"),
        ("longformer", "LongformerForSequenceClassification"),
        ("luke", "LukeForSequenceClassification"),
        ("markuplm", "MarkupLMForSequenceClassification"),
        ("mbart", "MBartForSequenceClassification"),
        ("megatron-bert", "MegatronBertForSequenceClassification"),
        ("mobilebert", "MobileBertForSequenceClassification"),
        ("modernbert-decoder", "ModernBertDecoderForSequenceClassification"),
        ("mpnet", "MPNetForSequenceClassification"),
        ("mpt", "MptForSequenceClassification"),
        ("mra", "MraForSequenceClassification"),
        ("mt5", "MT5ForSequenceClassification"),
        ("ministral", "MinistralForSequenceClassification"),
        ("mistral", "MistralForSequenceClassification"),
        ("mixtral", "MixtralForSequenceClassification"),
        ("mvp", "MvpForSequenceClassification"),
        ("nemotron", "NemotronForSequenceClassification"),
        ("perceiver", "PerceiverForSequenceClassification"),
        ("persimmon", "PersimmonForSequenceClassification"),
        ("plbart", "PLBartForSequenceClassification"),
        ("nystromformer", "NystromformerForSequenceClassification"),
        ("opt", "OPTForSequenceClassification"),
        ("phi", "PhiForSequenceClassification"),
        ("phi3", "Phi3ForSequenceClassification"),
        ("qwen2", "Qwen2ForSequenceClassification"),
        ("qwen2_moe", "Qwen2MoeForSequenceClassification"),
        ("qwen3_next", "Qwen3NextForSequenceClassification"),
        ("rembert", "RemBertForSequenceClassification"),
        ("roformer", "RoFormerForSequenceClassification"),
        ("roc_bert", "RoCBertForSequenceClassification"),
        ("roberta", "RobertaForSequenceClassification"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForSequenceClassification"),
        ("seed_oss", "SeedOssForSequenceClassification"),
        ("smollm3", "SmolLM3ForSequenceClassification"),
        ("stablelm", "StableLmForSequenceClassification"),
        ("starcoder2", "Starcoder2ForSequenceClassification"),
        ("t5", "T5ForSequenceClassification"),
        ("tapas", "TapasForSequenceClassification"),
        ("t5gemma", "T5GemmaForSequenceClassification"),
        ("umt5", "UMT5ForSequenceClassification"),
        ("xlm", "XLMForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForSequenceClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForSequenceClassification"),
        ("xlnet", "XLNetForSequenceClassification"),
        ("xmod", "XmodForSequenceClassification"),
        ("yoso", "YosoForSequenceClassification"),
        ("zamba", "ZambaForSequenceClassification"),
        ("zamba2", "Zamba2ForSequenceClassification"),
    ]
)

MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        # Model for Question Answering mapping
        ("albert", "AlbertForQuestionAnswering"),
        ("arcee", "ArceeForQuestionAnswering"),
        ("bart", "BartForQuestionAnswering"),
        ("bloom", "BloomForQuestionAnswering"),
        ("bert", "BertForQuestionAnswering"),
        ("camembert", "CamembertForQuestionAnswering"),
        ("canine", "CanineForQuestionAnswering"),
        ("convbert", "ConvBertForQuestionAnswering"),
        ("deberta", "DebertaForQuestionAnswering"),
        ("deberta-v2", "DebertaV2ForQuestionAnswering"),
        ("diffllama", "DiffLlamaForQuestionAnswering"),
        ("distilbert", "DistilBertForQuestionAnswering"),
        ("electra", "ElectraForQuestionAnswering"),
        ("exaone4", "Exaone4ForQuestionAnswering"),
        ("fnet", "FNetForQuestionAnswering"),
        ("falcon", "FalconForQuestionAnswering"),
        ("funnel", "FunnelForQuestionAnswering"),
        ("gpt_neo", "GPTNeoForQuestionAnswering"),
        ("gpt_neox", "GPTNeoXForQuestionAnswering"),
        ("gptj", "GPTJForQuestionAnswering"),
        ("ibert", "IBertForQuestionAnswering"),
        ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
        ("led", "LEDForQuestionAnswering"),
        ("lilt", "LiltForQuestionAnswering"),
        ("luke", "LukeForQuestionAnswering"),
        ("llama", "LlamaForQuestionAnswering"),
        ("longformer", "LongformerForQuestionAnswering"),
        ("markuplm", "MarkupLMForQuestionAnswering"),
        ("mbart", "MBartForQuestionAnswering"),
        ("megatron-bert", "MegatronBertForQuestionAnswering"),
        ("ministral", "MinistralForQuestionAnswering"),
        ("mistral", "MistralForQuestionAnswering"),
        ("mixtral", "MixtralForQuestionAnswering"),
        ("mobilebert", "MobileBertForQuestionAnswering"),
        ("mpnet", "MPNetForQuestionAnswering"),
        ("mpt", "MptForQuestionAnswering"),
        ("mra", "MraForQuestionAnswering"),
        ("mvp", "MvpForQuestionAnswering"),
        ("nemotron", "NemotronForQuestionAnswering"),
        ("nystromformer", "NystromformerForQuestionAnswering"),
        ("opt", "OPTForQuestionAnswering"),
        ("qwen2", "Qwen2ForQuestionAnswering"),
        ("qwen3_next", "Qwen3NextForQuestionAnswering"),
        ("rembert", "RemBertForQuestionAnswering"),
        ("roformer", "RoFormerForQuestionAnswering"),
        ("roc_bert", "RoCBertForQuestionAnswering"),
        ("roberta", "RobertaForQuestionAnswering"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForQuestionAnswering"),
        ("seed_oss", "SeedOssForQuestionAnswering"),
        ("squeezebert", "SqueezeBertForQuestionAnswering"),
        ("t5", "T5ForQuestionAnswering"),
        ("umt5", "UMT5ForQuestionAnswering"),
        ("xlm", "XLMForSequenceClassification"),
        ("xlm-roberta", "XLMRobertaForQuestionAnswering"),
        ("xlm-roberta-xl", "XLMRobertaXLForQuestionAnswering"),
        ("xlnet", "XLNetForQuestionAnsweringSimple"),
        ("xmod", "XmodForQuestionAnswering"),
        ("yoso", "YosoForQuestionAnswering"),
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

MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES = OrderedDict(
    [
        ("layoutlm", "LayoutLMForQuestionAnswering"),
        ("layoutlmv3", "LayoutLMv3ForQuestionAnswering"),
        ("qwen2_moe", "Qwen2MoeForQuestionAnswering"),
    ]
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        # Model for Token Classification mapping
        ("albert", "AlbertForTokenClassification"),
        ("apertus", "ApertusForTokenClassification"),
        ("arcee", "ArceeForTokenClassification"),
        ("bloom", "BloomForTokenClassification"),
        ("bert", "BertForTokenClassification"),
        ("biogpt", "BioGptForTokenClassification"),
        ("bloom", "BloomForTokenClassification"),
        ("camembert", "CamembertForTokenClassification"),
        ("canine", "CanineForTokenClassification"),
        ("convbert", "ConvBertForTokenClassification"),
        ("bros", "BrosForTokenClassification"),
        ("deberta", "DebertaForTokenClassification"),
        ("deberta-v2", "DebertaV2ForTokenClassification"),
        ("diffllama", "DiffLlamaForTokenClassification"),
        ("distilbert", "DistilBertForTokenClassification"),
        ("esm", "EsmForTokenClassification"),
        ("exaone4", "Exaone4ForTokenClassification"),
        ("fnet", "FNetForTokenClassification"),
        ("funnel", "FunnelForTokenClassification"),
        ("falcon", "FalconForTokenClassification"),
        ("glm", "GlmForTokenClassification"),
        ("gpt_neo", "GPTNeoForTokenClassification"),
        ("gpt_neox", "GPTNeoXForTokenClassification"),
        ("electra", "ElectraForTokenClassification"),
        ("helium", "HeliumForTokenClassification"),
        ("ibert", "IBertForTokenClassification"),
        ("layoutlm", "LayoutLMForTokenClassification"),
        ("layoutlmv3", "LayoutLMv3ForTokenClassification"),
        ("lilt", "LiltForTokenClassification"),
        ("llama", "LlamaForTokenClassification"),
        ("longformer", "LongformerForTokenClassification"),
        ("luke", "LukeForTokenClassification"),
        ("ministral", "MinistralForTokenClassification"),
        ("mistral", "MistralForTokenClassification"),
        ("mobilebert", "MobileBertForTokenClassification"),
        ("mpnet", "MPNetForTokenClassification"),
        ("mt5", "MT5ForTokenClassification"),
        ("megatron-bert", "MegatronBertForTokenClassification"),
        ("mixtral", "MixtralForTokenClassification"),
        ("markuplm", "MarkupLMForTokenClassification"),
        ("mpt", "MptForTokenClassification"),
        ("mra", "MraForTokenClassification"),
        ("nemotron", "NemotronForTokenClassification"),
        ("nystromformer", "NystromformerForTokenClassification"),
        ("persimmon", "PersimmonForTokenClassification"),
        ("phi", "PhiForTokenClassification"),
        ("phi3", "Phi3ForTokenClassification"),
        ("qwen2", "Qwen2ForTokenClassification"),
        ("qwen2_moe", "Qwen2MoeForTokenClassification"),
        ("qwen3_next", "Qwen3NextForTokenClassification"),
        ("rembert", "RemBertForTokenClassification"),
        ("roberta", "RobertaForTokenClassification"),
        ("roc_bert", "RoCBertForTokenClassification"),
        ("roberta", "RobertaForTokenClassification"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForTokenClassification"),
        ("roformer", "RoFormerForTokenClassification"),
        ("seed_oss", "SeedOssForTokenClassification"),
        ("smollm3", "SmolLM3ForTokenClassification"),
        ("starcoder2", "Starcoder2ForTokenClassification"),
        ("squeezebert", "SqueezeBertForTokenClassification"),
        ("stablelm", "StableLmForTokenClassification"),
        ("t5", "T5ForTokenClassification"),
        ("t5gemma", "T5GemmaForTokenClassification"),
        ("umt5", "UMT5ForTokenClassification"),
        ("xlm", "XLMForTokenClassification"),
        ("xlm-roberta", "XLMRobertaForTokenClassification"),
        ("xlm-roberta-xl", "XLMRobertaXLForTokenClassification"),
        ("xlnet", "XLNetForTokenClassification"),
        ("xmod", "XmodForTokenClassification"),
        ("yoso", "YosoForTokenClassification"),
    ]
)

MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES = OrderedDict(
    [
        # Model for Multiple Choice mapping
        ("albert", "AlbertForMultipleChoice"),
        ("bert", "BertForMultipleChoice"),
        ("camembert", "CamembertForMultipleChoice"),
        ("convbert", "ConvBertForMultipleChoice"),
        ("canine", "CanineForMultipleChoice"),
        ("deberta-v2", "DebertaV2ForMultipleChoice"),
        ("distilbert", "DistilBertForMultipleChoice"),
        ("electra", "ElectraForMultipleChoice"),
        ("fnet", "FNetForMultipleChoice"),
        ("funnel", "FunnelForMultipleChoice"),
        ("ibert", "IBertForMultipleChoice"),
        ("longformer", "LongformerForMultipleChoice"),
        ("luke", "LukeForMultipleChoice"),
        ("megatron-bert", "MegatronBertForMultipleChoice"),
        ("mobilebert", "MobileBertForMultipleChoice"),
        ("mra", "MraForMultipleChoice"),
        ("mpnet", "MPNetForMultipleChoice"),
        ("nystromformer", "NystromformerForMultipleChoice"),
        ("rembert", "RemBertForMultipleChoice"),
        ("roberta", "RobertaForMultipleChoice"),
        ("roformer", "RoFormerForMultipleChoice"),
        ("roc_bert", "RoCBertForMultipleChoice"),
        ("roberta-prelayernorm", "RobertaPreLayerNormForMultipleChoice"),
        ("squeezebert", "SqueezeBertForMultipleChoice"),
        ("xlm", "XLMForMultipleChoice"),
        ("xlm-roberta", "XLMRobertaForMultipleChoice"),
        ("xlm-roberta-xl", "XLMRobertaXLForMultipleChoice"),
        ("xlnet", "XLNetForMultipleChoice"),
        ("xmod", "XmodForMultipleChoice"),
        ("yoso", "YosoForMultipleChoice"),
    ]
)

MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("bert", "BertForNextSentencePrediction"),
        ("fnet", "FNetForNextSentencePrediction"),
        ("megatron-bert", "MegatronBertForNextSentencePrediction"),
        ("mobilebert", "MobileBertForNextSentencePrediction"),
    ]
)

MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("audio-spectrogram-transformer", "ASTForAudioClassification"),
        ("sew", "SEWForSequenceClassification"),
        ("sew-d", "SEWDForSequenceClassification"),
        ("unispeech", "UniSpeechForSequenceClassification"),
        ("unispeech-sat", "UniSpeechSatForSequenceClassification"),
        ("wav2vec2", "Wav2Vec2ForSequenceClassification"),
        ("wav2vec2-bert", "Wav2Vec2BertForSequenceClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForSequenceClassification"),
        ("wavlm", "WavLMForSequenceClassification"),
        ("whisper", "WhisperForAudioClassification"),
    ]
)

MODEL_FOR_CTC_MAPPING_NAMES = OrderedDict(
    [
        ("parakeet_ctc", "ParakeetForCTC"),
        ("sew", "SEWForCTC"),
        ("sew-d", "SEWDForCTC"),
        ("unispeech", "UniSpeechForCTC"),
        ("unispeech-sat", "UniSpeechSatForCTC"),
        ("wav2vec2", "Wav2Vec2ForCTC"),
        ("wav2vec2-bert", "Wav2Vec2BertForCTC"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForCTC"),
        ("wavlm", "WavLMForAudioFrameClassification"),
    ]
)

MODEL_FOR_AUDIO_FRAME_CLASSIFICATION_MAPPING_NAMES = OrderedDict(
    [
        ("unispeech-sat", "UniSpeechSatForAudioFrameClassification"),
        ("wav2vec2", "Wav2Vec2ForAudioFrameClassification"),
        ("wav2vec2-bert", "Wav2Vec2BertForAudioFrameClassification"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForAudioFrameClassification"),
    ]
)

MODEL_FOR_AUDIO_XVECTOR_MAPPING_NAMES = OrderedDict(
    [
        ("unispeech-sat", "UniSpeechSatForXVector"),
        ("wav2vec2", "Wav2Vec2ForXVector"),
        ("wav2vec2-bert", "Wav2Vec2BertForXVector"),
        ("wav2vec2-conformer", "Wav2Vec2ConformerForXVector"),
        ("wavlm", "WavLMForXVector"),
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
        ("csm", "CsmForConditionalGeneration"),
        ("fastspeech2_conformer", "FastSpeech2ConformerWithHifiGan"),
        ("musicgen", "MusicgenForConditionalGeneration"),
        ("musicgen_melody", "MusicgenMelodyForConditionalGeneration"),
        ("seamless_m4t", "SeamlessM4TForTextToSpeech"),
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
        ("metaclip_2", "MetaClip2Model"),
        ("siglip", "SiglipModel"),
    ]
)

MODEL_FOR_BACKBONE_MAPPING_NAMES = OrderedDict(
    [
        # Backbone mapping
        ("beit", "BeitBackbone"),
        ("convnext", "ConvNextBackbone"),
        ("convnextv2", "ConvNextV2Backbone"),
        ("dinov2", "Dinov2Backbone"),
        ("dinov2_with_registers", "Dinov2WithRegistersBackbone"),
        ("focalnet", "FocalNetBackbone"),
        ("hgnet_v2", "HGNetV2Backbone"),
        ("hiera", "HieraBackbone"),
        ("maskformer-swin", "MaskFormerSwinBackbone"),
        ("pvt_v2", "PvtV2Backbone"),
        ("resnet", "ResNetBackbone"),
        ("rt_detr_resnet", "RTDetrResNetBackbone"),
        ("swin", "SwinBackbone"),
        ("swinv2", "Swinv2Backbone"),
        ("textnet", "TextNetBackbone"),
        ("vitdet", "VitDetBackbone"),
        ("vitpose_backbone", "VitPoseBackbone"),
    ]
)

MODEL_FOR_MASK_GENERATION_MAPPING_NAMES = OrderedDict(
    [
        ("sam", "SamModel"),
        ("sam2", "Sam2Model"),
        ("sam2_video", "Sam2VideoModel"),
        ("sam_hq", "SamHQModel"),
    ]
)


MODEL_FOR_KEYPOINT_DETECTION_MAPPING_NAMES = OrderedDict(
    [
        ("efficientloftr", "EfficientLoFTRForKeypointMatching"),
    ]
)

MODEL_FOR_KEYPOINT_MATCHING_MAPPING_NAMES = OrderedDict(
    [
        ("efficientloftr", "EfficientLoFTRForKeypointMatching"),
    ]
)


MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES = OrderedDict(
    [
        ("albert", "AlbertModel"),
        ("bert", "BertModel"),
        ("deberta", "DebertaModel"),
        ("deberta-v2", "DebertaV2Model"),
        ("distilbert", "DistilBertModel"),
        ("electra", "ElectraModel"),
        ("emu3", "Emu3TextModel"),
        ("ibert", "IBertModel"),
        ("llama4", "Llama4TextModel"),
        ("longformer", "LongformerModel"),
        ("mllama", "MllamaTextModel"),
        ("mobilebert", "MobileBertModel"),
        ("mt5", "MT5EncoderModel"),
        ("nystromformer", "NystromformerModel"),
        ("rembert", "RemBertModel"),
        ("roformer", "RoFormerModel"),
        ("roc_bert", "RoCBertModel"),
        ("roberta", "RobertaModel"),
        ("roberta-prelayernorm", "RobertaPreLayerNormModel"),
        ("squeezebert", "SqueezeBertModel"),
        ("t5", "T5EncoderModel"),
        ("t5gemma", "T5GemmaEncoderModel"),
        ("umt5", "UMT5EncoderModel"),
        ("xlm", "XLMModel"),
        ("xlm-roberta", "XLMRobertaModel"),
        ("xlm-roberta-xl", "XLMRobertaXLModel"),
    ]
)

MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES = OrderedDict()

MODEL_FOR_TIME_SERIES_PREDICTION_MAPPING_NAMES = OrderedDict(
    [
        ("timesfm", "TimesFmModelForPrediction"),
    ]
)

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES = OrderedDict(
    [
        ("swin2sr", "Swin2SRForImageSuperResolution"),
    ]
)

MODEL_FOR_AUDIO_TOKENIZATION_NAMES = OrderedDict(
    [
        ("dac", "DacModel"),
    ]
)


if version.parse(transformers.__version__) >= version.parse("4.51.0"):
    MODEL_MAPPING_NAMES.update({"qwen3_moe": "Qwen3MoeModel"})
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.update({"qwen3_moe": "Qwen3MoeForCausalLM"})
    MODEL_MAPPING_NAMES.update({"qwen3": "Qwen3Model"})
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.update({"qwen3": "Qwen3ForCausalLM"})
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.update({"qwen3_moe": "Qwen3MoeForSequenceClassification"})
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.update({"qwen3": "Qwen3ForSequenceClassification"})
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.update({"qwen3_moe": "Qwen3MoeForQuestionAnswering"})
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES.update({"qwen3": "Qwen3ForQuestionAnswering"})
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES.update({"qwen3_moe": "Qwen3MoeForTokenClassification"})
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

if version.parse(transformers.__version__) >= version.parse("4.57.0"):
    MODEL_MAPPING_NAMES.update(
        {
            ("qwen3_vl", "Qwen3VLModel"),
            ("qwen3_vl_moe", "Qwen3VLMoeModel"),
            ("qwen3_vl_moe_text", "Qwen3VLMoeTextModel"),
            ("qwen3_vl_text", "Qwen3VLTextModel"),
        }
    )
    MODEL_FOR_VISION_2_SEQ_MAPPING_NAMES.update(
        {
            ("qwen3_vl", "Qwen3VLForConditionalGeneration"),
            ("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration"),
        }
    )
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES.update(
        {("qwen3_vl", "Qwen3VLForConditionalGeneration"), ("qwen3_vl_moe", "Qwen3VLMoeForConditionalGeneration")}
    )

    MODEL_FOR_TEXT_TO_WAVEFORM_MAPPING_NAMES.update({("qwen3_omni_moe", "Qwen3OmniMoeForConditionalGeneration")})

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

MODEL_FOR_KEYPOINT_MATCHING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_KEYPOINT_MATCHING_MAPPING_NAMES)

MODEL_FOR_TEXT_ENCODING_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_TEXT_ENCODING_MAPPING_NAMES)

MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_CLASSIFICATION_MAPPING_NAMES
)

MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TIME_SERIES_REGRESSION_MAPPING_NAMES
)

MODEL_FOR_IMAGE_TO_IMAGE_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_IMAGE_TO_IMAGE_MAPPING_NAMES)

MODEL_FOR_AUDIO_TOKENIZATION_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_FOR_AUDIO_TOKENIZATION_NAMES)


class AutoModelForMaskGeneration(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_MASK_GENERATION_MAPPING


class AutoModelForKeypointDetection(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_KEYPOINT_DETECTION_MAPPING


class AutoModelForKeypointMatching(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_KEYPOINT_MATCHING_MAPPING


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


class AutoModelForAudioTokenization(_BaseAutoModelClass):
    _model_mapping = MODEL_FOR_AUDIO_TOKENIZATION_MAPPING


AutoModelForAudioTokenization = auto_class_update(
    AutoModelForAudioTokenization, head_doc="audio tokenization through codebooks"
)


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
