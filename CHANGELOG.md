# Changelog

## [v0.4.0] - 2025-11-02

**Compatibility Updates:**
- **mindone.diffusers**: Compatible with 🤗 diffusers v0.35.0
- **mindone.transformers**: Compatible with 🤗 transformers v4.50
- **MindSpore**: Upgraded to require >=2.6.0



### mindone.transformers updates
- **Major upgrade**: Enhanced compatibility with 🤗 transformers v4.50
- **280+ models supported**: Comprehensive model library including vision, audio, multimodal, and text models

#### new models
- **Vision Models**: FLAVA (#1342), RT-DETR/RT-DETRv2 (#1317), SegGPT (#1318), Table Transformer (#1320), UperNet (#1319), Granite-Vision/MatCha/DePlot (#1334), ViT series/ZoeDepth (#1321), Grounding DINO (#1175), Idefics/Idefics3 (#1159, #1084), Aria (#1089), CLIPSeg (#1242), VideoLlava/VipLllava (#1238), Kosmos-2 (#1295), Pix2Struct (#1295)
- **Audio Models**: Wav2Vec2-Conformer/BERT (#1312), Seamless-M4T (#1293), Bark (#1313), Speech-Encoder-Decoder (#1281), UniSpeech/UniSpeech-SAT (#1277), Data2Vec (#1273), WavLM (#1323), HuBERT (#1128), CLVP (#1259)
- **Text/Multilingual Models**: Jamba (#1274), Udop (#1283), Cohere (#1304), GPT-NeoX/Japanese (#1114, #1112), GPT-J/BigCode (#1115, #1113), StableLM (#1070), OLMo/OLMo2 (#1095), ModernBERT/RWKV/Nystromformer/Zamba (#1241), Mamba/Mamba2 (#1162), Phi (#1073), MiniCPM4 (#1053), GLM-4.1V (#1109), Falcon-Mamba (#1176), X-MOD (#1176), Llama3 (#1084)
- **Multimodal Models**: Emu3 (#1233), BLIP/GLM4V/MPT (#1103), InstructBLIP/Video (#1295), BridgeTower (#1253), Aya Vision (#1253), LiLT (#1272), MGP-STR (#1262), TrOCR/TVP (#1297), GOT-OCR-2 (#1245), Segment Anything (SAM) (#1223), ColPali (#1259)
- **Architecture Models**: DiffLlama/OLMoE (#1147), LongT5/Longformer (#1234), NLLB-MoE (#1244), mBART (#1195), ELECTRA/Pegasus/X (#1295), SqueezeBERT (#1295), IBert (#1212), Bamba (#1241), FocalNet/RegNet (#1254), MobileNet v1/v2 (#1171), DistilBERT/Funnel/MLLaMA (#1256), Mistral3/Pixtral/ResNet (#1190), BERT Generation/DeiT (#1205), SigLIP2 (#1076)
- **Examples & Documentation**: BERT Japanese/BERTweet/ByT5/DialogGPT/Falcon3/Flan-T5/PhoBERT/XLM-V (#1328), Depth Anything V2/DiT (#1332), Granite-Vision/MatCha/DePlot (#1334), GLM4V processor (#1349)

### mindone.diffusers updates
- **Major upgrade**: Enhanced compatibility with 🤗 diffusers v0.35.1
- **70+ pipelines supported**: Comprehensive pipeline library for text-to-image, image-to-image, text-to-video, and audio generation
- **50+ model components**: Transformers, autoencoders, controlnets, and processing modules as building blocks

#### new pipelines
- **Video Generation**: QwenImage (#1288), HiDream (#1360), Wan-VACE (#1148), SkyReels-V2 (#1203), Chroma-Dev (#1157), Sana Sprint Img2Img/VisualCloze (#1145), HunyuanVideo (#1029), Wan (#1021), Lumina2 (#996), LTXCondition (#997), UniDiffuser (#979)
- **Image Generation**: Amused & Ledits++ (#976), OmniGen & Marigold (#1062), Stable Diffusion Attend & Excite (#1013), SD Unclip/PIA (#958)
- **Audio Generation**: AudioLDM2 (#981)
- **Advanced Sampling**: K-diffusion pipelines (#986)
- **Testing & Documentation**: UniDiffusers test (#1007), 'reuse a pipeline' docs (#989), diffusers mint changes (#992)

#### model components
- **Video Transformers**: transformer_qwenimage, transformer_hidream_image, transformer_wan_vace, transformer_skyreels_v2, transformer_chroma, transformer_cosmos, transformer_hunyuan_video_framepack, consisid_transformer_3d
- **Autoencoders**: autoencoder_kl_qwenimage, autoencoder_kl_cosmos
- **ControlNets**: controlnet_sana, multicontrolnet_union
- **Processing Modules**: cache_utils, auto_model, lora processing modules
- **Integration Components**: Model components for QwenImage, HiDream, Wan-VACE, SkyReels-V2, Chroma, Cosmos, HunyuanVideo, Sana, and other pipelines

### Examples Models
- Added Janus model for unified understanding and generation
- Added Emu3 model for multimodal tasks
- Added VAR model for class-conditional image generation
- Added HunyuanVideo and HunyuanVideo-I2V models
- Added Wan2.1 and Wan2.2 models for text/image-to-video generation
- Added OpenSora models (PKU and HPC-AI versions)
- Added MovieGen 30B model
- Added Step-Video-T2V model
- Added CogView4 model for text-to-image generation
- Added OmniGen and OmniGen2 models
- Added CannyEdit for image editing tasks
- Added SparkTTS for text-to-speech synthesis
- Added SAM2 for image segmentation
- Added LangSAM for language-guided segmentation
- Added MMaDA for multimodal generation

### PEFT (Parameter-Efficient Fine-Tuning)
- Added mindone.peft and upgraded to v0.15.2 (#1194)
- Added Qwen2.5-Omni LoRA finetuning script with transformers 4.53.0 (#1218)
- Fixed lora and lora_scale from each PEFT layer (#1187)

### Changed
- Upgraded mindone.diffusers from v0.34.0 to v0.35.1 (#1299)
- Updated GLM4V processor (#1349)
- Added fill mask pipeline support (#1038)
- Updated Llama3 documentation (#1337)
- Updated MMaDA performance in readme (#1377)
- Updated all documentation links to v0.4.0 branch
- Reorganized model support table by capability levels
- Improved installation instructions across all examples
- Updated MindSpore version requirements to >=2.6.0
- Enhanced model configuration and requirements documentation

### Fixed
- Fixed auto mapping alphabetical sorting (#1353)
- Fixed ReformerModel unit test issues (#1343)
- Fixed RAG model unit test bugs (#1340)
- Fixed fast UT errors of ShieldGemma2, AltClip, GroundingDino, UDOP, Jetmoe (#1354)
- Fixed v4.50.0 fast ut graph mode and random inputs (#1375)
- Fixed compute_diffs Division by zero check (#1351)
- Fixed seamless_m4t ut bugs (#1345)
- Fixed unexpected missing_keys warning (#1371)
- Fixed emu3 ut threshold (#1373)
- Fixed from_pretrained bug in opensora_pku and mmada (#1370)
- Fixed mindspore import bug in examples/qwen2_vl (#1368)
- Fixed lora and lora_scale from each PEFT layer (#1187)
- Moved MiniCPM example to correct location (#1356)
- Increased the OLMo models test error passing threshold for BF16 (#1358)
- Fixed missing cd commands before pip install requirements
- Corrected model links and paths in documentation
- Removed duplicate entries and non-existent models
- Fixed installation instructions and requirements
- Updated organization names and references

### Statistics
- **Total commits**: 52
- **Files changed**: 103
- **Transformers models supported**: 280+
- **Diffusers pipelines supported**: 70+
- **Examples updated**: 18
