# Changelog

## [v0.4.0] - 2025-11-02

### Added

#### Transformers Models
- Added FLAVA model (#1342)
- Added RT-DETR and RT-DETR v2 models (#1317)
- Added SegGPT model (#1318)
- Added PromptDepthAnything model (#1315)
- Added Table Transformer model (#1320)
- Added UperNet model (#1319)
- Added DiffLlama, OLMoE and other models (#1147)
- Added SwinV2 and GroupViT models (#1213)
- Added LongT5, Longformer and other models (#1234)
- Added usage examples for BERT Japanese, BERTweet, ByT5, DialoGPT, Falcon3, Flan-T5, PhoBERT and XLM-V (#1328)
- Added Granite-Vision, MatCha, and DePlot usage examples (#1334)
- Added Aya Vision, BridgeTower and other models (#1253)
- Added vision models including ViT series and ZoeDepth (#1321)
- Added Data2Vec model series (#1273)
- Added Vision Text Dual Encoder (#1225)
- Added Wav2Vec2-Conformer and Wav2Vec2-BERT models (#1312)
- Added MusicGen Melody model (#1322)
- Added MusicGen model (#1324)
- Added AST and WavLM models (#1323)
- Added Jamba model (#1274)
- Added Udop model (#1283)
- Added Cohere model (#1304)
- Added Nougat example (#1336)

#### Diffusers Pipelines
- Added pipelines and required modules of QwenImage in Diffusers Master (#1288)
- Added HiDream pipeline (#1360)
- Added Wan-VACE pipeline (#1148)
- Added SkyReels-V2 pipelines (#1203)
- Added Chroma-Dev pipeline (#1157)
- Added Sana Sprint Img2Img, VisualCloze and other pipelines (#1145)
- Added more HunyuanVideo pipelines (#1029)
- Added UniDiffusers test (#1007)
- Added OmniGen and Marigold intrinsics pipelines (#1062)
- Added LTXConditionPipeline (#997)
- Added Wan pipeline (#1021)
- Added AudioLDM2 pipeline (#981)
- Added Stable Diffusion Attend and Excite pipeline (#1013)
- Added Lumina2 pipeline (#996)
- Added K-diffusion pipelines (#986)

#### Examples Models
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

#### PEFT (Parameter-Efficient Fine-Tuning)
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
- **Models documented**: 26
- **Examples updated**: 18
