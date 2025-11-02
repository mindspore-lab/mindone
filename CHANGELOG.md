# Changelog

## [v0.4.0] - 2025-11-02

**Compatibility Updates:**
- **mindone.diffusers**: Compatible with 🤗 diffusers v0.35.0
- **mindone.transformers**: Compatible with 🤗 transformers v4.50
- **MindSpore**: Upgraded to require >=2.6.0

### Added

#### Transformers Models
- **Major upgrade**: Enhanced compatibility with 🤗 transformers v4.50
- **280+ models supported**: Comprehensive model library including vision, audio, multimodal, and text models
- Added latest model architectures: FLAVA, RT-DETR v2, SegGPT, Table Transformer, UperNet, Jamba, Udop, Cohere
- Enhanced audio models: Wav2Vec2-Conformer, MusicGen, AST, WavLM series
- Improved vision capabilities: Granite-Vision, MatCha, DePlot, ViT series, ZoeDepth
- Added multilingual support: BERT Japanese, BERTweet, XLM-V, PhoBERT examples
- Extended segmentation models: Nougat for document understanding

#### Diffusers Pipelines
- **Major upgrade**: Enhanced compatibility with 🤗 diffusers v0.35.1
- **160+ pipelines supported**: Comprehensive pipeline library for text-to-image, image-to-image, text-to-video, and audio generation
- Added latest diffusion architectures: QwenImage, HiDream, Wan-VACE, SkyReels-V2, Chroma-Dev
- Enhanced video generation: HunyuanVideo, Wan, Lumina2, LTXCondition pipelines
- Improved image generation: Sana Sprint, OmniGen, Marigold, Stable Diffusion variants
- Added audio capabilities: AudioLDM2 pipeline for text-to-audio generation
- Extended conditioning: ControlNet, IP-Adapter, T2I-Adapter pipelines
- Added advanced sampling: K-diffusion, Attend and Excite methods

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
