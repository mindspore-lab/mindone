# Changelog

## [v0.4.0] - 2025-10-30

### Added

#### Language Models
- Added FLAVA model support (#1342)
- Added RT-DETR and RT-DETR v2 models (#1317)
- Added SegGPT model (#1318)
- Added PromptDepthAnything model (#1315)
- Added Table Transformer model (#1320)
- Added UperNet model (#1319)
- Added DiffLlama, OLMoE and other models (#1147)
- Added SwinV2 and GroupViT models (#1213)
- Added LongT5, Longformer and other models (#1234)
- Added usage examples for BERT Japanese, BERTweet, ByT5, DialoGPT, Falcon3, Flan-T5, PhoBERT and XLM-V (#1328)

#### Vision Models
- Added Granite-Vision, MatCha, and DePlot usage examples (#1334)
- Added Aya Vision, BridgeTower and other models (#1253)
- Added vision models including ViT series and ZoeDepth (#1321)
- Added Data2Vec model series (#1273)
- Added Vision Text Dual Encoder (#1225)
- Added ImageToImage, ImageToText and VisualQuestionAnswering pipelines (#1197)
- Added image segmentation pipeline with SegFormer model (#1191)
- Added Llava series models (#1016)
- Added Aria model (#1089)
- Added EasyAnimateV5.1 text-to-video, image-to-video, and control-to-video models (#995)

#### Audio Models
- Added Wav2Vec2-Conformer and Wav2Vec2-BERT models (#1312)
- Added MusicGen Melody model (#1322)
- Added MusicGen model (#1324)
- Added AST and WavLM models (#1323)
- Added Speech-Encoder-Decoder model with Wav2Vec2 UT fixes (#1281)
- Added UniSpeech and UniSpeech-SAT models (#1277)
- Added Speech2Text model (#1284)
- Added Pop2Piano, FastSpeech2-Conformer and Seamless-M4T-V2 models (#1282)

#### Diffusers Pipelines
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

### Changed
- Updated Llama3 documentation (#1337)
- Added Jamba model support (#1274)
- Upgraded mindone.diffusers from v0.34.0 to v0.35.1 (#1299)
- Added Udop model (#1283)
- Added ShieldGemma2 model (#1308)
- Added Perceiver model (#1307)
- Added OneFormer model (#1306)
- Added Bros model (#1258)
- Added Seamless-M4T model (#1293)

### Fixed
- Fixed auto mapping alphabetical sorting (#1353)
- Moved MiniCPM example to correct location (#1356)
- Removed old models (#1350)
- Removed AdamW implementation variants (#1302)
- Fixed ReformerModel unit test issues (#1343)
- Fixed RAG model unit test bugs (#1340)

### Statistics
- **Total commits**: 302
- **Files changed**: 4,005
- **New models added**: 104
