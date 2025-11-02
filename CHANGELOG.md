# Changelog

## [v0.4.0] - 2025-11-02

### Compatibility Updates
- **mindone.diffusers**: Compatible with 🤗 diffusers v0.35.0
- **mindone.transformers**: Compatible with 🤗 transformers v4.50
- **MindSpore**: Upgraded to require >=2.6.0



### mindone.transformers updates
- **Major upgrade**: Enhanced compatibility with 🤗 transformers v4.50
- **280+ models supported**: Comprehensive model library including vision, audio, multimodal, and text models

#### new models
- **Vision Models**: FLAVA ([#1342](https://github.com/mindspore-lab/mindone/pull/1342)), RT-DETR/RT-DETRv2 ([#1317](https://github.com/mindspore-lab/mindone/pull/1317)), SegGPT ([#1318](https://github.com/mindspore-lab/mindone/pull/1318)), Table Transformer ([#1320](https://github.com/mindspore-lab/mindone/pull/1320)), UperNet ([#1319](https://github.com/mindspore-lab/mindone/pull/1319)), Granite-Vision/MatCha/DePlot ([#1334](https://github.com/mindspore-lab/mindone/pull/1334)), ViT series/ZoeDepth ([#1321](https://github.com/mindspore-lab/mindone/pull/1321)), Grounding DINO ([#1175](https://github.com/mindspore-lab/mindone/pull/1175)), Idefics/Idefics3 ([#1159](https://github.com/mindspore-lab/mindone/pull/1159), [#1084](https://github.com/mindspore-lab/mindone/pull/1084)), Aria ([#1089](https://github.com/mindspore-lab/mindone/pull/1089)), CLIPSeg ([#1242](https://github.com/mindspore-lab/mindone/pull/1242)), VideoLlava/VipLllava ([#1238](https://github.com/mindspore-lab/mindone/pull/1238)), Kosmos-2 ([#1295](https://github.com/mindspore-lab/mindone/pull/1295)), Pix2Struct ([#1295](https://github.com/mindspore-lab/mindone/pull/1295))
- **Audio Models**: Wav2Vec2-Conformer/BERT ([#1312](https://github.com/mindspore-lab/mindone/pull/1312)), Seamless-M4T ([#1293](https://github.com/mindspore-lab/mindone/pull/1293)), Bark ([#1313](https://github.com/mindspore-lab/mindone/pull/1313)), Speech-Encoder-Decoder ([#1281](https://github.com/mindspore-lab/mindone/pull/1281)), UniSpeech/UniSpeech-SAT ([#1277](https://github.com/mindspore-lab/mindone/pull/1277)), Data2Vec ([#1273](https://github.com/mindspore-lab/mindone/pull/1273)), WavLM ([#1323](https://github.com/mindspore-lab/mindone/pull/1323)), HuBERT ([#1128](https://github.com/mindspore-lab/mindone/pull/1128)), CLVP ([#1259](https://github.com/mindspore-lab/mindone/pull/1259))
- **Text/Multilingual Models**: Jamba ([#1274](https://github.com/mindspore-lab/mindone/pull/1274)), Udop ([#1283](https://github.com/mindspore-lab/mindone/pull/1283)), Cohere ([#1304](https://github.com/mindspore-lab/mindone/pull/1304)), GPT-NeoX/Japanese ([#1114](https://github.com/mindspore-lab/mindone/pull/1114), [#1112](https://github.com/mindspore-lab/mindone/pull/1112)), GPT-J/BigCode ([#1115](https://github.com/mindspore-lab/mindone/pull/1115), [#1113](https://github.com/mindspore-lab/mindone/pull/1113)), StableLM ([#1070](https://github.com/mindspore-lab/mindone/pull/1070)), OLMo/OLMo2 ([#1095](https://github.com/mindspore-lab/mindone/pull/1095)), ModernBERT/RWKV/Nystromformer/Zamba ([#1241](https://github.com/mindspore-lab/mindone/pull/1241)), Mamba/Mamba2 ([#1162](https://github.com/mindspore-lab/mindone/pull/1162)), Phi ([#1073](https://github.com/mindspore-lab/mindone/pull/1073)), MiniCPM4 ([#1053](https://github.com/mindspore-lab/mindone/pull/1053)), GLM-4.1V ([#1109](https://github.com/mindspore-lab/mindone/pull/1109)), Falcon-Mamba ([#1176](https://github.com/mindspore-lab/mindone/pull/1176)), X-MOD ([#1176](https://github.com/mindspore-lab/mindone/pull/1176)), Llama3 ([#1084](https://github.com/mindspore-lab/mindone/pull/1084))
- **Multimodal Models**: Emu3 ([#1233](https://github.com/mindspore-lab/mindone/pull/1233)), BLIP/GLM4V/MPT ([#1103](https://github.com/mindspore-lab/mindone/pull/1103)), InstructBLIP/Video ([#1295](https://github.com/mindspore-lab/mindone/pull/1295)), BridgeTower ([#1253](https://github.com/mindspore-lab/mindone/pull/1253)), Aya Vision ([#1253](https://github.com/mindspore-lab/mindone/pull/1253)), LiLT ([#1272](https://github.com/mindspore-lab/mindone/pull/1272)), MGP-STR ([#1262](https://github.com/mindspore-lab/mindone/pull/1262)), TrOCR/TVP ([#1297](https://github.com/mindspore-lab/mindone/pull/1297)), GOT-OCR-2 ([#1245](https://github.com/mindspore-lab/mindone/pull/1245)), Segment Anything (SAM) ([#1223](https://github.com/mindspore-lab/mindone/pull/1223)), ColPali ([#1259](https://github.com/mindspore-lab/mindone/pull/1259))
- **Architecture Models**: DiffLlama/OLMoE ([#1147](https://github.com/mindspore-lab/mindone/pull/1147)), LongT5/Longformer ([#1234](https://github.com/mindspore-lab/mindone/pull/1234)), NLLB-MoE ([#1244](https://github.com/mindspore-lab/mindone/pull/1244)), mBART ([#1195](https://github.com/mindspore-lab/mindone/pull/1195)), ELECTRA/Pegasus/X ([#1295](https://github.com/mindspore-lab/mindone/pull/1295)), SqueezeBERT ([#1295](https://github.com/mindspore-lab/mindone/pull/1295)), IBert ([#1212](https://github.com/mindspore-lab/mindone/pull/1212)), Bamba ([#1241](https://github.com/mindspore-lab/mindone/pull/1241)), FocalNet/RegNet ([#1254](https://github.com/mindspore-lab/mindone/pull/1254)), MobileNet v1/v2 ([#1171](https://github.com/mindspore-lab/mindone/pull/1171)), DistilBERT/Funnel/MLLaMA ([#1256](https://github.com/mindspore-lab/mindone/pull/1256)), Mistral3/Pixtral/ResNet ([#1190](https://github.com/mindspore-lab/mindone/pull/1190)), BERT Generation/DeiT ([#1205](https://github.com/mindspore-lab/mindone/pull/1205)), SigLIP2 ([#1076](https://github.com/mindspore-lab/mindone/pull/1076))
- **Examples & Documentation**: BERT Japanese/BERTweet/ByT5/DialogGPT/Falcon3/Flan-T5/PhoBERT/XLM-V ([#1328](https://github.com/mindspore-lab/mindone/pull/1328)), Depth Anything V2/DiT ([#1332](https://github.com/mindspore-lab/mindone/pull/1332)), Granite-Vision/MatCha/DePlot ([#1334](https://github.com/mindspore-lab/mindone/pull/1334)), GLM4V processor ([#1349](https://github.com/mindspore-lab/mindone/pull/1349))

### mindone.diffusers updates
- **Major upgrade**: Enhanced compatibility with 🤗 diffusers v0.35.1
- **70+ pipelines supported**: Comprehensive pipeline library for text-to-image, image-to-image, text-to-video, and audio generation
- **50+ model components**: Transformers, autoencoders, controlnets, and processing modules as building blocks

#### new pipelines
- **Video Generation**: QwenImage ([#1288](https://github.com/mindspore-lab/mindone/pull/1288)), HiDream ([#1360](https://github.com/mindspore-lab/mindone/pull/1360)), Wan-VACE ([#1148](https://github.com/mindspore-lab/mindone/pull/1148)), SkyReels-V2 ([#1203](https://github.com/mindspore-lab/mindone/pull/1203)), Chroma-Dev ([#1157](https://github.com/mindspore-lab/mindone/pull/1157)), Sana Sprint Img2Img/VisualCloze ([#1145](https://github.com/mindspore-lab/mindone/pull/1145)), HunyuanVideo ([#1029](https://github.com/mindspore-lab/mindone/pull/1029)), Wan ([#1021](https://github.com/mindspore-lab/mindone/pull/1021)), Lumina2 ([#996](https://github.com/mindspore-lab/mindone/pull/996)), LTXCondition ([#997](https://github.com/mindspore-lab/mindone/pull/997)), UniDiffuser ([#979](https://github.com/mindspore-lab/mindone/pull/979))
- **Image Generation**: Amused & Ledits++ ([#976](https://github.com/mindspore-lab/mindone/pull/976)), OmniGen & Marigold ([#1062](https://github.com/mindspore-lab/mindone/pull/1062)), Stable Diffusion Attend & Excite ([#1013](https://github.com/mindspore-lab/mindone/pull/1013)), SD Unclip/PIA ([#958](https://github.com/mindspore-lab/mindone/pull/958))
- **Audio Generation**: AudioLDM2 ([#981](https://github.com/mindspore-lab/mindone/pull/981))
- **Advanced Sampling**: K-diffusion pipelines ([#986](https://github.com/mindspore-lab/mindone/pull/986))
- **Testing & Documentation**: UniDiffusers test ([#1007](https://github.com/mindspore-lab/mindone/pull/1007)), 'reuse a pipeline' docs ([#989](https://github.com/mindspore-lab/mindone/pull/989)), diffusers mint changes ([#992](https://github.com/mindspore-lab/mindone/pull/992))

#### model components
- **Video Transformers**: transformer_qwenimage ([#1288](https://github.com/mindspore-lab/mindone/pull/1288)), transformer_hidream_image, transformer_wan_vace ([#1148](https://github.com/mindspore-lab/mindone/pull/1148)), transformer_skyreels_v2 ([#1203](https://github.com/mindspore-lab/mindone/pull/1203)), transformer_chroma ([#1157](https://github.com/mindspore-lab/mindone/pull/1157)), transformer_cosmos ([#1196](https://github.com/mindspore-lab/mindone/pull/1196)), transformer_hunyuan_video_framepack ([#1029](https://github.com/mindspore-lab/mindone/pull/1029)), consisid_transformer_3d ([#1124](https://github.com/mindspore-lab/mindone/pull/1124))
- **Autoencoders**: autoencoder_kl_qwenimage ([#1288](https://github.com/mindspore-lab/mindone/pull/1288)), autoencoder_kl_cosmos ([#1196](https://github.com/mindspore-lab/mindone/pull/1196))
- **ControlNets**: controlnet_sana ([#1145](https://github.com/mindspore-lab/mindone/pull/1145)), multicontrolnet_union ([#1158](https://github.com/mindspore-lab/mindone/pull/1158))
- **Processing Modules**: cache_utils ([#1299](https://github.com/mindspore-lab/mindone/pull/1299)), auto_model ([#1158](https://github.com/mindspore-lab/mindone/pull/1158)), lora processing modules ([#1158](https://github.com/mindspore-lab/mindone/pull/1158))

### mindone.peft updates
- Added mindone.peft and upgraded to v0.15.2 ([#1194](https://github.com/mindspore-lab/mindone/pull/1194))
- Added Qwen2.5-Omni LoRA finetuning script with transformers 4.53.0 ([#1218](https://github.com/mindspore-lab/mindone/pull/1218))
- Fixed lora and lora_scale from each PEFT layer ([#1187](https://github.com/mindspore-lab/mindone/pull/1187))

### models under examples (mostly with finetune/training scripts)
- Added Janus model for unified understanding and generation ([#1378](https://github.com/mindspore-lab/mindone/pull/1378))
- Added Emu3 model for multimodal tasks ([#1233](https://github.com/mindspore-lab/mindone/pull/1233))
- Added VAR model for class-conditional image generation ([#905](https://github.com/mindspore-lab/mindone/pull/905))
- Added HunyuanVideo and HunyuanVideo-I2V models ([#1029](https://github.com/mindspore-lab/mindone/pull/1029), [#883](https://github.com/mindspore-lab/mindone/pull/883))
- Added Wan2.1 model for text/image-to-video generation ([#1363](https://github.com/mindspore-lab/mindone/pull/1363))
- Added Wan2.2 model for text/image-to-video generation ([#1243](https://github.com/mindspore-lab/mindone/pull/1243))
- Added OpenSora models (PKU and HPC-AI versions) ([#687](https://github.com/mindspore-lab/mindone/pull/687))
- Added MovieGen 30B model ([#1362](https://github.com/mindspore-lab/mindone/pull/1362))
- Added Step-Video-T2V model ([#848](https://github.com/mindspore-lab/mindone/pull/848))
- Added CogView4 model for text-to-image generation ([#874](https://github.com/mindspore-lab/mindone/pull/874))
- Added OmniGen and OmniGen2 models ([#1227](https://github.com/mindspore-lab/mindone/pull/1227))
- Added CannyEdit for image editing tasks ([#1346](https://github.com/mindspore-lab/mindone/pull/1346))
- Added SparkTTS for text-to-speech synthesis
- Added SAM2 for image segmentation ([#1200](https://github.com/mindspore-lab/mindone/pull/1200))
- Added LangSAM for language-guided segmentation ([#1369](https://github.com/mindspore-lab/mindone/pull/1369))
- Added MMaDA for multimodal generation ([#1116](https://github.com/mindspore-lab/mindone/pull/1116))

### Changed

### Fixed
- Fixed auto mapping alphabetical sorting ([#1353](https://github.com/mindspore-lab/mindone/pull/1353))
- Fixed ReformerModel unit test issues ([#1343](https://github.com/mindspore-lab/mindone/pull/1343))
- Fixed RAG model unit test bugs ([#1340](https://github.com/mindspore-lab/mindone/pull/1340))
- Fixed fast UT errors of ShieldGemma2, AltClip, GroundingDino, UDOP, Jetmoe ([#1354](https://github.com/mindspore-lab/mindone/pull/1354))
- Fixed v4.50.0 fast ut graph mode and random inputs ([#1375](https://github.com/mindspore-lab/mindone/pull/1375))
- Fixed compute_diffs Division by zero check ([#1351](https://github.com/mindspore-lab/mindone/pull/1351))
- Fixed seamless_m4t ut bugs ([#1345](https://github.com/mindspore-lab/mindone/pull/1345))
- Fixed unexpected missing_keys warning ([#1371](https://github.com/mindspore-lab/mindone/pull/1371))
- Fixed emu3 ut threshold ([#1373](https://github.com/mindspore-lab/mindone/pull/1373))
- Fixed from_pretrained bug in opensora_pku and mmada ([#1370](https://github.com/mindspore-lab/mindone/pull/1370))
- Fixed mindspore import bug in examples/qwen2_vl ([#1368](https://github.com/mindspore-lab/mindone/pull/1368))
- Fixed lora and lora_scale from each PEFT layer ([#1187](https://github.com/mindspore-lab/mindone/pull/1187))
- Moved MiniCPM example to correct location ([#1356](https://github.com/mindspore-lab/mindone/pull/1356))
- Increased the OLMo models test error passing threshold for BF16 ([#1358](https://github.com/mindspore-lab/mindone/pull/1358))
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
