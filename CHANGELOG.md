# Changelog

## [v0.5.0] - 2025-12-23

### Compatibility Updates
- **mindone.diffusers**: Compatible with 🤗 diffusers v0.35.2, preview supports for sota v0.36 pipelines
- **mindone.transformers**: Compatible with 🤗 transformers v4.57.1
- **ComfyUI**: Added initial ComfyUI integration support
- **MindSpore**: Compatible with MindSpore 2.6.0 - 2.7.1

### mindone.transformers updates
- **Major upgrade**: Enhanced compatibility with 🤗 transformers v4.54 and v4.57.1. Check supported models [here](./mindone/transformers/SUPPORT_LIST.md).

#### Base Updates
- Transformers 4.54 base support ([#1387](https://github.com/mindspore-lab/mindone/pull/1387))
- Transformers 4.57 base support ([#1445](https://github.com/mindspore-lab/mindone/pull/1445))

#### New Models
- **Vision Models**:
  - AIMv2 ([#1456](https://github.com/mindspore-lab/mindone/pull/1456))
  - DINOv3 ViT/ConvNeXt (v4.57.1) ([#1439](https://github.com/mindspore-lab/mindone/pull/1439))
  - SAM-HQ (v4.57.1) ([#1457](https://github.com/mindspore-lab/mindone/pull/1457))
  - Bria ([#1384](https://github.com/mindspore-lab/mindone/pull/1384))
  - Florence2 ([#1453](https://github.com/mindspore-lab/mindone/pull/1453))
  - EfficientLoftr ([#1456](https://github.com/mindspore-lab/mindone/pull/1456))
  - HGNet_v2 ([#1395](https://github.com/mindspore-lab/mindone/pull/1395))
  - Ovis2 ([#1454](https://github.com/mindspore-lab/mindone/pull/1454))

- **Audio/Speech Models**:
  - Granite Speech ([#1406](https://github.com/mindspore-lab/mindone/pull/1406))
  - Kyutai Speech-to-Text ([#1407](https://github.com/mindspore-lab/mindone/pull/1407))
  - Voxtral ([#1456](https://github.com/mindspore-lab/mindone/pull/1456))
  - Parakeet ([#1451](https://github.com/mindspore-lab/mindone/pull/1451))
  - XCodec ([#1452](https://github.com/mindspore-lab/mindone/pull/1452))
  - Dia ([#1404](https://github.com/mindspore-lab/mindone/pull/1404))
  - CSM (v4.54.1) ([#1399](https://github.com/mindspore-lab/mindone/pull/1399))

- **Text/Language Models**:
  - Llama4 ([#1470](https://github.com/mindspore-lab/mindone/pull/1470))
  - Arcee ([#1470](https://github.com/mindspore-lab/mindone/pull/1470))
  - Falcon H1 ([#1465](https://github.com/mindspore-lab/mindone/pull/1465))
  - Dots1 ([#1469](https://github.com/mindspore-lab/mindone/pull/1469))
  - SmolLM3 (v4.54.1) ([#1391](https://github.com/mindspore-lab/mindone/pull/1391))
  - ModernBERT Decoder (v4.54.1) ([#1397](https://github.com/mindspore-lab/mindone/pull/1397))
  - Hunyuan V1 Dense/MoE (v4.57.1) ([#1401](https://github.com/mindspore-lab/mindone/pull/1401))
  - Evolla (v4.54.1) ([#1440](https://github.com/mindspore-lab/mindone/pull/1440))
  - EXAONE ([#1396](https://github.com/mindspore-lab/mindone/pull/1396))
  - Doge ([#1392](https://github.com/mindspore-lab/mindone/pull/1392))
  - ERNIE 4.5 & ERNIE 4.5 MoE ([#1393](https://github.com/mindspore-lab/mindone/pull/1393))
  - GLM4 MoE ([#1409](https://github.com/mindspore-lab/mindone/pull/1409))
  - Flex OLMo ([#1442](https://github.com/mindspore-lab/mindone/pull/1442))
  - T5Gemma ([#1420](https://github.com/mindspore-lab/mindone/pull/1420))
  - VaultGemma ([#1450](https://github.com/mindspore-lab/mindone/pull/1450))
  - BLT/Apertus/Ministral ([#1462](https://github.com/mindspore-lab/mindone/pull/1462))
  - EOMT/TimesFM ([#1403](https://github.com/mindspore-lab/mindone/pull/1403))
  - Seed OSS ([#1441](https://github.com/mindspore-lab/mindone/pull/1441))

- **Multimodal Models**:
  - Qwen3 Omni ([#1411](https://github.com/mindspore-lab/mindone/pull/1411))
  - Qwen3 Next ([#1476](https://github.com/mindspore-lab/mindone/pull/1476))
  - ColQwen2 (v4.54.1) ([#1414](https://github.com/mindspore-lab/mindone/pull/1414))
  - Cohere2 Vision (v4.57.1) ([#1473](https://github.com/mindspore-lab/mindone/pull/1473))
  - InternVL (v4.57) ([#1463](https://github.com/mindspore-lab/mindone/pull/1463))
  - Janus (v4.57) ([#1463](https://github.com/mindspore-lab/mindone/pull/1463))
  - Kosmos-2.5 ([#1456](https://github.com/mindspore-lab/mindone/pull/1456))
  - LFM2/LFM2-VL ([#1456](https://github.com/mindspore-lab/mindone/pull/1456))
  - MetaCLIP 2 ([#1456](https://github.com/mindspore-lab/mindone/pull/1456))
  - Mlcd ([#1472](https://github.com/mindspore-lab/mindone/pull/1472))

#### Processor Updates for vllm-mindspore community
- Qwen2.5VL ImageProcessor Fast / VideoProcessor ([#1429](https://github.com/mindspore-lab/mindone/pull/1429))
- Qwen3_VL Video Processor & Qwen2_VL Image Processor Fast ([#1419](https://github.com/mindspore-lab/mindone/pull/1419))
- Phi4/Whisper/Ultravox/InternVL/Qwen2_audio/MiniCPMV/LLaVA-Next/LLaVA-Next-Video processors ([#1471](https://github.com/mindspore-lab/mindone/pull/1471))

#### Model Updates
- Update Mistral3 to v4.57.1 ([#1464](https://github.com/mindspore-lab/mindone/pull/1464))
- Update Qwen2.5VL to v4.54.1 ([#1421](https://github.com/mindspore-lab/mindone/pull/1421))

### mindone.diffusers updates

#### New Features
- Context parallelism: Ring & Ulysses & Unified Attention ([#1438](https://github.com/mindspore-lab/mindone/pull/1438))
- Added AutoencoderMixin ([#1444](https://github.com/mindspore-lab/mindone/pull/1444))

### New Pipelines
- Kandinsky5 ([#1388](https://github.com/mindspore-lab/mindone/pull/1388))
- Lucy ([#1390](https://github.com/mindspore-lab/mindone/pull/1390))

#### Bug Fixes
- Fixed some diffusers bugs ([#1448](https://github.com/mindspore-lab/mindone/pull/1448))

### Examples Updates
- Added OmniGen2 fine-tuning script ([#1410](https://github.com/mindspore-lab/mindone/pull/1410))
- Added back examples/dit_infer_acceleration (renamed to accelerated_pipelines) ([#1433](https://github.com/mindspore-lab/mindone/pull/1433))
- Updated Emu3 performance for MindSpore 2.6.0 and 2.7.0 ([#1417](https://github.com/mindspore-lab/mindone/pull/1417))
- Updated HunyuanVideo-I2V to MS 2.6.0 and MS 2.7.0 ([#1385](https://github.com/mindspore-lab/mindone/pull/1385))

### ComfyUI Integration
- Added ComfyUI root files and CLI args ([#1480](https://github.com/mindspore-lab/mindone/pull/1480))
- Added text encoder files ([#1481](https://github.com/mindspore-lab/mindone/pull/1481))
- Updated clip_model.py ([#1479](https://github.com/mindspore-lab/mindone/pull/1479))

### Fixed
- Fixed AIMv2/Arcee rely on torch bug ([#1485](https://github.com/mindspore-lab/mindone/pull/1485))
- Fixed bugs of mindone.transformers models that rely on torch ([#1482](https://github.com/mindspore-lab/mindone/pull/1482))
- Fixed Qwen2.5VLProcessor tokenizer converting tensor bug ([#1483](https://github.com/mindspore-lab/mindone/pull/1483))
- Fixed Qwen3_VL text attention selection bug ([#1455](https://github.com/mindspore-lab/mindone/pull/1455))
- Fixed GLM4.1V bs>1 generation index bug ([#1437](https://github.com/mindspore-lab/mindone/pull/1437))
- Fixed training issue in TrainOneStepWrapper ([#1408](https://github.com/mindspore-lab/mindone/pull/1408))
- Fixed import error if env contains accelerate module ([#1431](https://github.com/mindspore-lab/mindone/pull/1431))
- ZeRO: Support training with MS 2.6.0 and 2.7.0 ([#1383](https://github.com/mindspore-lab/mindone/pull/1383))
- Misc bugfixes ([#1424](https://github.com/mindspore-lab/mindone/pull/1424))

### Statistics
- **Total commits**: 63
- **Files changed**: 679
- **Lines added**: 121,102
- **Lines deleted**: 23,268
