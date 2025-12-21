# Changelog

## [v0.5.0] - 2025-12-21

### Highlights
- Added dozens of new transformer checkpoints and processors (e.g., CSM, ModernBERT Decoder, SmolLM3, Hunyuan v1, GLM4 variants, Qwen3/Qwen3-Omni, Mlcd) and refreshed the transformers base to 4.57.1, expanding support for the latest model families. (#1391, #1396, #1397, #1399, #1401, #1403, #1409, #1411, #1462, #1464, #1472)
- Expanded generative pipelines and diffusion tooling with new QwenImage, HiDream, CannyEdit, and OmniGen2 fine-tuning utilities alongside ComfyUI root assets and text encoder resources. (#1288, #1346, #1360, #1410, #1480, #1481)
- Improved interoperability and performance through AutoencoderMixin, context parallelism enhancements, accelerated pipelines, and wide-ranging compatibility fixes for MindSpore 2.6/2.7 and downstream examples. (#1433, #1438, #1444, #1383, #1385, #1417)

### Added
- Introduced many new transformer models and processors, including Exaone, Florence2, Parakeet, VaultGemma, T5Gemma, DOGE, Ernie 4.5 (dense & MoE), SAM-HQ, BLT/Apertus/Ministral, Mistral3 (4.57.1), EOMT, TimesFM, XCodec, Kyutai Speech-to-Text, and more. (#1392, #1393, #1395, #1396, #1397, #1399, #1401, #1403, #1407, #1420, #1451, #1452, #1453, #1454, #1457, #1462, #1464)
- Added new pipelines and demos such as QwenImage diffusers support, HiDream, CannyEdit, updated accelerated pipelines, and ComfyUI root/CLI assets; added text encoder files for inference setups. (#1288, #1346, #1360, #1433, #1480, #1481)
- Extended example coverage with OmniGen2 fine-tuning, MovieGen MindSpore 2.7.0 support, and new examples for MMS, Nougat, MyT5, Granite-Vision, MatCha, DePlot, fill-mask pipe usage, Jamba, and others. (#1329, #1333, #1334, #1335, #1336, #1337, #1341, #1362, #1410)
- Added AutoencoderMixin and updated CLI/processors for Phi4, Whisper, UltraVox, InternVL, Qwen2 Audio, MiniCPMV, LLaVA Next (image/video), plus supplemental GLM4V processors. (#1349, #1429, #1471, #1444)

### Fixed
- Resolved multiple transformer and pipeline bugs, including Qwen3-VL text attention selection, GLM4.1V batch generation index handling, Accelerate import guards, FA missing_keys warnings, Reformer UT stability, and division-by-zero in compute_diffs. (#1351, #1343, #1354, #1368, #1371, #1431, #1437, #1455)
- Stabilized UT thresholds for BF16, Seamless M4T, Emu3, Zero helper, and RT-DETR fast UT cases; updated fast unit tests for various models across MindSpore versions. (#1187, #1345, #1358, #1359, #1366, #1373, #1383)
- Fixed from_pretrained loading in OpenSora/OmniGen pipelines, added lora scale coverage in diffusers tests, and sorted auto-mapping to reduce nondeterminism. (#1187, #1370, #1353)

### Documentation & Examples
- Refreshed READMEs and tutorials for Llama2/Llama3, SparkTTS, SAM2/LanguageSAM, MmaDA, Wan2.1, Janus-Pro, Emu3, and general repository instructions; improved performance notes for multiple demos. (#1344, #1377, #1369, #1376, #1378, #1417)
- Updated OpenSora-HPC AI, HunyuanVideo I2V, and Emu3 examples for MindSpore 2.6.0/2.7.0 compatibility and performance. (#1385, #1417)
- Added new documentation for GPT-SW3/MADLAD-400, MMS, and various model usage guides. (#1329, #1335)

## [v0.4.0]

For details of the previous release, see the [v0.4.0 changelog](https://github.com/mindspore-lab/mindone/blob/refs/tags/v0.4.0/CHANGELOG.md).
