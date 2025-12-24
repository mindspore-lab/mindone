# Support List

## [v4.50.0-v4.57.1]

mindone.transformers has been upgraded from v0.45.0 to v4.57.1 in mindone v0.5.0, adding 78 new model interfaces, aligned with 🤗 Transformers v4.57.1.

Support list for new added models.
- fp32/fp16/bf16: ✅ = passed fast UT for that precision (performed on pruned models)
- Inference: ✅ = verified with official weights.
- The usage and performance details for each model can be found in the respective PR. (e.g., `arcee` in [pr#1470](https://github.com/mindspore-lab/mindone/pull/1470)).


*  **Text models**

    | model | fp32 | fp16 | bf16 | inference | notes |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | apertus | ✅ | ✅ | ✅ | ✅ | [#1462](https://github.com/mindspore-lab/mindone/pull/1462)  |
    | arcee | ✅ | ✅ | ✅ | ✅ |  [#1470](https://github.com/mindspore-lab/mindone/pull/1470) |
    | bitnet | ✅ | ✅ | ✅ | ✖️ | quantized model inference is temporarily unsupported. [#1416](https://github.com/mindspore-lab/mindone/pull/1416) |
    | blt | ✅ | ✅ | ✅ | ✅ | [#1462](https://github.com/mindspore-lab/mindone/pull/1462)  |
    | deepseek_v2 | ✅ | ✅ | ✅ | ✅ | [#1477](https://github.com/mindspore-lab/mindone/pull/1477)  |
    | deepseek_v3 | ✅ | ✅ | ✅ | ✖️ | quantized model inference is temporarily unsupported. [#1415](https://github.com/mindspore-lab/mindone/pull/1415) |
    | doge | ✅ | ✅ | ✅ | ✅ | [#1392](https://github.com/mindspore-lab/mindone/pull/1392)   |
    | dots1 | ✅ | ✅ | ✅ | ✅ |  [#1469](https://github.com/mindspore-lab/mindone/pull/1469) |
    | ernie4_5 | ✅ | ✅ | ✅ | ✅ | [#1393](https://github.com/mindspore-lab/mindone/pull/1393)  |
    | ernie4_5_moe | ✅ | ✅ | ✅ | ✅ | 21b. requires zero3 parallel inference with 2p. [#1393](https://github.com/mindspore-lab/mindone/pull/1393) |
    | exaone4 | ✅ | ✅ | ✅ | ✅ | [#1396](https://github.com/mindspore-lab/mindone/pull/1396)  |
    | falcon_h1 | ✅ | ✅ | ✅ | ✅ | [#1465](https://github.com/mindspore-lab/mindone/pull/1465) |
    | flex_olmo | ✅ | ✅ | ✅ | ✅ | 49b. requires zero3 parallel inference with 4p. [#1442](https://github.com/mindspore-lab/mindone/pull/1442) |
    | glm4 | ✅ | ✅ | ✅ | ✅ | explore detailed usage in [example](../../examples/transformers/glm4v/README.md) |
    | glm4_moe | ✅ | ✅ | ✅ | ✖️ | 108b. not validated due to large size. [#1409](https://github.com/mindspore-lab/mindone/pull/1409) (also see [llama4](https://github.com/mindspore-lab/mindone/pull/1470) as a reference for moe+zero3 inference attempts). |
    | gpt_oss | ✖️ | ✖️ | ✖️ | ✖️ | quantized models are not yet supported. [#1209](https://github.com/mindspore-lab/mindone/pull/1209) attempts to provide a temporary workaround to bypass quantization. |
    | granitemoehybrid | ✅ | ✅ | ✅ | ✅ | [#1405](https://github.com/mindspore-lab/mindone/pull/1405)   |
    | hunyuan_v1_dense | ✅ | ✅ | ✅ | ✅ | [#1401](https://github.com/mindspore-lab/mindone/pull/1401)  |
    | hunyuan_v1_moe | ✅ | ✅ | ✅ | ✖️ | not validated. official models to be released. [#1401](https://github.com/mindspore-lab/mindone/pull/1401) |
    | lfm2 | ✅ | ✅ | ✅ | ✅ | [#1456](https://github.com/mindspore-lab/mindone/pull/1456)  |
    | longcat_flash | ✅ | ✅ | ✅ | ✖️ | 560b. not validated due to large size. [#1443](https://github.com/mindspore-lab/mindone/pull/1443) |
    | minimax | ✅ | ✅ | ✅ | ✖️ | 1TB. not validated due to the large size. [#1186](https://github.com/mindspore-lab/mindone/pull/1186) |
    | ministral | ✅ | ✅ | ✅ | ✅ | [#1462](https://github.com/mindspore-lab/mindone/pull/1462)  |
    | modernbert_decoder | ✅ | ✅ | ✅ | ✅ | [#1397](https://github.com/mindspore-lab/mindone/pull/1397)   |
    | olmo3 | ✅ | ✅ | ✅ | ✅ | [#1467](https://github.com/mindspore-lab/mindone/pull/1467)  |
    | qwen3 | ✅ | ✅ | ✅ | ✅ |  explore detailed usage in [example](../../examples/transformers/qwen3/README.md)  |
    | qwen3_moe | ✅ | ✅ | ✅ | ✖️ | not validated. [#1181](https://github.com/mindspore-lab/mindone/pull/1181) |
    | qwen3_next | ✅ | ✅ | ✅ | ✅ | 80b-A3b. requires zero3 parallel inference. [#1476](https://github.com/mindspore-lab/mindone/pull/1476) |
    | seed_oss | ✅ | ✅ | ✅ | ✅ | 36b. requires zero3 parallel inference with 4p. [#1441](https://github.com/mindspore-lab/mindone/pull/1441) |
    | t5gemma | ✅ | ✅ | ✅ | ✅ |  [#1420](https://github.com/mindspore-lab/mindone/pull/1420) |
    | vaultgemma | ✅ | ✅ | ✅ | ✅ | [#1450](https://github.com/mindspore-lab/mindone/pull/1450)  |
    | xlstm | ✅ | ✅ | ✅ | ✅ | [#1466](https://github.com/mindspore-lab/mindone/pull/1466)  |

* **Vision models**

    | model | fp32 | fp16 | bf16 | inference | notes |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | aimv2 | ✅ | ✅ | ✅ | ✅ |  [#1456](https://github.com/mindspore-lab/mindone/pull/1456) |
    | d_fine | ✖️ | ✖️ | ✖️ | ✅ | the order of results returned by `ms.mint.topk()` and `torch.topk()`may differ  when tensor elements are identical. temporarily skip the comparative tests. The model remains fully functional for users. |
    | dinov3_vit | ✅ | ✅ | ✅ | ✅ | a precision gap of ~1e-3 exists in image processing due to resize implementation differences; hence the HF processor is retained. [#1439](https://github.com/mindspore-lab/mindone/pull/1439) |
    | efficientloftr | ✅ | ✅ | ✅ | ✖️ | 🤗 transformers model raise error in `model.generate`. see [issue 42581](https://github.com/huggingface/transformers/issues/42581). |
    | eomt | ✅ | ✅ | ✅ | ✅ |  [#1403](https://github.com/mindspore-lab/mindone/pull/1403)   |
    | hgnet_v2 | ✅ | ✅ | ✖️ | ✅ | `mindspore.nn.MaxPool2d` does not support bf16 inputs. |
    | lightglue | ✖️ | ✖️ | ✖️ | ✖️ | depends on the unsupported legacy model `SuperPoint`, see [#1348](https://github.com/mindspore-lab/mindone/pull/1348) for attempts. |
    | mlcd | ✅ | ✅ | ✅ | ✅ | [#1472](https://github.com/mindspore-lab/mindone/pull/1472)   |
    | sam2 | ✅ | ✅ | ✅ | ✅ | [#1434](https://github.com/mindspore-lab/mindone/pull/1434)  |
    | sam_hq | ✅ | ✅ | ✅ | ✅ | [#1457](https://github.com/mindspore-lab/mindone/pull/1457)  |

* **Multimodal models**

    | model | fp32 | fp16 | bf16 | inference | notes |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | cohere2_vision | ✅ | ✅ | ✅ | ✅ | 112b. requires zero3 parallel inference with 6p. [#1473](https://github.com/mindspore-lab/mindone/pull/1473) |
    | colqwen2 | ✅ | ✅ | ✅ | ✅ | [#1414](https://github.com/mindspore-lab/mindone/pull/1414)  |
    | deepseek_vl | ✅ | ✅ | ✅ | ✅ | [#1477](https://github.com/mindspore-lab/mindone/pull/1477)  |
    | deepseek_vl_hybrid | ✅ | ✅ | ✅ | ✅ | [#1477](https://github.com/mindspore-lab/mindone/pull/1477)  |
    | edgetam | ✖️ | ✖️ | ✖️ | ✖️ | use `MobileNetV5` from `timm`. temporarily unsupported. |
    | edgetam_video | ✖️ | ✖️ | ✖️ | ✖️ | use `repvit_mi` from `timm`. temporarily unsupported. |
    | evolla | ✅ | ✅ | ✅ | ✅ |  [#1440](https://github.com/mindspore-lab/mindone/pull/1440) |
    | florence2 | ✅ | ✅ | ✅ | ✅ | [#1453](https://github.com/mindspore-lab/mindone/pull/1453)  |
    | gemma3n | ✖️ | ✖️ | ✖️ | ✖️ | use `MobileNetV5` from `timm`. temporarily unsupported. |
    | glm4v | ✅ | ✅ | ✅ | ✅ | [#1109](https://github.com/mindspore-lab/mindone/pull/1109). explore detailed usage in [examples](../../examples/transformers/glm4v/README.md) |
    | glm4v_moe | ✅ | ✅ | ✅ | ✖️ | >100b. not validated due to the large size.[#1477](https://github.com/mindspore-lab/mindone/pull/1447) |
    | internvl | ✅ | ✅ | ✅ | ✅ | [#1463](https://github.com/mindspore-lab/mindone/pull/1463)  |
    | janus | ✅ | ✅ | ✅ | ✅ | [#1463](https://github.com/mindspore-lab/mindone/pull/1463)   |
    | kosmos2_5 | ✅ | ✅ | ✅ | ✅ | [#1456](https://github.com/mindspore-lab/mindone/pull/1456)  |
    | lfm2_vl | ✅ | ✅ | ✅ | ✅ | [#1456](https://github.com/mindspore-lab/mindone/pull/1456)   |
    | llama4 | ✅ | ✅ | ✅ | ✅ | specific moe layers are adapted to zero-3 sharding. [#1470](https://github.com/mindspore-lab/mindone/pull/1470). |
    | metaclip_2 | ✅ | ✅ | ✅ | ✅ |  [#1456](https://github.com/mindspore-lab/mindone/pull/1456)  |
    | mm_grounding_dino | ✅ | ✅ | ✅ | ✅ | use fp32 for model inference. [#1486](https://github.com/mindspore-lab/mindone/pull/1486) |
    | ovis2 | ✅ | ✅ | ✅ | ✅ | [#1454](https://github.com/mindspore-lab/mindone/pull/1454)  |
    | perception_lm | ✖️ | ✖️ | ✖️ | ✖️ | use `eva` from `timm` for vision model. temporarily unsupported. |
    | phi4_multimodal | ✅ | ✅ | ✅ | ✖️ | [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) requires transformers v4.48.2. temporarily unsupported. [#1468](https://github.com/mindspore-lab/mindone/pull/1468) |
    | qwen2_5_omni | ✅ | ✅ | ✅ | ✅ | also support lora fine-tune, see [examples](https://github.com/mindspore-lab/mindone/tree/master/examples/transformers/qwen2_5_omni) |
    | qwen3_omni_moe | ✅ | ✅ | ✅ | ✅ | see [#1411](https://github.com/mindspore-lab/mindone/pull/1411) for detailed usage. |
    | qwen3_vl | ✅ | ✅ | ✅ | ✅ | refer to examples/transformers/qwen3_vl for detailed usage. [#1310](https://github.com/mindspore-lab/mindone/pull/1310) |
    | qwen3_vl_moe | ✅ | ✅ | ✅ | ✅ | specific moe layers are adapted to zero-3 sharding. refer to examples/transformers/qwen3_vl for detialed usage. [#1310](https://github.com/mindspore-lab/mindone/pull/1310) |
    | smollm3 | ✅ | ✅ | ✅ | ✅ |  [#1391](https://github.com/mindspore-lab/mindone/pull/1391)  |
    | voxtral | ✅ | ✅ | ✅ | ✅ |  [#1456](https://github.com/mindspore-lab/mindone/pull/1456)   |


* **Time series models**

    | model | fp32 | fp16 | bf16 | inference | notes |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | timesfm | ✅ | ✖️ | ✅ | ✅ | fp16 infernece has `nan` ouputs in torch or mindspore. [#1403](https://github.com/mindspore-lab/mindone/pull/1403)  |

* **Audio / Video models**

    | model | fp32 | fp16 | bf16 | inference | notes |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | csm | ✅ | ✅ | ✅ | ✅ | [#1399](https://github.com/mindspore-lab/mindone/pull/1399) |
    | dia | ✅ | ✅ | ✅ | ✅ | [#1404](https://github.com/mindspore-lab/mindone/pull/1404) |
    | granite_speech | ✅ | ✅ | ✅ | ✅ | [#1406](https://github.com/mindspore-lab/mindone/pull/1406) |
    | kyutai_speech_to_text | ✅ | ✅ | ✅ | ✅ | [#1407](https://github.com/mindspore-lab/mindone/pull/1407) |
    | parakeet | ✅ | ✅ | ✅ | ✅ | [#1451](https://github.com/mindspore-lab/mindone/pull/1451) |
    | xcodec | ✅ | ✅ | ✅ | ✅ | [#1452](https://github.com/mindspore-lab/mindone/pull/1452) |
    | sam2_video | ✅ | ✅ | ✅ | ✅ | [#1434](https://github.com/mindspore-lab/mindone/pull/1434) |
    | vjepa2 | ✅ | ✅ | ✅ | ✅ | [#1125](https://github.com/mindspore-lab/mindone/pull/1125) |



## [previous version]

In the previous version (aligned with 🤗 Transformers v4.50.0), 240+ models were added. Some scripts have been upgraded to ensure that all existing model interfaces pass the fast unit tests. We do not guarantee that all model scripts are fully consistent with v4.57.1.

Fast UT validates pruned models that match the 🤗 Transformers' precision. For real‑weight inference, please switch back to mindone v0.4.0 for attempts. Community upgrades are very welcome.


| model | fp32 | fp16 | bf16 |
| --- | --- | --- | --- |
| albert | ✅ | ✅ | ✖️ |
| align | ✅ | ✅ | ✅ |
| altclip | ✅ | ✅ | ✅ |
| aria | ✅ | ✅ | ✅ |
| audio_spectrogram_transformer | ✅ | ✅ | ✅ |
| aya_vision | ✅ | ✅ | ✅ |
| bamba | ✅ | ✅ | ✅ |
| bark | ✅ | ✅ | ✅ |
| bart | ✅ | ✅ | ✅ |
| beit | ✅ | ✅ | ✅ |
| bert | ✅ | ✅ | ✖️ |
| bert_generation | ✅ | ✅ | ✅ |
| big_bird | ✅ | ✅ | ✅ |
| bigbird_pegasus | ✅ | ✅ | ✅ |
| biogpt | ✅ | ✅ | ✅ |
| bit | ✅ | ✅ | ✖️ |
| blenderbot | ✅ | ✅ | ✅ |
| blenderbot_small | ✅ | ✅ | ✅ |
| blip | ✅ | ✖️ | ✅ |
| blip_2 | ✅ | ✖️ | ✅ |
| bloom | ✅ | ✅ | ✅ |
| bridgetower | ✅ | ✅ | ✅ |
| bros | ✅ | ✅ | ✅ |
| camembert | ✅ | ✅ | ✅ |
| canine | ✅ | ✅ | ✅ |
| chameleon | ✅ | ✅ | ✅ |
| chinese_clip | ✅ | ✅ | ✅ |
| clip | ✅ | ✅ | ✅ |
| clipseg | ✅ | ✅ | ✅ |
| clvp | ✅ | ✅ | ✅ |
| codegen | ✅ | ✅ | ✅ |
| cohere | ✅ | ✅ | ✅ |
| cohere2 | ✅ | ✅ | ✅ |
| colpali | ✅ | ✅ | ✅ |
| convbert | ✅ | ✅ | ✅ |
| convnext | ✅ | ✅ | ✅ |
| convnextv2 | ✅ | ✅ | ✅ |
| ctrl | ✅ | ✖️ | ✖️ |
| cvt | ✅ | ✅ | ✅ |
| dac | ✅ | ✅ | ✅ |
| dbrx | ✅ | ✅ | ✅ |
| deberta | ✅ | ✅ | ✖️ |
| deberta_v2 | ✅ | ✅ | ✖️ |
| deit | ✅ | ✅ | ✅ |
| depth_anything | ✅ | ✖️ | ✅ |
| depth_pro | ✅ | ✅ | ✅ |
| diffllama | ✅ | ✅ | ✅ |
| dinov2 | ✅ | ✅ | ✅ |
| dinov2_with_registers | ✅ | ✅ | ✅ |
| distilbert | ✅ | ✅ | ✅ |
| dpr | ✅ | ✅ | ✅ |
| dpt | ✅ | ✖️ | ✖️ |
| electra | ✅ | ✅ | ✅ |
| emu3 | ✅ | ✅ | ✅ |
| encodec | ✅ | ✖️ | ✖️ |
| encoder_decoder | ✅ | ✅ | ✅ |
| esm | ✅ | ✅ | ✅ |
| falcon | ✅ | ✅ | ✅ |
| falcon_mamba | ✅ | ✅ | ✅ |
| fastspeech2_conformer | ✅ | ✖️ | ✖️ |
| flaubert | ✅ | ✅ | ✖️ |
| flava | ✅ | ✅ | ✅ |
| fnet | ✅ | ✅ | ✅ |
| focalnet | ✅ | ✅ | ✅ |
| fsmt | ✅ | ✅ | ✅ |
| funnel | ✅ | ✖️ | ✖️ |
| fuyu | ✅ | ✅ | ✅ |
| gemma | ✅ | ✅ | ✅ |
| gemma2 | ✅ | ✅ | ✅ |
| gemma3 | ✅ | ✅ | ✅ |
| git | ✅ | ✅ | ✅ |
| glpn | ✅ | ✖️ | ✖️ |
| got_ocr2 | ✅ | ✅ | ✅ |
| gpt2 | ✅ | ✅ | ✅ |
| gpt_bigcode | ✅ | ✅ | ✅ |
| gpt_neo | ✅ | ✅ | ✅ |
| gpt_neox | ✅ | ✅ | ✅ |
| gpt_neox_japanese | ✅ | ✅ | ✅ |
| gptj | ✅ | ✅ | ✅ |
| granite | ✅ | ✅ | ✅ |
| granitemoe | ✅ | ✅ | ✅ |
| granitemoeshared | ✅ | ✅ | ✅ |
| grounding_dino | ✅ | ✅ | ✅ |
| groupvit | ✅ | ✅ | ✅ |
| helium | ✅ | ✅ | ✅ |
| hiera | ✅ | ✖️ | ✖️ |
| hubert | ✅ | ✅ | ✅ |
| ibert | ✅ | ✅ | ✅ |
| idefics | ✅ | ✅ | ✅ |
| idefics2 | ✅ | ✅ | ✅ |
| idefics3 | ✅ | ✅ | ✅ |
| ijepa | ✅ | ✅ | ✅ |
| imagegpt | ✅ | ✅ | ✅ |
| instructblip | ✅ | ✅ | ✅ |
| instructblipvideo | ✅ | ✅ | ✅ |
| jamba | ✅ | ✅ | ✅ |
| jetmoe | ✅ | ✅ | ✅ |
| kosmos2 | ✅ | ✅ | ✅ |
| layoutlm | ✅ | ✅ | ✅ |
| layoutlmv3 | ✅ | ✅ | ✅ |
| led | ✅ | ✅ | ✅ |
| levit | ✅ | ✖️ | ✅ |
| lilt | ✅ | ✅ | ✅ |
| llama | ✅ | ✅ | ✅ |
| llava_next | ✅ | ✅ | ✅ |
| llava_next_video | ✅ | ✅ | ✅ |
| llava_onevision | ✅ | ✅ | ✅ |
| longformer | ✅ | ✅ | ✅ |
| longt5 | ✅ | ✖️ | ✅ |
| luke | ✅ | ✅ | ✅ |
| m2m_100 | ✅ | ✅ | ✅ |
| mamba | ✅ | ✅ | ✅ |
| mamba2 | ✅ | ✅ | ✅ |
| marian | ✅ | ✅ | ✅ |
| markuplm | ✅ | ✅ | ✅ |
| mask2former | ✅ | ✅ | ✅ |
| maskformer | ✅ | ✅ | ✅ |
| mbart | ✅ | ✅ | ✅ |
| mgp_str | ✅ | ✅ | ✅ |
| mimi | ✅ | ✅ | ✖️ |
| mistral | ✅ | ✅ | ✅ |
| mistral3 | ✅ | ✖️ | ✖️ |
| mixtral | ✅ | ✅ | ✅ |
| mllama | ✅ | ✅ | ✅ |
| mobilebert | ✅ | ✅ | ✖️ |
| mobilenet_v1 | ✅ | ✅ | ✖️ |
| mobilenet_v2 | ✅ | ✅ | ✖️ |
| mobilevit | ✅ | ✖️ | ✅ |
| mobilevitv2 | ✅ | ✖️ | ✅ |
| modernbert | ✅ | ✅ | ✖️ |
| moonshine | ✅ | ✅ | ✅ |
| moshi | ✅ | ✅ | ✅ |
| mpnet | ✅ | ✅ | ✅ |
| mpt | ✅ | ✅ | ✅ |
| mra | ✅ | ✖️ | ✖️ |
| mt5 | ✅ | ✅ | ✅ |
| musicgen | ✅ | ✅ | ✅ |
| musicgen_melody | ✅ | ✅ | ✅ |
| mvp | ✅ | ✅ | ✅ |
| nemotron | ✅ | ✅ | ✅ |
| nllb_moe | ✅ | ✅ | ✅ |
| nystromformer | ✅ | ✅ | ✖️ |
| olmo | ✅ | ✅ | ✅ |
| olmoe | ✅ | ✅ | ✅ |
| oneformer | ✅ | ✅ | ✅ |
| opt | ✅ | ✅ | ✅ |
| owlv2 | ✅ | ✅ | ✅ |
| owlvit | ✅ | ✅ | ✅ |
| paligemma | ✅ | ✅ | ✅ |
| pegasus | ✅ | ✅ | ✅ |
| pegasus_x | ✅ | ✅ | ✅ |
| perceiver | ✅ | ✅ | ✅ |
| persimmon | ✅ | ✅ | ✅ |
| phi | ✅ | ✅ | ✅ |
| phi3 | ✅ | ✅ | ✅ |
| phimoe | ✅ | ✅ | ✅ |
| pix2struct | ✅ | ✖️ | ✅ |
| pixtral | ✅ | ✅ | ✅ |
| plbart | ✅ | ✅ | ✅ |
| poolformer | ✅ | ✅ | ✅ |
| pop2piano | ✅ | ✅ | ✅ |
| prompt_depth_anything | ✅ | ✅ | ✅ |
| prophetnet | ✅ | ✅ | ✅ |
| pvt | ✅ | ✅ | ✅ |
| pvt_v2 | ✅ | ✅ | ✅ |
| qwen2 | ✅ | ✅ | ✅ |
| qwen2_5_vl | ✅ | ✅ | ✅ |
| qwen2_audio | ✅ | ✅ | ✅ |
| qwen2_moe | ✅ | ✅ | ✅ |
| qwen2_vl | ✅ | ✅ | ✅ |
| rag | ✅ | ✅ | ✅ |
| recurrent_gemma | ✅ | ✅ | ✅ |
| reformer | ✅ | ✅ | ✅ |
| regnet | ✅ | ✅ | ✅ |
| rembert | ✅ | ✅ | ✖️ |
| resnet | ✅ | ✅ | ✅ |
| roberta | ✅ | ✅ | ✅ |
| roberta_prelayernorm | ✅ | ✅ | ✅ |
| roc_bert | ✅ | ✅ | ✅ |
| roformer | ✅ | ✅ | ✅ |
| rwkv | ✅ | ✅ | ✅ |
| sam | ✅ | ✅ | ✅ |
| seamless_m4t | ✅ | ✅ | ✅ |
| seamless_m4t_v2 | ✅ | ✅ | ✅ |
| segformer | ✅ | ✅ | ✅ |
| seggpt | ✅ | ✅ | ✅ |
| sew | ✅ | ✅ | ✖️ |
| sew_d | ✅ | ✅ | ✖️ |
| shieldgemma2 | ✅ | ✅ | ✅ |
| siglip | ✅ | ✅ | ✅ |
| smolvlm | ✅ | ✖️ | ✖️ |
| speech_encoder_decoder | ✅ | ✅ | ✅ |
| speech_to_text | ✅ | ✅ | ✅ |
| splinter | ✅ | ✅ | ✅ |
| squeezebert | ✅ | ✅ | ✅ |
| stablelm | ✅ | ✅ | ✅ |
| starcoder2 | ✅ | ✅ | ✅ |
| swiftformer | ✅ | ✅ | ✅ |
| swin | ✅ | ✅ | ✅ |
| swin2sr | ✅ | ✅ | ✅ |
| swinv2 | ✅ | ✅ | ✅ |
| t5 | ✅ | ✅ | ✅ |
| tapas | ✅ | ✅ | ✅ |
| textnet | ✅ | ✅ | ✅ |
| timesformer | ✅ | ✅ | ✅ |
| trocr | ✅ | ✅ | ✅ |
| tvp | ✅ | ✅ | ✅ |
| udop | ✅ | ✅ | ✅ |
| unispeech | ✅ | ✅ | ✅ |
| unispeech_sat | ✅ | ✅ | ✅ |
| univnet | ✅ | ✖️ | ✅ |
| upernet | ✅ | ✅ | ✅ |
| video_llava | ✅ | ✅ | ✅ |
| videomae | ✅ | ✅ | ✅ |
| vilt | ✅ | ✅ | ✅ |
| vipllava | ✅ | ✅ | ✅ |
| vision_text_dual_encoder | ✅ | ✅ | ✅ |
| visual_bert | ✅ | ✅ | ✅ |
| vit | ✅ | ✅ | ✅ |
| vit_mae | ✅ | ✅ | ✅ |
| vit_msn | ✅ | ✅ | ✅ |
| vitdet | ✅ | ✅ | ✅ |
| vitmatte | ✅ | ✅ | ✖️ |
| vitpose | ✅ | ✅ | ✖️ |
| vitpose_backbone | ✅ | ✅ | ✅ |
| vivit | ✅ | ✅ | ✅ |
| wav2vec2 | ✅ | ✅ | ✅ |
| wav2vec2_bert | ✅ | ✅ | ✅ |
| wavlm | ✅ | ✖️ | ✖️ |
| x_clip | ✅ | ✅ | ✅ |
| xglm | ✅ | ✖️ | ✖️ |
| xlm | ✅ | ✅ | ✅ |
| xlm_roberta_xl | ✅ | ✅ | ✅ |
| xlnet | ✅ | ✅ | ✅ |
| xmod | ✅ | ✅ | ✅ |
| yolos | ✅ | ✅ | ✅ |
| yoso | ✅ | ✅ | ✅ |
| zamba | ✅ | ✅ | ✅ |
| zamba2 | ✅ | ✅ | ✅ |
| zoedepth | ✅ | ✅ | ✖️ |
