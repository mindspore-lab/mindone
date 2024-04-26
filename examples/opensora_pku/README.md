# Open-Sora Plan
<!--
[[Project Page]](https://pku-yuangroup.github.io/Open-Sora-Plan/) [[中文主页]](https://pku-yuangroup.github.io/Open-Sora-Plan/blog_cn.html)
-->

[![slack badge](https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp)](https://discord.gg/vqGmpjkSaz)
[![WeChat badge](https://img.shields.io/badge/微信-加入-green?logo=wechat&amp)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/issues/53#issuecomment-1987226516)
[![Twitter](https://img.shields.io/badge/-Twitter@LinBin46984-black?logo=twitter&logoColor=1D9BF0)](https://x.com/LinBin46984/status/1763476690385424554?s=20) <br>
[![hf_space](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/LanguageBind/Open-Sora-Plan-v1.0.0)
[![Replicate demo and cloud API](https://replicate.com/camenduru/open-sora-plan-512x512/badge)](https://replicate.com/camenduru/open-sora-plan-512x512)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camenduru/Open-Sora-Plan-jupyter/blob/main/Open_Sora_Plan_jupyter.ipynb) <br>
[![License](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/LICENSE)
[![GitHub repo contributors](https://img.shields.io/github/contributors-anon/PKU-YuanGroup/Open-Sora-Plan?style=flat&label=Contributors)](https://github.com/PKU-YuanGroup/Open-Sora-Plan/graphs/contributors)


This is a MindSpore version of [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/main) from Peking University. We would like to express our gratitude to their contributions! :+1:

## Major Features

Currently, we are in line with **Open-Sora-Plan v1.0.0**. The **autoencoder** model is a `CausalVideoVAE` with a spatial-temporal compression to the videos by 4×8×8. The **transformer** model is a Latte-based text-to-video model. They also collect a high-quality video-caption dataset ([Open-Sora-Dataset](https://github.com/PKU-YuanGroup/Open-Sora-Dataset)) used to train the autoencoder and the tranformer model. See more details in the original [report-v1.0.0](https://github.com/PKU-YuanGroup/Open-Sora-Plan/blob/main/docs/Report-v1.0.0.md).
