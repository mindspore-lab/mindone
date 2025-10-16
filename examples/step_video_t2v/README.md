# Step-Video-T2V on MindSpore

## 🔥 News!!

* **Feb 22, 2025**: 👋 We have reproduced the inference of the excellent work Step-Video-T2V, which was open-sourced by Step-Fun, on MindSpore.

## 📺 Video Demos

<table border="0" style="width: 100%; text-align: center; margin-top: 1px;">
  <tr>
    <td style="text-align: center;">
      <video src="https://github.com/user-attachments/assets/07dcec30-7f43-4751-adf2-9b09e93127e7" width="100%" controls autoplay loop muted></video>
      <p style="margin-top: 0.5rem; font-size: 0.8rem; color: #666;">gen by mindspore</p>
    </td>
  </tr>
</table>

## Table of Contents

1. [Introduction](#1-introduction)
2. [Model Download](#2-model-download)
3. [Model Usage](#3-model-usage)
4. [Acknowledgement](#4-ackownledgement)

## 1. Introduction

We have reproduced the excellent work of Step-Fun, **Step-Video-T2V**, on **[MindSpore](https://www.mindspore.cn/)**.

**[Step-Video-T2V](https://github.com/stepfun-ai/Step-Video-T2V)**, a state-of-the-art (SoTA) text-to-video pre-trained model with 30 billion parameters and the capability to generate videos up to 204 frames. To enhance efficiency, stepfun-ai propose a deep compression VAE for videos, achieving 16x16 spatial and 8x temporal compression ratios. Direct Preference Optimization (DPO) is applied in the final stage to further enhance the visual quality of the generated videos. Step-Video-T2V's performance is evaluated on a novel video generation benchmark, **Step-Video-T2V-Eval**, demonstrating its SoTA text-to-video quality compared to both open-source and commercial engines.

## 2. Weight Download

| Models   | 🤗Huggingface    |  🤖Modelscope |
|:-------:|:-------:|:-------:|
| **Step-Video-T2V** | [download](https://huggingface.co/stepfun-ai/stepvideo-t2v) | [download](https://www.modelscope.cn/models/stepfun-ai/stepvideo-t2v)
| **Step-Video-T2V-Turbo (Inference Step Distillation)** | [download](https://huggingface.co/stepfun-ai/stepvideo-t2v-turbo) | [download](https://www.modelscope.cn/models/stepfun-ai/stepvideo-t2v-turbo)

## 3. Model Usage

### 📜 3.1 Performance

The following table shows the requirements for running **Step-Video-T2V** model (batch size = 1, w/o cfg distillation) to generate videos:

|     Model    |  height/width/frame | infer steps | Peak NPU Memory | time cost w/ mindspore 2.5.0 | time cost w/ mindspore 2.6.0 | time cost w/ mindspore 2.7.0 |
|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
| **Step-Video-T2V**   |        768px768px204f      | 50 | 46.72 GB | ~73min | ~72min | ~73min |
| **Step-Video-T2V**   |        544px992px204f      | 50 | 45.83 GB | ~64min | ~63min | ~63min |
| **Step-Video-T2V**   |        544px992px136f      | 50 | 40.48 GB | ~36min | ~36min | ~35min |

* The above time cost does not include *initialization, weights loading and server time*.
* An **Ascend Atlas 800T A2** NPU with CANN support is required.
  * The model is tested on 6 NPUs. (Including 2 NPUs used to provide prompt encoding and VAE video decoding services.)
* Tested operating system: **EulerOS**

### ⬇️ 3.2 Dependencies and Installation

| mindspore  | ascend driver  |  firmware   |cann toolkit/kernel |
|:----------:|:--------------:|:-----------:|:------------------:|
|   **2.5.0**    |    24.1.RC2    | 7.3.0.1.231 |   8.0.0.beta1    |
|   **2.6.0**    |    24.1.RC2    | 7.3.0.1.231 |   8.1.RC1    |
|   **2.7.0**    |    24.1.RC2    | 7.3.0.1.231 |   8.2.RC1    |

To install other dependent packages:

```bash
git clone https://github.com/mindspore-lab/mindone.git

# install mindone
cd mindone
pip install -e .

# install requirements
cd examples/step_video_t2v
pip install -r requirements.txt
```

### 🔧 3.3. Prepare Weight Format

convert `.bin` weight (**hunyuan-clip**) format from `pytorch_model.bin` to `model.safetensors`

```shell
python convert.py --pt_filename where_bin_file --sf_filename where_safetensors_file --config_path where_{config.json}_file

# example as:
python convert.py --pt_filename /path_to/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/pytorch_model.bin --sf_filename /path_to/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/model.safetensors --config_path /path_to/stepfun-ai/stepvideo-t2v/hunyuan_clip/clip_text_encoder/config.json
```

### 📖 3.4 Inference Scripts (Multi-Cards Parallel Deployment)

- ⚠️ **First**, confirm the security risks from `pickle`, then manually decomment **two places** of code in file `api/call_remote_server.py` that involve security risks.

  ```python
  def get(self):
    # ==================== We comment the follow code, because security risks.  ======================
    # ===========           You need to manually decomment it before running.            =============
    # ========                             The second place.                                ==========
    # ================================================================================================
    #
    # import pickle
    # feature = pickle.loads(request.get_data())
    # feature["api"] = "caption"
    # feature = {k: v for k, v in feature.items() if v is not None}
    # embeddings = self.caption_pipeline.embedding(**feature)
    # response = pickle.dumps(embeddings)
    # return Response(response)
    #
    ### ------------------------------------------------------------------------------------------ ###

    raise NotImplementedError(
        "There are some security risks from pickle here. \n"
        "You need to confirm it and manually decomment the code above before running them."
    )
  ```

- 🚀 **Then, running**

  ```shell
  # We employed a decoupling strategy for the text encoder, VAE decoding,
  # and DiT to optimize NPU resource utilization by DiT.
  # As a result, a dedicated NPU is needed to handle the API services
  # for the text encoder's embeddings and VAE decoding.

  # (1) first, start vae/captioner server on single-card (Ascend Atlas 800T A2 machine)
  # !!! This command will return the URL for both the caption API and the VAE API. Please use the returned URL in the following command.
  model_dir=where_you_download_dir
  ASCEND_RT_VISIBLE_DEVICES=0 python api/call_remote_server.py --model_dir $model_dir --enable_vae True &
  ASCEND_RT_VISIBLE_DEVICES=1 python api/call_remote_server.py --model_dir $model_dir --enable_llm True &

  # !!! ⚠️ wait...a few minutes, vae/llm is loading...

  # (2) second, setting and replace the `url` from before command print
  parallel=4
  sp=2
  pp=2
  vae_url='127.0.0.1'
  caption_url='127.0.0.1'
  model_dir=where_you_download_dir

  # (3) finally, run parallel dit model on 4-cards (Ascend Atlas 800T A2 machines)
  ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 msrun --bind_core=True --worker_num=$parallel --local_worker_num=$parallel --master_port=9000 --log_dir=outputs/parallel_logs python -u \
  run_parallel.py --model_dir $model_dir --vae_url $vae_url --caption_url $caption_url  --ulysses_degree $sp --pp_degree $pp --prompt "一名宇航员在月球上发现一块石碑，上面印有“stepfun”字样，闪闪发光" --infer_steps 30  --cfg_scale 9.0 --time_shift 13.0 --num_frames 136 --height 544 --width 992
  ```

### 🚀 3.5 Best-of-Practice Inference settings

**Step-Video-T2V** exhibits robust performance in inference settings, consistently generating high-fidelity and dynamic videos. However, our experiments reveal that variations in inference hyperparameters can have a substantial effect on the trade-off between video fidelity and dynamics. To achieve optimal results, we recommend the following best practices for tuning inference parameters:

| models   | infer_steps   | cfg_scale  | time_shift | num_frames |
|:-------:|:-------:|:-------:|:-------:|:-------:|
| **Step-Video-T2V** | 30-50 | 9.0 |  13.0 | 204
| **Step-Video-T2V-Turbo (Inference Step Distillation)** | 10-15 | 5.0 | 17.0 | 204 |

## 4. Acknowledgement

This project uses code from [stepfun-ai/Step-Video-T2V](https://github.com/stepfun-ai/Step-Video-T2V), thanks to the **stepfun-ai** team for their contribution.
