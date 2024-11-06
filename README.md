# MindONE

This repository contains SoTA algorithms, models, and interesting projects in the area of multimodal understanding and content generation

ONE is short for "ONE for all"

## News

**Hello MindSpore** from **Stable Diffusion 3**!

<div>
<img src="https://github.com/townwish4git/mindone/assets/143256262/8c25ae9a-67b1-436f-abf6-eca36738cd17" alt="sd3" width="512" height="512">
</div>

- [mindone/diffusers](mindone/diffusers) now supports [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium). Give it a try yourself!

    ```py
    import mindspore
    from mindone.diffusers import StableDiffusion3Pipeline

    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers",
        mindspore_dtype=mindspore.float16,
    )
    prompt = "A cat holding a sign that says 'Hello MindSpore'"
    image = pipe(prompt)[0][0]
    image.save("sd3.png")
    ```

### supported models under mindone/examples
| model  |  features  
| :---   |  :--  |
| [hpcai open sora](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_hpcai)      | support v1.0/1.1/1.2 large scale training with dp/sp/zero |
| [open sora plan](https://github.com/mindspore-lab/mindone/blob/master/examples/opensora_pku) | support v1.0/1.1/1.2 large scale training with dp/sp/zero |
| [stable diffusion](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_v2) | support sd 1.5/2.0/2.1, vanilla fine tune, lora, dreambooth, text inversion|
| [stable diffusion xl](https://github.com/mindspore-lab/mindone/blob/master/examples/stable_diffusion_xl)  |support sai style(stability AI) vanilla fine tune, lora, dreambooth |
| [dit](https://github.com/mindspore-lab/mindone/blob/master/examples/dit)     | support text to image fine tune |
| [hunyuan_dit](https://github.com/mindspore-lab/mindone/blob/master/examples/hunyuan_dit)     | support text to image fine tune |
| [pixart_sigma](https://github.com/mindspore-lab/mindone/blob/master/examples/pixart_sigma)     | suuport text to image fine tune at different aspect ratio |
| [latte](https://github.com/mindspore-lab/mindone/blob/master/examples/latte)     | support uncondition text to image fine tune |
| [animate diff](https://github.com/mindspore-lab/mindone/blob/master/examples/animatediff) | support motion module and lora training |
| [dynamicrafter](https://github.com/mindspore-lab/mindone/blob/master/examples/dynamicrafter)     | support image to video generation |

###  run hf diffusers on mindspore
mindone diffusers is under active development.

| component  |  features  
| :---   |  :--  
| [pipeline](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/pipelines) | support text2image,text2video,text2audio tasks 30+
| [models](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/models) | support audoencoder & transformers base models same as hf diffusers
| [schedulers](https://github.com/mindspore-lab/mindone/tree/master/mindone/diffusers/schedulers) | support ddpm & dpm solver 10+ schedulers same as hf diffusers
