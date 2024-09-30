# Limitations

Due to differences in framework, some APIs & models will not be identical to [huggingface/diffusers](https://github.com/huggingface/diffusers) in the foreseeable future.

## APIs

### `xxx.from_pretrained`

- `torch_dtype` is renamed to `mindspore_dtype`
- `device_map`, `max_memory`, `offload_folder`, `offload_state_dict`, `low_cpu_mem_usage` will not be supported.

### `BaseOutput`

- Default value of `return_dict` is changed to `False`, for `GRAPH_MODE` does not allow to construct an instance of it.

### Output of `AutoencoderKL.encode`

Unlike the output `posterior = DiagonalGaussianDistribution(latent)`, which can do sampling by `posterior.sample()`.
We can only output the `latent` and then do sampling through `AutoencoderKL.diag_gauss_dist.sample(latent)`.

### `self.config` in `construct()`

For many models, parameters used in initialization will be registered in `self.config`. They are often accessed during the `construct` like using `if self.config.xxx == xxx` to determine execution paths in origin 🤗diffusers. However getting attributes like this is not supported by static graph syntax of MindSpore. Two feasible replacement options are

- set new attributes in initialization for `self` like `self.xxx = self.config.xxx`, then use `self.xxx` in `construct` instead.
- use `self.config["xxx"]` as `self.config` is an `OrderedDict` and getting items like this is supported in static graph mode.

When `self.config.xxx` changed, we change `self.xxx` and `self.config["xxx"]` both.

## Models

The table below represents the current support in mindone/diffusers for each of those modules, whether they have support in Pynative fp16 mode, Graph fp16 mode, Pynative fp32 mode or Graph fp32 mode.

|              Names               | Pynative FP16 | Pynative FP32 | Graph FP16 | Graph FP32 |                       Description                       |  
|:--------------------------------:|:-------------:|:-------------:|:----------:|:----------:|:-------------------------------------------------------:|  
|        StableCascadeUNet         |       ❌       |       ✅       |     ❌      |     ✅      |  huggingface/diffusers output NaN when using float16.   |
|            nn.Conv3d             |       ✅       |       ❌       |     ✅      |     ❌      |             FP32 is not supported on Ascend             |
|        TemporalConvLayer         |       ✅       |       ❌       |     ✅      |     ❌      |                   contains nn.Conv3d                    |
|       TemporalResnetBlock        |       ✅       |       ❌       |     ✅      |     ❌      |                   contains nn.Conv3d                    |
|      SpatioTemporalResBlock      |       ✅       |       ❌       |     ✅      |     ❌      |              contains TemporalResnetBlock               |
|     UNetMidBlock3DCrossAttn      |       ✅       |       ❌       |     ✅      |     ❌      |               contains TemporalConvLayer                |
|       CrossAttnDownBlock3D       |       ✅       |       ❌       |     ✅      |     ❌      |               contains TemporalConvLayer                |
|           DownBlock3D            |       ✅       |       ❌       |     ✅      |     ❌      |               contains TemporalConvLayer                |
|        CrossAttnUpBlock3D        |       ✅       |       ❌       |     ✅      |     ❌      |               contains TemporalConvLayer                |
|            UpBlock3D             |       ✅       |       ❌       |     ✅      |     ❌      |               contains TemporalConvLayer                |
|     MidBlockTemporalDecoder      |       ✅       |       ❌       |     ✅      |     ❌      |             contains SpatioTemporalResBlock             |
|      UpBlockTemporalDecoder      |       ✅       |       ❌       |     ✅      |     ❌      |             contains SpatioTemporalResBlock             |
|    UNetMidBlockSpatioTemporal    |       ✅       |       ❌       |     ✅      |     ❌      |             contains SpatioTemporalResBlock             |
|     DownBlockSpatioTemporal      |       ✅       |       ❌       |     ✅      |     ❌      |             contains SpatioTemporalResBlock             |
| CrossAttnDownBlockSpatioTemporal |       ✅       |       ❌       |     ✅      |     ❌      |             contains SpatioTemporalResBlock             |
|      UpBlockSpatioTemporal       |       ✅       |       ❌       |     ✅      |     ❌      |             contains SpatioTemporalResBlock             |
|  CrossAttnUpBlockSpatioTemporal  |       ✅       |       ❌       |     ✅      |     ❌      |             contains SpatioTemporalResBlock             |
|         TemporalDecoder          |       ✅       |       ❌       |     ✅      |     ❌      |    contains nn.Conv3d, MidBlockTemporalDecoder etc.     |
|       UNet3DConditionModel       |       ✅       |       ❌       |     ✅      |     ❌      |          contains UNetMidBlock3DCrossAttn etc.          |
|           I2VGenXLUNet           |       ✅       |       ❌       |     ✅      |     ❌      |          contains UNetMidBlock3DCrossAttn etc.          |
|   AutoencoderKLTemporalDecoder   |       ✅       |       ❌       |     ✅      |     ❌      |          contains MidBlockTemporalDecoder etc.          |
| UNetSpatioTemporalConditionModel |       ✅       |       ❌       |     ✅      |     ❌      |        contains UNetMidBlockSpatioTemporal etc.         |
|          FirUpsample2D           |       ❌       |       ✅       |     ✅      |     ✅      | ops.Conv2D has poor precision in fp16 and PyNative mode |
|         FirDownsample2D          |       ❌       |       ✅       |     ✅      |     ✅      | ops.Conv2D has poor precision in fp16 and PyNative mode |
|        AttnSkipUpBlock2D         |       ❌       |       ✅       |     ✅      |     ✅      |                 contains FirUpsample2D                  |
|          SkipUpBlock2D           |       ❌       |       ✅       |     ✅      |     ✅      |                 contains FirUpsample2D                  |
|       AttnSkipDownBlock2D        |       ❌       |       ✅       |     ✅      |     ✅      |                contains FirDownsample2D                 |
|         SkipDownBlock2D          |       ❌       |       ✅       |     ✅      |     ✅      |                contains FirDownsample2D                 |
|   ResnetBlock2D (kernel='fir')   |       ❌       |       ✅       |     ✅      |     ✅      | ops.Conv2D has poor precision in fp16 and PyNative mode |

## Pipelines
The table below represents the current support in mindone/diffusers for each of those pipelines in **MindSpore 2.3.0**,
whether they have support in Pynative fp16 mode, Graph fp16 mode, Pynative fp32 mode or Graph fp32 mode.

> precision issues of pipelines, the experiments in the table below default to upcasting GroupNorm to FP32 to avoid
> this issue.

|               **Pipelines**                | **Pynative FP16** | **Pynative FP32** | **Graph FP16** | **Graph FP32** |                                                                                **Description**                                                                                |
|:------------------------------------------:|:-----------------:|:-----------------:|:--------------:|:--------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|            AnimateDiffPipeline             |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|      AnimateDiffVideoToVideoPipeline       |         ✅         |         ❌         |       ✅        |       ✅        |                                                        In FP32 and Pynative mode, this pipeline will run out of memory                                                        |
|           BlipDiffusionPipeline            |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|          ConsistencyModelPipeline          |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|                DDIMPipeline                |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|                DDPMPipeline                |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|                DiTPipeline                 |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|              I2VGenXLPipeline              |         ✅         |         ✅         |       ✅        |       ✅        |                               ops.bmm and ops.softmax have precision issues under FP16, so we need to upcast them to FP32 to get a good result                                |
|             IFImg2ImgPipeline              |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|      IFImg2ImgSuperResolutionPipeline      |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|            IFInpaintingPipeline            |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|    IFInpaintingSuperResolutionPipeline     |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|                 IFPipeline                 |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|         IFSuperResolutionPipeline          |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|         Kandinsky3Img2ImgPipeline          |         ❌         |         ❌         |       ❌        |       ❌        | Kandinsky3 only provides FP16 weights; additionally, T5 has precision issues, so to achieve the desired results, you need to directly input prompt_embeds and attention_mask. |
|             Kandinsky3Pipeline             |         ❌         |         ❌         |       ❌        |       ❌        | Kandinsky3 only provides FP16 weights; additionally, T5 has precision issues, so to achieve the desired results, you need to directly input prompt_embeds and attention_mask. |
|          KandinskyImg2ImgPipeline          |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|          KandinskyInpaintPipeline          |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|             KandinskyPipeline              |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|   KandinskyV22ControlnetImg2ImgPipeline    |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|       KandinskyV22ControlnetPipeline       |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|        KandinskyV22Img2ImgPipeline         |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|        KandinskyV22InpaintPipeline         |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|            KandinskyV22Pipeline            |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|   LatentConsistencyModelImg2ImgPipeline    |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|       LatentConsistencyModelPipeline       |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|         LDMSuperResolutionPipeline         |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|           LDMTextToImagePipeline           |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|            PixArtAlphaPipeline             |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|            ShapEImg2ImgPipeline            |         ✅         |         ✅         |       ❌        |       ❌        |                                                               The syntax in Render only supports Pynative mode                                                                |
|               ShapEPipeline                |         ✅         |         ✅         |       ❌        |       ❌        |                                                               The syntax in Render only supports Pynative mode                                                                |
|           StableCascadePipeline            |         ❌         |         ✅         |       ❌        |       ✅        |                                                          This pipeline does not support FP16 due to precision issues                                                          |
|          StableDiffusion3Pipeline          |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|       StableDiffusionAdapterPipeline       |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|  StableDiffusionControlNetImg2ImgPipeline  |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|  StableDiffusionControlNetInpaintPipeline  |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|     StableDiffusionControlNetPipeline      |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|      StableDiffusionDepth2ImgPipeline      |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|      StableDiffusionDiffEditPipeline       |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|       StableDiffusionGLIGENPipeline        |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|   StableDiffusionGLIGENTextImagePipeline   |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|   StableDiffusionImageVariationPipeline    |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|       StableDiffusionImg2ImgPipeline       |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|       StableDiffusionInpaintPipeline       |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|   StableDiffusionInstructPix2PixPipeline   |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|    StableDiffusionLatentUpscalePipeline    |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|          StableDiffusionPipeline           |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|       StableDiffusionUpscalePipeline       |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|      StableDiffusionXLAdapterPipeline      |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
| StableDiffusionXLControlNetImg2ImgPipeline |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
| StableDiffusionXLControlNetInpaintPipeline |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|    StableDiffusionXLControlNetPipeline     |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|      StableDiffusionXLImg2ImgPipeline      |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|      StableDiffusionXLInpaintPipeline      |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|  StableDiffusionXLInstructPix2PixPipeline  |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|         StableDiffusionXLPipeline          |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|        StableVideoDiffusionPipeline        |         ✅         |         ❌         |       ✅        |       ❌        |       This pipeline will run out of memory under FP32; ops.bmm and ops.softmax have precision issues under FP16, so we need to upcast them to FP32 to get a good result       |
|        UnCLIPImageVariationPipeline        |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|               UnCLIPPipeline               |         ✅         |         ✅         |       ✅        |       ✅        |                                                                                                                                                                               |
|             WuerstchenPipeline             |         ✅         |         ✅         |       ✅        |       ✅        |                                    GlobalResponseNorm has precision issue under FP16, so we need to upcast it to FP32 to get a good result                                    |
