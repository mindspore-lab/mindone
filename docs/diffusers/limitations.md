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
