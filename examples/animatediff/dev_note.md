MS-optimized AnimateDiff

Since torch AD relies heavily on diffusers and transformers, we will build a new code framework.
    1. SD modules are from LDM, base config: ../stable_diffusion_v2/configs/v1-inference.yaml

## Design Rules
1. Build framework. Then get what we want from those sources.
2. Double check input-output when using existing modules.
3. user-end api, make it as closer to original AD as possible

## TODO:
1. Code framework
    - setup code file structure
    - setup code key API structure
    - config method:
        Need to merge sd_v1_inferernce.yaml with ad_inference_v1.yaml

2. Net Modules
    - CLIP-vit l/14 text encoder
        - API: input config, return clip
        [ ] transfer from SD
        [ ] check: padding token!
    - VAE
    - UNet 3D
        - Basic Structure
            -conv_in = InflatedConv
                - in: b c f h w = (2 4 16 64 64)
                - proc: reshape to b*f c h w -> conv -> reshape back
                - out: b c f h w = ([2, 320, 16, 64, 64])
            -down_blocks
                -CrossAttnDownBlock3D x 3     # ftr_channel c=320, will be widen by channel_mult [1, 2, 4].  h and w will be reduced 1/2 at each block
                    -in: (b c f h w)
                    -out: (b c f h//2 w//2)
                    -ResBlock3D
                        - in: x: (b c f h w), temb: (b dt)
                        - out: (b c f h w)
                        - proc:
                            - GN: support two types
                                - normal: (b c f h w) -> GN -> (b*f c h w)  # mean on whole whole -> inference_v1.yaml, **use_inflated_groupnorm**=False
                                - inflated: (b c f h w) -> (b*f c h w) -> GN -> (b*f c h w) -> (b c f h w)  # mean on each frame, -> inference_v2.yaml, True
                            - SiLU
                            - InflatedConv
                                - reshape to bxf ..., conv, reshape back
                            - GN
                            - SiLU, dropout, InflatedConv
                    -SpatialTransformer3D
                        - input: x (b c f h w), context (b 77 768)
                        - x -> (b*f c h w) -> GN -> (b*f h w c) -> (b*f h*w c)
                        - context -> (b*f 77 768)
                        - BasicTransformer:
                            - CrossAttention:
                            - LayerNorm
                            - FF
                        - output:
                    -MotionModuel /TemporalTransformer3DModel
                        -Case 1 origin:
                            -in: (b c f h w)
                            -proc:
                                - rearrange_in: b c f h w -> b*f c h w      # our impl can skip this since the output of SpatialTrasformer is already (b*f c h w)
                                - norm-GroupNorm
                                - reshape -> (b*f h*w c)   # feature sequence
                                - proj_in = Linear(c, c) # projection in
                                - TemporalTransformerBlock x N  # (N=num_transformer_block=1)
                                    - in: (b*f h*w c)
                                    - norm-LayerNorm
                                    - Multi-head Self-Attention (named VersatileAttention) x N2   # (N2=len(attention_block_types)=2)
                                        - in: x: (b*f h*w c),
                                        - proc:
                                            - rearrange: x: b*f h*w c -> (b*h*w f c)  # sequence compute on f-time axis, e.g. (2*16, 64*64, 320) -> (8192, 16, 320)
                                            - temporal positional encoding: (bhw f c) -> (bhw f c)  PositionalEncoding on f sequence (max_seq_len: v1: 24, v2: 32)  # sinusoidal?
                                            - to_q, to_k, to_v: (bhw f c) -> (bhw f c),  c = attn_dim*num_heads, where num_heads=8, attn_dim=40, c=320 are previously defined.
                                            - reshape_heads_to_batch_dim: (bhw f attn_dim*num_heads) -> (b*h*w*num_heads f attn_dim) # (8192*8, 16, 40) = (65536, 16, 40)
                                            - attn compute(q, k, v)
                                                - in: (bhw*num_heads f attn_dim)  q: (65536, 16, 40), k: (65536, 16, 40), v (65536, 16, 40)
                                                - attn_map = qk' * scale                 #  (bhwn f f )   (65536, 16, 16)
                                                - attn_prob = softmax(attn_map, axis=-1) # (bhwn f f)   (65536, 16, 16)
                                                - h = bmm(attn_prob, v)             # (bhwn f attn_dim)  (65536, 16, 40)
                                                - reshape back: (bhwn f attn_dim) -> (bhw f attn_dim*num_heads) = (bhw f c)
                                                - out: (bhw f num_heads*attn_dim) = (bhw f c)     # (8192, 16, 320)
                                            - to_out: (bhw f c) -> (bhw f c), weights [320, 320], bias=True
                                            - dropout
                                            - rearrange back: (bhw f c) ->  (b*f h*w c)
                                        - out: (b*f h*w c)
                                    - LayerNorm: (b*f h*w c)
                                    - FF: (b*f h*w c)
                                        - geglu
                                            - linear, split
                                            - x * F.gelu(x)
                                        - dropout
                                        - linear
                                    - out: (b*f h*w c)
                                - proj_out = Linaer(c, c)
                                - reshape back:  (b*f h*w c) -> (b*f h w c) -> (b*f c h w)
                                - rearrange_out: (b*f c h w) -> (b c f h w)         # our impl can skip this, since the input format for next ResBlock is (b*f c h w)
                        -Case 2: my impl **PAY attention to FP32/FP16**
                            - Depth 1: MM init & MM construct, TemporalTransformer3DModel [Done]
                            - Depth 2: TemporalTransformerBlock [Doing] TODO: test
                                - VersatileAttention [Done]
                                    - Forward test => Forward compute is tested to be aligned on Jupyter Notebook
                                - FeedForward [Done, Test Doing]
                                    -GEGLU  [Done]
                                    - Forward test [Doing]
                                - LayerNorm:
                                -Forward, Done.
                            - Depth 3: Multi-head Self-Attention (VersatileAttention) [Done & Forward tested]
                    - Downsample3D
                        - in: (b c f h w)
                        - out: (b c f h//2 w//2)
                -DownBlock3D w/o Attentionx1
                    - ResnetBlock3D -> MotionModule
                    - ResnetBlock3D -> MotionModule
            -middle_block:
                - in: (2, 1280, 16, 8, 8) # feature channels is upsampled from 320 to 1280 ch_mult=(1,2,4,4). attention/feature resolution is downsampled from 64x64 to 8x8,
                - out: (2, 1280, 16, 8, 8)
                - ResBlock
                - SpatialTransformer
                - MotionModule
                - ResBlock
            -up_block:
                - out: b c f h w = ([2, 320, 16, 64, 64])
                - UpBlock3D
                    - ResnetBlock3D ->  MotionModule
                    - ResnetBlock3D ->  MotionModule
                    - ResnetBlock3D ->  MotionModule
                    - UpSampler
                - CrossAttnUpBlock3D x 3
                    - ResnetBlock3D -> SpatialTransformer ->  MotionModule
                    - ResnetBlock3D -> SpatialTransformer ->  MotionModule
                    - ResnetBlock3D -> SpatialTransformer ->  MotionModule
            -postprocess:
                - proc: GN - SiLU - Conv
                - out:  b c f h w = ([2, 4, 16, 64, 64])
        - lift 2d to pseudo 3d
            - Our logics:
                - input_blocks: nn.CellList
                    - noname-conv_in: nn.CellList(conv_nd)
                    - layers1: nn.CellList([ResBlock, SpatialTrans, MM])  or nn.CellList([ResBlock, MM]), equivlant of half of CrossAttnDownBlock / DownBlock in diffusers
                    - layers2, ...
                    - nonname-downsample: nn.CellList([Downsample(...)]

        - Inject MM to our pseudo 3d unet
            - init. [Done, need deubg]
            - forward, need video_length as input param! [Doing] => parse F in init, or construct?  TemporalTransformerBlock.construct need it.
            - log inject architecture to check. 21 MM in total. [Doing]
            - Checkpoint conversion script for MM
            - Load sd1.5 at first, then converted MM
            - test infer and overall forward error

3. Inference pipeline
    - Basic t2i
    -
