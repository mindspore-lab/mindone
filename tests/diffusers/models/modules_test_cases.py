# This file defined a list containing all generalized test cases. Each test case is represented as a list or tuple following the structure:
#    [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs].
#
# Parameters:
#     name:
#         A string identifier for the test case, primarily for diagnostic purposes and not utilized during execution.
#     ms_module:
#         The module from 'mindone.diffusers' under test, e.g., 'mindone.diffusers.models.model0'.
#     pt_module:
#         The counterpart module from the original 'diffusers' library for accuracy benchmarking, matching ms_module in functionality.
#     init_args, init_kwargs:
#         Arguments for initializing the modules, positional and keyword respectively.
#     inputs_args, inputs_kwargs:
#         Arguments for model inputs, positional and keyword respectively. These are initially defined with numpy for compatibility and are converted
#         to PyTorch or MindSpore formats via the `.modeling_test_utils.generalized_parse_args` utility. If this utility's conversions do not suffice,
#         a specific unit test should be developed rather than relying on generic test cases.

import numpy as np

# layers
NORMALIZATION_CASES = [
    [
        "AdaLayerNormZero",
        "diffusers.models.normalization.AdaLayerNormZero",
        "mindone.diffusers.models.normalization.AdaLayerNormZero",
        (16, 8),
        {},
        (
            np.random.randn(3, 4, 16).astype(np.float32),
            np.random.randint(0, 8, size=(3,)),
            np.random.randint(0, 8, size=(3,)),
        ),
        {},
    ],
    [
        "AdaGroupNorm",
        "diffusers.models.normalization.AdaGroupNorm",
        "mindone.diffusers.models.normalization.AdaGroupNorm",
        (16, 12, 2),
        {},
        (
            np.random.randn(3, 12, 4, 4).astype(np.float32),
            np.random.randn(3, 16).astype(np.float32),
        ),
        {},
    ],
    [
        "AdaLayerNormContinuous",
        "diffusers.models.normalization.AdaLayerNormContinuous",
        "mindone.diffusers.models.normalization.AdaLayerNormContinuous",
        (16, 12),
        {},
        (
            np.random.randn(3, 4, 16).astype(np.float32),
            np.random.randn(3, 12).astype(np.float32),
        ),
        {},
    ],
    [
        "LayerNorm",
        "torch.nn.LayerNorm",
        "mindone.diffusers.models.normalization.LayerNorm",
        (16, 1e-5, False),
        {},
        (np.random.randn(3, 4, 16).astype(np.float32),),
        {},
    ],
    [
        "GroupNorm",
        "torch.nn.GroupNorm",
        "mindone.diffusers.models.normalization.GroupNorm",
        (4, 16),
        {},
        (np.random.randn(3, 16, 4, 4).astype(np.float32),),
        {},
    ],
    [
        "GlobalResponseNorm",
        "diffusers.models.normalization.GlobalResponseNorm",
        "mindone.diffusers.models.normalization.GlobalResponseNorm",
        (16,),
        {},
        (np.random.randn(3, 4, 4, 16).astype(np.float32),),
        {},
    ],
]


EMBEDDINGS_CASES = [
    [
        "PatchEmbed",
        "diffusers.models.embeddings.PatchEmbed",
        "mindone.diffusers.models.embeddings.PatchEmbed",
        (),
        {},
        (np.random.randn(3, 3, 224, 224).astype(np.float32),),
        {},
    ],
    [
        "Timesteps",
        "diffusers.models.embeddings.Timesteps",
        "mindone.diffusers.models.embeddings.Timesteps",
        (128, False, 1),
        {},
        (np.random.randint(0, 100, size=(3,)).astype(np.int32),),
        {},
    ],
    [
        "TimestepEmbedding",
        "diffusers.models.embeddings.TimestepEmbedding",
        "mindone.diffusers.models.embeddings.TimestepEmbedding",
        (16, 128),
        {},
        (np.random.randn(3, 16).astype(np.float32),),
        {},
    ],
    [
        "GaussianFourierProjection",
        "diffusers.models.embeddings.GaussianFourierProjection",
        "mindone.diffusers.models.embeddings.GaussianFourierProjection",
        (),
        {},
        (np.random.randint(1, 20, size=(3,)).astype(np.float32),),
        {},
    ],
    [
        "SinusoidalPositionalEmbedding",
        "diffusers.models.embeddings.SinusoidalPositionalEmbedding",
        "mindone.diffusers.models.embeddings.SinusoidalPositionalEmbedding",
        (128, 32),
        {},
        (np.random.randn(3, 16, 128).astype(np.float32),),
        {},
    ],
    [
        "ImagePositionalEmbeddings",
        "diffusers.models.embeddings.ImagePositionalEmbeddings",
        "mindone.diffusers.models.embeddings.ImagePositionalEmbeddings",
        (192, 16, 12, 128),
        {},
        (np.random.randint(0, 192, size=(3, 16)).astype(np.int32),),
        {},
    ],
    [
        "LabelEmbedding",
        "diffusers.models.embeddings.LabelEmbedding",
        "mindone.diffusers.models.embeddings.LabelEmbedding",
        (100, 128, 0.1),
        {},
        (np.random.randint(0, 100, size=(3, 16)).astype(np.int32),),
        {},
    ],
    [
        "TextImageProjection",
        "diffusers.models.embeddings.TextImageProjection",
        "mindone.diffusers.models.embeddings.TextImageProjection",
        (),
        {},
        (
            np.random.randn(3, 77, 1024).astype(np.float32),
            np.random.randn(3, 768).astype(np.float32),
        ),
        {},
    ],
    [
        "ImageProjection",
        "diffusers.models.embeddings.ImageProjection",
        "mindone.diffusers.models.embeddings.ImageProjection",
        (),
        {},
        (np.random.randn(3, 768).astype(np.float32),),
        {},
    ],
    [
        "CombinedTimestepLabelEmbeddings",
        "diffusers.models.embeddings.CombinedTimestepLabelEmbeddings",
        "mindone.diffusers.models.embeddings.CombinedTimestepLabelEmbeddings",
        (100, 128),
        {},
        (np.random.randint(0, 100, size=(3,)).astype(np.int32), np.random.randint(0, 100, size=(3,)).astype(np.int32)),
        {},
    ],
    [
        "TextTimeEmbedding",
        "diffusers.models.embeddings.TextTimeEmbedding",
        "mindone.diffusers.models.embeddings.TextTimeEmbedding",
        (32, 128, 8),
        {},
        (np.random.randn(3, 4, 32).astype(np.float32),),
        {},
    ],
    [
        "TextImageTimeEmbedding",
        "diffusers.models.embeddings.TextImageTimeEmbedding",
        "mindone.diffusers.models.embeddings.TextImageTimeEmbedding",
        (32, 24, 64),
        {},
        (
            np.random.randn(3, 16, 32).astype(np.float32),
            np.random.randn(3, 16, 24).astype(np.float32),
        ),
        {},
    ],
    [
        "ImageTimeEmbedding",
        "diffusers.models.embeddings.ImageTimeEmbedding",
        "mindone.diffusers.models.embeddings.ImageTimeEmbedding",
        (32, 128),
        {},
        (np.random.randn(3, 16, 32).astype(np.float32),),
        {},
    ],
    [
        "ImageHintTimeEmbedding",
        "diffusers.models.embeddings.ImageHintTimeEmbedding",
        "mindone.diffusers.models.embeddings.ImageHintTimeEmbedding",
        (32, 128),
        {},
        (
            np.random.randn(3, 16, 32).astype(np.float32),
            np.random.randn(3, 3, 128, 128).astype(np.float32),
        ),
        {},
    ],
]


UPSAMPLE2D_CASES = [
    [
        "Upsample2D_default",
        "diffusers.models.upsampling.Upsample2D",
        "mindone.diffusers.models.upsampling.Upsample2D",
        (),
        dict(channels=32, use_conv=False),
        (np.random.randn(1, 32, 32, 32).astype(np.float32),),
        {},
    ],
    [
        "Upsample2D_with_conv",
        "diffusers.models.upsampling.Upsample2D",
        "mindone.diffusers.models.upsampling.Upsample2D",
        (),
        dict(channels=32, use_conv=True),
        (np.random.randn(1, 32, 32, 32).astype(np.float32),),
        {},
    ],
    [
        "Upsample2D_with_conv_out_dim",
        "diffusers.models.upsampling.Upsample2D",
        "mindone.diffusers.models.upsampling.Upsample2D",
        (),
        dict(channels=32, use_conv=True, out_channels=64),
        (np.random.randn(1, 32, 32, 32).astype(np.float32),),
        {},
    ],
    [
        "Upsample2D_with_transpose",
        "diffusers.models.upsampling.Upsample2D",
        "mindone.diffusers.models.upsampling.Upsample2D",
        (),
        dict(channels=32, use_conv=False, use_conv_transpose=True),
        (np.random.randn(1, 32, 32, 32).astype(np.float32),),
        {},
    ],
]


DOWNSAMPLE2D_CASES = [
    [
        "Downsample2D_default",
        "diffusers.models.downsampling.Downsample2D",
        "mindone.diffusers.models.downsampling.Downsample2D",
        (),
        dict(channels=32, use_conv=False),
        (np.random.randn(1, 32, 64, 64).astype(np.float32),),
        {},
    ],
    [
        "Downsample2D_with_conv",
        "diffusers.models.downsampling.Downsample2D",
        "mindone.diffusers.models.downsampling.Downsample2D",
        (),
        dict(channels=32, use_conv=True),
        (np.random.randn(1, 32, 64, 64).astype(np.float32),),
        {},
    ],
    [
        "Downsample2D_with_conv_pad1",
        "diffusers.models.downsampling.Downsample2D",
        "mindone.diffusers.models.downsampling.Downsample2D",
        (),
        dict(channels=32, use_conv=True, padding=1),
        (np.random.randn(1, 32, 64, 64).astype(np.float32),),
        {},
    ],
    [
        "Downsample2D_with_conv_out_dim",
        "diffusers.models.downsampling.Downsample2D",
        "mindone.diffusers.models.downsampling.Downsample2D",
        (),
        dict(channels=32, use_conv=True, out_channels=16),
        (np.random.randn(1, 32, 64, 64).astype(np.float32),),
        {},
    ],
]


RESNET_CASES = [
    [
        "ResnetBlock2D_default",
        "diffusers.models.resnet.ResnetBlock2D",
        "mindone.diffusers.models.resnet.ResnetBlock2D",
        (),
        {"in_channels": 32, "temb_channels": 128},
        (np.random.randn(1, 32, 64, 64).astype(np.float32), np.random.randn(1, 128).astype(np.float32)),
        {},
    ],
    [
        "ResnetBlock2D_with_use_in_shortcut",
        "diffusers.models.resnet.ResnetBlock2D",
        "mindone.diffusers.models.resnet.ResnetBlock2D",
        (),
        {"in_channels": 32, "temb_channels": 128, "use_in_shortcut": True},
        (np.random.randn(1, 32, 64, 64).astype(np.float32), np.random.randn(1, 128).astype(np.float32)),
        {},
    ],
    [
        "ResnetBlock2D_up",
        "diffusers.models.resnet.ResnetBlock2D",
        "mindone.diffusers.models.resnet.ResnetBlock2D",
        (),
        {"in_channels": 32, "temb_channels": 128, "up": True},
        (np.random.randn(1, 32, 64, 64).astype(np.float32), np.random.randn(1, 128).astype(np.float32)),
        {},
    ],
    [
        "ResnetBlock2D_down",
        "diffusers.models.resnet.ResnetBlock2D",
        "mindone.diffusers.models.resnet.ResnetBlock2D",
        (),
        {"in_channels": 32, "temb_channels": 128, "down": True},
        (np.random.randn(1, 32, 64, 64).astype(np.float32), np.random.randn(1, 128).astype(np.float32)),
        {},
    ],
    [
        "ResnetBlock2D_down_with_kernel_fir",
        "diffusers.models.resnet.ResnetBlock2D",
        "mindone.diffusers.models.resnet.ResnetBlock2D",
        (),
        {"in_channels": 32, "temb_channels": 128, "kernel": "fir", "down": True},
        (np.random.randn(1, 32, 64, 64).astype(np.float32), np.random.randn(1, 128).astype(np.float32)),
        {},
    ],
    [
        "ResnetBlock2D_down_with_kernel_sde_vp",
        "diffusers.models.resnet.ResnetBlock2D",
        "mindone.diffusers.models.resnet.ResnetBlock2D",
        (),
        {"in_channels": 32, "temb_channels": 128, "kernel": "sde_vp", "down": True},
        (np.random.randn(1, 32, 64, 64).astype(np.float32), np.random.randn(1, 128).astype(np.float32)),
        {},
    ],
    [
        "ResnetBlock2D_up_with_kernel_fir",
        "diffusers.models.resnet.ResnetBlock2D",
        "mindone.diffusers.models.resnet.ResnetBlock2D",
        (),
        {"in_channels": 32, "temb_channels": 128, "kernel": "fir", "up": True},
        (np.random.randn(1, 32, 64, 64).astype(np.float32), np.random.randn(1, 128).astype(np.float32)),
        {},
    ],
    [
        "ResnetBlock2D_up_with_kernel_sde_vp",
        "diffusers.models.resnet.ResnetBlock2D",
        "mindone.diffusers.models.resnet.ResnetBlock2D",
        (),
        {"in_channels": 32, "temb_channels": 128, "kernel": "sde_vp", "up": True},
        (np.random.randn(1, 32, 64, 64).astype(np.float32), np.random.randn(1, 128).astype(np.float32)),
        {},
    ],
    [
        "TemporalConvLayer",
        "diffusers.models.resnet.TemporalConvLayer",
        "mindone.diffusers.models.resnet.TemporalConvLayer",
        (),
        dict(
            in_dim=32,
            out_dim=32,
            dropout=0.1,
            norm_num_groups=8,
        ),
        (np.random.randn(24, 32, 24, 24).astype(np.float32), 1),
        {},
    ],
]


T2I_ADAPTER_CASES = [
    [
        "T2IAdapter",
        "diffusers.models.adapter.T2IAdapter",
        "mindone.diffusers.models.adapter.T2IAdapter",
        (),
        dict(
            in_channels=3,
            channels=[32, 64, 128],
            num_res_blocks=2,
            downscale_factor=2,
            adapter_type="full_adapter",
        ),
        (),
        {
            "x": np.random.randn(4, 3, 32, 32).astype(np.float32),
        },
    ],
]


CONTROL_NET_CASES = [
    [
        "ControlNetModel",
        "diffusers.models.controlnet.ControlNetModel",
        "mindone.diffusers.models.controlnet.ControlNetModel",
        (),
        dict(
            block_out_channels=(4, 8),
            layers_per_block=2,
            in_channels=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            cross_attention_dim=32,
            conditioning_embedding_out_channels=(16, 32),
            norm_num_groups=1,
        ),
        (),
        {
            "sample": np.random.randn(2, 4, 32, 32).astype(np.float32),
            "timestep": np.array([10]).astype(np.int64),
            "encoder_hidden_states": np.random.randn(2, 77, 32).astype(np.float32),
            "controlnet_cond": np.random.randn(2, 3, 64, 64).astype(np.float32),
            "return_dict": False,
        },
    ],
]


LAYERS_CASES = (
    NORMALIZATION_CASES + EMBEDDINGS_CASES + UPSAMPLE2D_CASES + DOWNSAMPLE2D_CASES + RESNET_CASES + T2I_ADAPTER_CASES
)


# autoencoders
# VQModel: volatile in fp16(fyi: 2%-20% diff when torch.fp16 vs torch.fp32)
VQ_CASES = [
    [
        "VQModel",  # volatile with random init: 2%-20% diff when torch.float16 v.s. torch.float32
        "diffusers.models.vq_model.VQModel",
        "mindone.diffusers.models.vq_model.VQModel",
        (),
        {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "up_block_types": ["UpDecoderBlock2D", "UpDecoderBlock2D"],
            "latent_channels": 3,
        },
        (),
        {"sample": np.random.randn(4, 3, 32, 32).astype(np.float32), "return_dict": False},
    ],
]


VAE_CASES = [
    [
        "AutoencoderKL",
        "diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
        "mindone.diffusers.models.autoencoders.autoencoder_kl.AutoencoderKL",
        (),
        {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D"] * len([32, 64]),
            "up_block_types": ["UpDecoderBlock2D"] * len([32, 64]),
            "latent_channels": 4,
            "norm_num_groups": 32,
        },
        (),
        {
            "sample": np.random.randn(4, 3, 32, 32).astype(np.float32),
            "return_dict": False,
        },
    ],
    [
        "AsymmetricAutoencoderKL",
        "diffusers.models.autoencoders.autoencoder_asym_kl.AsymmetricAutoencoderKL",
        "mindone.diffusers.models.autoencoders.autoencoder_asym_kl.AsymmetricAutoencoderKL",
        (),
        {
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D"] * len([32, 64]),
            "down_block_out_channels": [32, 64],
            "layers_per_down_block": 1,
            "up_block_types": ["UpDecoderBlock2D"] * len([32, 64]),
            "up_block_out_channels": [32, 64],
            "layers_per_up_block": 1,
            "act_fn": "silu",
            "latent_channels": 4,
            "norm_num_groups": 32,
            "sample_size": 32,
            "scaling_factor": 0.18215,
        },
        (),
        {
            "sample": np.random.randn(4, 3, 32, 32).astype(np.float32),
            "mask": np.random.randn(4, 1, 32, 32).astype(np.float32),
            "return_dict": False,
        },
    ],
    [
        "AutoencoderTiny",
        "diffusers.models.autoencoders.autoencoder_tiny.AutoencoderTiny",
        "mindone.diffusers.models.autoencoders.autoencoder_tiny.AutoencoderTiny",
        (),
        {
            "in_channels": 3,
            "out_channels": 3,
            "encoder_block_out_channels": [32, 32],
            "decoder_block_out_channels": [32, 32],
            "num_encoder_blocks": [b // min([32, 32]) for b in [32, 32]],
            "num_decoder_blocks": [b // min([32, 32]) for b in reversed([32, 32])],
        },
        (),
        {
            "sample": np.random.randn(4, 3, 32, 32).astype(np.float32),
            "return_dict": False,
        },
    ],
    [
        "AutoencoderKLTemporalDecoder",
        "diffusers.models.autoencoders.autoencoder_kl_temporal_decoder.AutoencoderKLTemporalDecoder",
        "mindone.diffusers.models.autoencoders.autoencoder_kl_temporal_decoder.AutoencoderKLTemporalDecoder",
        (),
        {
            "block_out_channels": [32, 64],
            "in_channels": 3,
            "out_channels": 3,
            "down_block_types": ["DownEncoderBlock2D", "DownEncoderBlock2D"],
            "latent_channels": 4,
            "layers_per_block": 2,
        },
        (),
        {
            "sample": np.random.randn(3, 3, 32, 32).astype(np.float32),
            "num_frames": 3,
            "return_dict": False,
        },
    ],
]


AE_CASES = VQ_CASES + VAE_CASES


# transformers
TRANSFORMER2D_CASES = [
    [
        "SpatialTransformer2DModel_default",
        "diffusers.models.transformers.transformer_2d.Transformer2DModel",
        "mindone.diffusers.models.transformers.transformer_2d.Transformer2DModel",
        (),
        dict(
            in_channels=32,
            num_attention_heads=1,
            attention_head_dim=32,
            dropout=0.0,
            cross_attention_dim=None,
        ),
        (np.random.randn(1, 32, 64, 64).astype(np.float32),),
        dict(return_dict=False),
    ],
    [
        "SpatialTransformer2DModel_cross_attention_dim",
        "diffusers.models.transformers.transformer_2d.Transformer2DModel",
        "mindone.diffusers.models.transformers.transformer_2d.Transformer2DModel",
        (),
        dict(
            in_channels=64,
            num_attention_heads=2,
            attention_head_dim=32,
            dropout=0.0,
            cross_attention_dim=64,
        ),
        (np.random.randn(1, 64, 64, 64).astype(np.float32), np.random.randn(1, 4, 64).astype(np.float32)),
        dict(return_dict=False),
    ],
    [
        "SpatialTransformer2DModel_dropout",
        "diffusers.models.transformers.transformer_2d.Transformer2DModel",
        "mindone.diffusers.models.transformers.transformer_2d.Transformer2DModel",
        (),
        dict(
            in_channels=32,
            num_attention_heads=2,
            attention_head_dim=16,
            dropout=0.3,
            cross_attention_dim=None,
        ),
        (np.random.randn(1, 32, 64, 64).astype(np.float32),),
        dict(return_dict=False),
    ],
    [
        "SpatialTransformer2DModel_discrete",
        "diffusers.models.transformers.transformer_2d.Transformer2DModel",
        "mindone.diffusers.models.transformers.transformer_2d.Transformer2DModel",
        (),
        dict(
            num_attention_heads=1,
            attention_head_dim=32,
            num_vector_embeds=5,
            sample_size=16,
        ),
        (np.random.randint(0, 5, (1, 32)).astype(np.int64),),
        dict(return_dict=False),
    ],
]


PRIOR_TRANSFORMER_CASES = [
    [
        "PriorTransformer",
        "diffusers.models.transformers.prior_transformer.PriorTransformer",
        "mindone.diffusers.models.transformers.prior_transformer.PriorTransformer",
        (),
        {
            "num_attention_heads": 2,
            "attention_head_dim": 4,
            "num_layers": 2,
            "embedding_dim": 8,
            "num_embeddings": 7,
            "additional_embeddings": 4,
        },
        (),
        {
            "hidden_states": np.random.randn(4, 8).astype(np.float32),
            "timestep": 2,
            "proj_embedding": np.random.randn(4, 8).astype(np.float32),
            "encoder_hidden_states": np.random.randn(4, 7, 8).astype(np.float32),
            "return_dict": False,
        },
    ],
]


TRANSFORMERS_CASES = TRANSFORMER2D_CASES + PRIOR_TRANSFORMER_CASES


# unet
UNET1D_CASES = [
    [
        "UNet1DModel",
        "diffusers.models.unets.unet_1d.UNet1DModel",
        "mindone.diffusers.models.unets.unet_1d.UNet1DModel",
        (),
        dict(
            block_out_channels=(32, 64, 128, 256),
            in_channels=14,
            out_channels=14,
            time_embedding_type="positional",
            use_timestep_embedding=True,
            flip_sin_to_cos=False,
            freq_shift=1.0,
            out_block_type="OutConv1DBlock",
            mid_block_type="MidResTemporalBlock1D",
            down_block_types=("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"),
            up_block_types=("UpResnetBlock1D", "UpResnetBlock1D", "UpResnetBlock1D"),
            act_fn="swish",
        ),
        (),
        {
            "sample": np.random.randn(4, 14, 16).astype(np.float32),
            "timestep": np.array([10] * 4, dtype=np.int64),
            "return_dict": False,
        },
    ],
    [
        "UNetRLModel",
        "diffusers.models.unets.unet_1d.UNet1DModel",
        "mindone.diffusers.models.unets.unet_1d.UNet1DModel",
        (),
        dict(
            in_channels=14,
            out_channels=14,
            down_block_types=("DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D", "DownResnetBlock1D"),
            up_block_types=(),
            out_block_type="ValueFunction",
            mid_block_type="ValueFunctionMidBlock1D",
            block_out_channels=(32, 64, 128, 256),
            layers_per_block=1,
            downsample_each_block=True,
            use_timestep_embedding=True,
            freq_shift=1.0,
            flip_sin_to_cos=False,
            time_embedding_type="positional",
            act_fn="mish",
        ),
        (),
        {
            "sample": np.random.randn(4, 14, 16).astype(np.float32),
            "timestep": np.array([10] * 4, dtype=np.int64),
            "return_dict": False,
        },
    ],
]


UVIT2D_CASES = [
    [
        "UVit2DModel",
        "diffusers.models.unets.uvit_2d.UVit2DModel",
        "mindone.diffusers.models.unets.uvit_2d.UVit2DModel",
        (),
        dict(
            hidden_size=32,
            use_bias=False,
            hidden_dropout=0.0,
            cond_embed_dim=32,
            micro_cond_encode_dim=2,
            micro_cond_embed_dim=10,
            encoder_hidden_size=32,
            vocab_size=32,
            codebook_size=32,
            in_channels=32,
            block_out_channels=32,
            num_res_blocks=1,
            downsample=True,
            upsample=True,
            block_num_heads=1,
            num_hidden_layers=1,
            num_attention_heads=1,
            attention_dropout=0.0,
            intermediate_size=32,
            layer_norm_eps=1e-06,
            ln_elementwise_affine=True,
        ),
        (),
        {
            "input_ids": np.random.randint(0, 32, (2, 4, 4)).astype(np.int64),
            "encoder_hidden_states": np.random.randn(2, 77, 32).astype(np.float32),
            "pooled_text_emb": np.random.randn(2, 32).astype(np.float32),
            "micro_conds": np.random.randn(2, 5).astype(np.float32),
        },
    ],
]


KANDINSKY3_CASES = [
    [
        "Kandinsky3UNet",
        "diffusers.models.unets.unet_kandinsky3.Kandinsky3UNet",
        "mindone.diffusers.models.unets.unet_kandinsky3.Kandinsky3UNet",
        (),
        dict(
            in_channels=4,
            time_embedding_dim=4,
            groups=2,
            attention_head_dim=4,
            layers_per_block=3,
            block_out_channels=(32, 64),
            cross_attention_dim=4,
            encoder_hid_dim=32,
        ),
        (),
        {
            "sample": np.random.randn(2, 4, 8, 8).astype(np.float32),
            "timestep": np.array([10]).astype(np.int64),
            "encoder_hidden_states": np.random.randn(2, 36, 32).astype(np.float32),
            "encoder_attention_mask": np.ones((2, 36)).astype(np.float32),
            "return_dict": False,
        },
    ],
]


UNET3D_CONDITION_MODEL_CASES = [
    [
        "UNet3DConditionModel",
        "diffusers.models.unets.unet_3d_condition.UNet3DConditionModel",
        "mindone.diffusers.models.unets.unet_3d_condition.UNet3DConditionModel",
        (),
        {
            "block_out_channels": (32, 64),
            "down_block_types": (
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            "up_block_types": ("UpBlock3D", "CrossAttnUpBlock3D"),
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
            "sample_size": 32,
            "norm_num_groups": 32,
        },
        (),
        {
            "sample": np.random.randn(4, 4, 4, 32, 32).astype(np.float32),
            "timestep": np.array([10]).astype(np.int64),
            "encoder_hidden_states": np.random.randn(4, 4, 32).astype(np.float32),
            "return_dict": False,
        },
    ],
]


UNET_SPATIO_TEMPORAL_CONDITION_MODEL_CASES = [
    [
        "UNetSpatioTemporalConditionModel",
        "diffusers.models.unets.unet_spatio_temporal_condition.UNetSpatioTemporalConditionModel",
        "mindone.diffusers.models.unets.unet_spatio_temporal_condition.UNetSpatioTemporalConditionModel",
        (),
        {
            "block_out_channels": (32, 64),
            "down_block_types": (
                "CrossAttnDownBlockSpatioTemporal",
                "DownBlockSpatioTemporal",
            ),
            "up_block_types": (
                "UpBlockSpatioTemporal",
                "CrossAttnUpBlockSpatioTemporal",
            ),
            "cross_attention_dim": 32,
            "num_attention_heads": 8,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 2,
            "sample_size": 32,
            "projection_class_embeddings_input_dim": 32 * 3,
            "addition_time_embed_dim": 32,
        },
        (),
        {
            "sample": np.random.randn(2, 2, 4, 32, 32).astype(np.float32),
            "timestep": np.array([10]).astype(np.int64),
            "encoder_hidden_states": np.random.randn(2, 1, 32).astype(np.float32),
            "added_time_ids": np.array([[6, 127, 0.02], [6, 127, 0.02]]).astype(np.float32),
            "return_dict": False,
        },
    ],
]


UNET_I2VGEN_XL_CASES = [
    [
        "I2VGenXLUNet",
        "diffusers.models.unets.unet_i2vgen_xl.I2VGenXLUNet",
        "mindone.diffusers.models.unets.unet_i2vgen_xl.I2VGenXLUNet",
        (),
        dict(
            sample_size=None,
            in_channels=4,
            out_channels=4,
            down_block_types=(
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            up_block_types=(
                "UpBlock3D",
                "CrossAttnUpBlock3D",
            ),
            block_out_channels=(32, 64),
            layers_per_block=2,
            norm_num_groups=4,
            cross_attention_dim=32,
            attention_head_dim=4,
            num_attention_heads=None,
        ),
        (),
        {
            "sample": np.random.randn(2, 4, 2, 32, 32).astype(np.float32),
            "timestep": np.array([10]).astype(np.int64),
            "fps": np.array([2]).astype(np.int64),
            "image_latents": np.random.randn(2, 4, 2, 32, 32).astype(np.float32),
            "image_embeddings": np.random.randn(2, 32).astype(np.float32),
            "encoder_hidden_states": np.random.randn(2, 1, 32).astype(np.float32),
            "return_dict": False,
        },
    ],
]


UNET_MOTION_MODEL_TEST = [
    [
        "UNetMotionModel",
        "diffusers.models.unets.unet_motion_model.UNetMotionModel",
        "mindone.diffusers.models.unets.unet_motion_model.UNetMotionModel",
        (),
        {
            "block_out_channels": (32, 64),
            "down_block_types": ("CrossAttnDownBlockMotion", "DownBlockMotion"),
            "up_block_types": ("UpBlockMotion", "CrossAttnUpBlockMotion"),
            "cross_attention_dim": 32,
            "num_attention_heads": 4,
            "out_channels": 4,
            "in_channels": 4,
            "layers_per_block": 1,
            "sample_size": 32,
        },
        (),
        {
            "sample": np.random.randn(4, 4, 8, 32, 32).astype(np.float32),
            "timestep": np.array([10]).astype(np.int64),
            "encoder_hidden_states": np.random.randn(4, 4, 32).astype(np.float32),
            "return_dict": False,
        },
    ],
]


UNETSTABLECASCADE_CASES = [
    [
        "UNetStableCascadeModel_prior",
        "diffusers.models.unets.unet_stable_cascade.StableCascadeUNet",
        "mindone.diffusers.models.unets.unet_stable_cascade.StableCascadeUNet",
        (),
        dict(
            block_out_channels=(2048, 2048),
            block_types_per_layer=(
                ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
                ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
            ),
            clip_image_in_channels=768,
            clip_seq=4,
            clip_text_in_channels=1280,
            clip_text_pooled_in_channels=1280,
            conditioning_dim=2048,
            down_blocks_repeat_mappers=(1, 1),
            down_num_layers_per_block=(8, 24),
            dropout=(0.1, 0.1),
            effnet_in_channels=None,
            in_channels=16,
            kernel_size=3,
            num_attention_heads=(32, 32),
            out_channels=16,
            patch_size=1,
            pixel_mapper_in_channels=None,
            self_attn=True,
            switch_level=(False,),
            timestep_conditioning_type=("sca", "crp"),
            timestep_ratio_embedding_dim=64,
            up_blocks_repeat_mappers=(1, 1),
            up_num_layers_per_block=(24, 8),
        ),
        (),
        {
            "sample": np.random.randn(1, 16, 24, 24).astype(np.float32),
            "timestep_ratio": np.array([1], dtype=np.float32),
            "clip_text_pooled": np.random.randn(1, 1, 1280).astype(np.float32),
            "clip_text": np.random.randn(1, 77, 1280).astype(np.float32),
            "clip_img": np.random.randn(1, 1, 768).astype(np.float32),
            "pixels": np.random.randn(1, 3, 8, 8).astype(np.float32),
            "return_dict": False,
        },
    ],
    [
        "UNetStableCascadeModel_decoder",
        "diffusers.models.unets.unet_stable_cascade.StableCascadeUNet",
        "mindone.diffusers.models.unets.unet_stable_cascade.StableCascadeUNet",
        (),
        dict(
            block_out_channels=(320, 640, 1280, 1280),
            block_types_per_layer=(
                ("SDCascadeResBlock", "SDCascadeTimestepBlock"),
                ("SDCascadeResBlock", "SDCascadeTimestepBlock"),
                ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
                ("SDCascadeResBlock", "SDCascadeTimestepBlock", "SDCascadeAttnBlock"),
            ),
            clip_image_in_channels=None,
            clip_seq=4,
            clip_text_in_channels=None,
            clip_text_pooled_in_channels=1280,
            conditioning_dim=1280,
            down_blocks_repeat_mappers=(1, 1, 1, 1),
            down_num_layers_per_block=(2, 6, 28, 6),
            dropout=(0, 0, 0.1, 0.1),
            effnet_in_channels=16,
            in_channels=4,
            kernel_size=3,
            num_attention_heads=(0, 0, 20, 20),
            out_channels=4,
            patch_size=2,
            pixel_mapper_in_channels=3,
            self_attn=True,
            switch_level=None,
            timestep_conditioning_type=("sca",),
            timestep_ratio_embedding_dim=64,
            up_blocks_repeat_mappers=(3, 3, 2, 2),
            up_num_layers_per_block=(6, 28, 6, 2),
        ),
        (),
        {
            "sample": np.random.randn(1, 4, 256, 256).astype(np.float32),
            "timestep_ratio": np.array([1], dtype=np.float32),
            "clip_text_pooled": np.random.randn(1, 1, 1280).astype(np.float32),
            "clip_text": np.random.randn(1, 77, 1280).astype(np.float32),
            "pixels": np.random.randn(1, 3, 8, 8).astype(np.float32),
            "return_dict": False,
        },
    ],
]


UNETS_CASES = (
    UNET1D_CASES
    + UVIT2D_CASES
    + KANDINSKY3_CASES
    + UNET3D_CONDITION_MODEL_CASES
    + UNET_SPATIO_TEMPORAL_CONDITION_MODEL_CASES
    + UNET_I2VGEN_XL_CASES
    + UNET_MOTION_MODEL_TEST
    + UNETSTABLECASCADE_CASES
)


# all
ALL_CASES = LAYERS_CASES + AE_CASES + TRANSFORMERS_CASES + UNETS_CASES
