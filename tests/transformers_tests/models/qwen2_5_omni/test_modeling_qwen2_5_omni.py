"""Adapted from https://github.com/huggingface/transformers/tree/main/tests/models/qwen2_5_omni/test_modeling_qwen2_5_omni.py."""

# This module contains test cases that are defined in the `.test_cases.py` file, structured as lists or tuples like
#     [name, pt_module, ms_module, init_args, init_kwargs, inputs_args, inputs_kwargs, outputs_map].
#
# Each defined case corresponds to a pair consisting of PyTorch and MindSpore modules, including their respective
# initialization parameters and inputs for the forward. The testing framework adopted here is designed to generically
# parse these parameters to assess and compare the precision of forward outcomes between the two frameworks.
#
# In cases where models have unique initialization procedures or require testing with specialized output formats,
# it is necessary to develop distinct, dedicated test cases.

# NOTE: install transformers by `pip install transformers==4.53.0`

import inspect

import numpy as np
import pytest
import torch
import transformers

import mindspore as ms

from tests.modeling_test_utils import (
    MS_DTYPE_MAPPING,
    PT_DTYPE_MAPPING,
    compute_diffs,
    generalized_parse_args,
    get_modules,
)
from tests.transformers_tests.models.modeling_common import floats_numpy, ids_numpy

DTYPE_AND_THRESHOLDS = {"fp32": 5e-5, "fp16": 5e-3, "bf16": 5e-2}
MODES = [1]

if transformers.__version__ >= "4.52.3":
    from transformers.models.qwen2_5_omni import (
        Qwen2_5OmniConfig,
        Qwen2_5OmniTalkerConfig,
        Qwen2_5OmniThinkerConfig,
        Qwen2_5OmniToken2WavConfig,
    )

    class Qwen2_5_OmniModelTester:
        def __init__(
            self,
            batch_size=3,
            feat_seq_length=30,
            num_channels=3,
            image_size=14,
            seq_length=39,
            vision_config={
                "depth": 2,
                "embed_dim": 32,
                "hidden_act": "quick_gelu",
                "hidden_size": 32,
                "out_hidden_size": 32,
                "intermediate_size": 24,
                "mlp_ratio": 4,
                "num_heads": 4,
                "patch_size": 14,
                "spatial_merge_size": 1,
                "temporal_patch_size": 2,
                "fullatt_block_indexes": [0],
                "initializer_range": 0.02,
            },
            audio_config={
                "model_type": "qwen_omni_thinker_audio_encoder",
                "d_model": 32,
                "encoder_attention_heads": 4,
                "encoder_ffn_dim": 32,
                "encoder_layers": 2,
                "num_mel_bins": 20,
                "max_source_positions": 1500,
                "initializer_range": 0.02,
                "n_window": 100,
                "output_dim": 32,
            },
            text_config={  # mrope_section: sum*2=8 = head_dim = hidden_size//num_attention_heads = 32//4=8
                "rope_scaling": {"mrope_section": [1, 1, 2], "rope_type": "default", "type": "default"},
                "vocab_size": 99,
                "hidden_size": 32,
                "intermediate_size": 37,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_act": "silu",
                "max_position_embeddings": 1024,
                "rms_norm_eps": 1e-06,
                "use_cache": False,
                "tie_word_embeddings": False,
                "rope_theta": 1000000.0,
                "use_sliding_window": False,
                "sliding_window": 50,
                "max_window_layers": 3,
                "attention_dropout": 0.0,
                "pad_token_id": 0,
                "initializer_range": 0.02,
            },
            dit_config={
                "hidden_size": 32,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "ff_mult": 2,
                "emb_dim": 32,
                "head_dim": 8,
                "rope_theta": 10000.0,
                "max_position_embeddings": 1024,
                "block_size": 24,
                "look_ahead_layers": [1],
                "look_backward_layers": [0, 2],
                "repeats": 2,
                "num_embeds": 32,
                "mel_dim": 10,
                "dropout": 0.1,
                "enc_emb_dim": 32,
                "enc_dim": 16,
                "enc_channels": [256, 256, 256, 256, 768],
                "enc_kernel_sizes": [5, 3, 3, 3, 1],
                "enc_dilations": [1, 2, 3, 4, 1],
                "enc_attention_channels": 8,
                "enc_res2net_scale": 2,
                "enc_se_channels": 8,
            },
            bigvgan_config={
                "mel_dim": 10,
                "upsample_initial_channel": 24,
                "resblock_kernel_sizes": [3, 7, 11],
                "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                "upsample_rates": [5, 3, 2, 2, 2, 2],
                "upsample_kernel_sizes": [11, 7, 4, 4, 4, 4],
            },
            audio_token_index=1,
            image_token_index=2,
            video_token_index=3,
            position_id_per_seconds=25,
            seconds_per_chunk=2,
            audio_start_token_id=4,
            audio_end_token_id=5,
            user_token_id=6,
            vision_start_token_id=7,
            vision_end_token_id=8,
            initializer_range=0.02,
            attn_implementation="eager",
        ):
            self.batch_size = batch_size
            self.feat_seq_length = feat_seq_length
            self.num_channels = num_channels
            self.image_size = image_size
            self.seq_length = seq_length
            self.vision_config = vision_config
            self.audio_config = audio_config
            self.text_config = text_config
            self.dit_config = dit_config
            self.bigvgan_config = bigvgan_config
            self.audio_token_index = audio_token_index
            self.image_token_index = image_token_index
            self.video_token_index = video_token_index
            self.position_id_per_seconds = position_id_per_seconds
            self.seconds_per_chunk = seconds_per_chunk
            self.audio_start_token_id = audio_start_token_id
            self.audio_end_token_id = audio_end_token_id
            self.user_token_id = user_token_id
            self.vision_start_token_id = vision_start_token_id
            self.vision_end_token_id = vision_end_token_id
            self.initializer_range = initializer_range
            self.attn_implementation = attn_implementation

            # common model tests
            self.intermediate_size = self.text_config["intermediate_size"]
            self.max_position_embeddings = self.text_config["max_position_embeddings"]
            self.rope_scaling = self.text_config["rope_scaling"]
            self.use_cache = self.text_config["use_cache"]
            self.use_sliding_window = self.text_config["use_sliding_window"]
            self.num_hidden_layers = self.text_config["num_hidden_layers"]
            self.hidden_size = self.text_config["hidden_size"]
            self.num_attention_heads = self.text_config["num_attention_heads"]
            self.vocab_size = self.text_config["vocab_size"]

        def get_large_model_config(self):
            return Qwen2_5OmniConfig.from_pretrained("Qwen/Qwen2.5-Omni-7B")

        def get_config(self):
            thinker_config = Qwen2_5OmniThinkerConfig(
                audio_config=self.audio_config,
                vision_config=self.vision_config,
                text_config=self.text_config,
                audio_token_index=self.audio_token_index,
                image_token_index=self.image_token_index,
                video_token_index=self.video_token_index,
                position_id_per_seconds=self.position_id_per_seconds,
                seconds_per_chunk=self.seconds_per_chunk,
                audio_start_token_id=self.audio_start_token_id,
                audio_end_token_id=self.audio_end_token_id,
                vision_start_token_id=self.vision_start_token_id,
                vision_end_token_id=self.vision_end_token_id,
                user_token_id=self.user_token_id,
                use_cache=self.use_cache,
                initializer_range=self.initializer_range,
                attn_implementation=self.attn_implementation,
            )

            talker_config = Qwen2_5OmniTalkerConfig(
                audio_token_index=self.audio_token_index,
                image_token_index=self.image_token_index,
                video_token_index=self.video_token_index,
                vocab_size=self.vocab_size,
                tts_codec_start_token_id=93,
                tts_codec_end_token_id=94,
                tts_codec_pad_token_id=92,
                tts_codec_mask_token_id=96,
                embedding_size=self.hidden_size,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                num_hidden_layers=self.num_hidden_layers,
                num_attention_heads=self.num_attention_heads,
                max_position_embeddings=self.max_position_embeddings,
                rope_scaling=self.rope_scaling,
                head_dim=self.hidden_size // self.num_attention_heads,
                use_cache=self.use_cache,
                use_sliding_window=self.use_sliding_window,
                audio_start_token_id=self.audio_start_token_id,
                audio_end_token_id=self.audio_end_token_id,
                attn_implementation=self.attn_implementation,
            )
            token2wav_config = Qwen2_5OmniToken2WavConfig(
                dit_config=self.dit_config,
                bigvgan_config=self.bigvgan_config,
                attn_implementation=self.attn_implementation,
            )

            return thinker_config, talker_config, token2wav_config

        def prepare_config_and_inputs(self):
            thinker_config, talker_config, token2wav_config = self.get_config()
            patch_size = thinker_config.vision_config.patch_size
            temporal_patch_size = thinker_config.vision_config.temporal_patch_size
            pixel_values = floats_numpy(
                [
                    self.batch_size * (self.image_size**2) // (patch_size**2),
                    self.num_channels * (patch_size**2) * temporal_patch_size,
                ]
            )
            pixel_grid_thw = np.array(
                [[1, self.image_size / patch_size, self.image_size / patch_size]] * self.batch_size,
                dtype=np.int64,
            )
            input_features_values = floats_numpy(
                [self.batch_size, self.audio_config["num_mel_bins"], self.feat_seq_length]
            )
            feature_attention_mask = np.ones([self.batch_size, self.feat_seq_length], dtype=np.int64)

            return (
                thinker_config,
                talker_config,
                token2wav_config,
                pixel_values,
                pixel_grid_thw,
                input_features_values,
                feature_attention_mask,
            )

        def prepare_config_and_inputs_for_common(self):
            config_and_inputs = self.prepare_config_and_inputs()
            (
                thinker_config,
                talker_config,
                token2wav_config,
                pixel_values,
                pixel_grid_thw,
                input_features_values,
                feature_attention_mask,
            ) = config_and_inputs
            input_ids = (
                ids_numpy([self.batch_size, self.seq_length], thinker_config.get_text_config().vocab_size - 3) + 3
            )
            attention_mask = np.ones(input_ids.shape, dtype=np.int64)

            # Thinker inputs
            # Make sure no other tokens are set to special, to prevetn flakiness
            tokens_to_replace = np.array(
                [
                    thinker_config.image_token_index,
                    thinker_config.audio_token_index,
                    thinker_config.audio_start_token_id,
                    thinker_config.audio_end_token_id,
                    thinker_config.vision_start_token_id,
                    thinker_config.vision_end_token_id,
                ]
            )
            input_ids[np.isin(input_ids, tokens_to_replace)] = thinker_config.text_config.pad_token_id

            attention_mask[:, :1] = 0

            # Audio token placeholders should be wrapped in start and end token ids
            audio_feat_length = ((self.feat_seq_length - 1) // 2 + 1 - 2) // 2 + 1
            input_ids[:, 1] = thinker_config.audio_start_token_id
            input_ids[:, 2 : (2 + audio_feat_length)] = thinker_config.audio_token_index
            input_ids[:, 2 + audio_feat_length] = thinker_config.audio_end_token_id

            # Image token placeholders should be wrapped in start and end token ids
            input_ids[:, -4:-1] = np.array(
                [
                    thinker_config.vision_start_token_id,
                    thinker_config.image_token_index,
                    thinker_config.vision_end_token_id,
                ]
            )
            thinker_inputs_dict = {
                "input_features": input_features_values,
                "feature_attention_mask": feature_attention_mask,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_grid_thw": pixel_grid_thw,
                "pixel_values": pixel_values,
            }

            # Talker inputs
            talker_text_bos_token = talker_config.tts_text_start_token_id
            talker_input_text_ids = np.concatenate(
                [
                    input_ids,
                    np.array([[talker_text_bos_token]] * self.batch_size, dtype=np.int64),
                ],
                axis=-1,
            )
            talker_input_ids = np.concatenate(
                [
                    np.full_like(input_ids, fill_value=talker_config.tts_codec_mask_token_id),
                    np.array([[talker_config.tts_codec_pad_token_id]] * self.batch_size, dtype=np.int64),
                    np.array([[talker_config.tts_codec_start_token_id]] * self.batch_size, dtype=np.int64),
                ],
                axis=1,
            )
            talker_attention_mask = np.concatenate([attention_mask, np.ones((self.batch_size, 2))], axis=1)
            talker_inputs_dict = {
                "input_ids": talker_input_ids,
                "input_text_ids": talker_input_text_ids,
                "attention_mask": talker_attention_mask,
            }
            return thinker_config, talker_config, token2wav_config, thinker_inputs_dict, talker_inputs_dict

    model_tester = Qwen2_5_OmniModelTester()
    (
        thinker_config,
        talker_config,
        token2wav_config,
        thinker_inputs_dict,
        talker_inputs_dict,
    ) = model_tester.prepare_config_and_inputs_for_common()

    TEST_CASES = [
        [
            "Qwen2_5OmniThinkerForConditionalGeneration",
            "transformers.Qwen2_5OmniThinkerForConditionalGeneration",
            "mindone.transformers.Qwen2_5OmniThinkerForConditionalGeneration",
            (thinker_config,),
            {},
            (),
            thinker_inputs_dict,
            {
                "logits": 0,
            },
        ],
        [
            "Qwen2_5OmniThinkerTextModel",
            "transformers.Qwen2_5OmniThinkerTextModel",
            "mindone.transformers.Qwen2_5OmniThinkerTextModel",
            (thinker_config.text_config,),
            {},
            (),
            {
                "input_ids": thinker_inputs_dict["input_ids"],
                "attention_mask": thinker_inputs_dict["attention_mask"],
            },
            {
                "last_hidden_state": 0,
            },
        ],
        [
            "Qwen2_5OmniTalkerModel",
            "transformers.Qwen2_5OmniTalkerModel",
            "mindone.transformers.Qwen2_5OmniTalkerModel",
            (talker_config,),
            {},
            (),
            {
                "input_ids": talker_inputs_dict["input_ids"],
                "attention_mask": talker_inputs_dict["attention_mask"],
            },
            {
                "last_hidden_state": 0,
            },
        ],
        # [
        #     "Qwen2_5OmniToken2WavModel",
        #     "transformers.Qwen2_5OmniToken2WavModel",
        #     "mindone.transformers.Qwen2_5OmniToken2WavModel",
        #     (token2wav_config,),
        #     {},
        #     (),
        #     {
        #         "code": code,
        #         "conditioning": conditioning,
        #         "reference_mel": reference_mel,
        #     },
        #     {},
        # ],
    ]

    @pytest.mark.parametrize(
        "name,pt_module,ms_module,init_args,init_kwargs,inputs_args,inputs_kwargs,outputs_map,dtype,mode",
        [
            case
            + [
                dtype,
            ]
            + [
                mode,
            ]
            for case in TEST_CASES
            for dtype in DTYPE_AND_THRESHOLDS.keys()
            for mode in MODES
        ],
    )
    @pytest.mark.skipif(transformers.__version__ < "4.52.3", reason="need to set specific transformers version")
    def test_named_modules(
        name,
        pt_module,
        ms_module,
        init_args,
        init_kwargs,
        inputs_args,
        inputs_kwargs,
        outputs_map,
        dtype,
        mode,
    ):
        ms.set_context(mode=mode)

        (
            pt_model,
            ms_model,
            pt_dtype,
            ms_dtype,
        ) = get_modules(pt_module, ms_module, dtype, *init_args, **init_kwargs)
        pt_inputs_args, pt_inputs_kwargs, ms_inputs_args, ms_inputs_kwargs = generalized_parse_args(
            pt_dtype, ms_dtype, *inputs_args, **inputs_kwargs
        )
        # set `hidden_dtype` if requiring, for some modules always compute in float
        # precision and require specific `hidden_dtype` to cast before return
        if "hidden_dtype" in inspect.signature(pt_model.forward).parameters:
            pt_inputs_kwargs.update({"hidden_dtype": PT_DTYPE_MAPPING[pt_dtype]})
            ms_inputs_kwargs.update({"hidden_dtype": MS_DTYPE_MAPPING[ms_dtype]})

        with torch.no_grad():
            pt_outputs = pt_model(*pt_inputs_args, **pt_inputs_kwargs)
        ms_outputs = ms_model(*ms_inputs_args, **ms_inputs_kwargs)
        # print("ms:", ms_outputs)
        # print("pt:", pt_outputs)
        if outputs_map:
            pt_outputs_n = []
            ms_outputs_n = []
            for pt_key, ms_idx in outputs_map.items():
                # print("===map", pt_key, ms_idx)
                pt_output = getattr(pt_outputs, pt_key)
                ms_output = ms_outputs[ms_idx]
                if isinstance(pt_output, (list, tuple)):
                    pt_outputs_n += list(pt_output)
                    ms_outputs_n += list(ms_output)
                else:
                    pt_outputs_n.append(pt_output)
                    ms_outputs_n.append(ms_output)
            diffs = compute_diffs(pt_outputs_n, ms_outputs_n)
        else:
            diffs = compute_diffs(pt_outputs, ms_outputs)

        THRESHOLD = DTYPE_AND_THRESHOLDS[ms_dtype]
        assert (np.array(diffs) < THRESHOLD).all(), (
            f"ms_dtype: {ms_dtype}, pt_type:{pt_dtype}, "
            f"Outputs({np.array(diffs).tolist()}) has diff bigger than {THRESHOLD}"
        )
