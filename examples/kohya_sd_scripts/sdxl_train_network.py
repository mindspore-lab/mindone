import argparse

import train_network
from library import sdxl_model_util, sdxl_train_util, train_util
from library.utils import setup_logging

import mindspore as ms
from mindspore import ops

setup_logging()
import logging

logger = logging.getLogger(__name__)


class SdxlNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.vae_scale_factor = sdxl_model_util.VAE_SCALE_FACTOR
        self.is_sdxl = True

    def assert_extra_args(self, args, train_dataset_group):
        super().assert_extra_args(args, train_dataset_group)
        sdxl_train_util.verify_sdxl_training_args(args)

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used \
                / Text Encoderの出力をキャッシュするときはcaption_dropout_rate, shuffle_caption, token_warmup_step, caption_tag_dropout_rateは使えません"

        assert (
            args.network_train_unet_only or not args.cache_text_encoder_outputs
        ), "network for Text Encoder cannot be trained with caching Text Encoder outputs / Text Encoderの出力をキャッシュしながらText Encoderのネットワークを学習することはできません"

        train_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype):
        (
            load_stable_diffusion_format,
            text_encoder1,
            text_encoder2,
            vae,
            unet,
            logit_scale,
            ckpt_info,
        ) = sdxl_train_util.load_target_model(args, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, weight_dtype)

        self.load_stable_diffusion_format = load_stable_diffusion_format
        self.logit_scale = logit_scale
        self.ckpt_info = ckpt_info

        return sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, [text_encoder1, text_encoder2], vae, unet

    def load_tokenizer(self, args):
        tokenizer = sdxl_train_util.load_tokenizers(args)
        return tokenizer

    def is_text_encoder_outputs_cached(self, args):
        return args.cache_text_encoder_outputs

    def cache_text_encoder_outputs_if_needed(
        self, args, unet, vae, tokenizers, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            raise NotImplementedError
        else:
            # Text Encoderから毎回出力を取得するので、GPUに乗せておく
            text_encoders[0].to(dtype=weight_dtype)
            text_encoders[1].to(dtype=weight_dtype)

    def get_text_cond(
        self,
        args,
        tokenizer1_model_max_length,
        tokenizer2_model_max_length,
        tokenizer2_eos_token_id,
        text_encoder1,
        text_encoder2,
        input_ids,
        input_ids2,
        text_encoder_outputs1_list,
        text_encoder_outputs2_list,
        text_encoder_pool2_list,
        weight_dtype,
    ):
        if text_encoder_outputs1_list is None:
            input_ids1 = input_ids
            input_ids2 = input_ids2
            encoder_hidden_states1, encoder_hidden_states2, pool2 = train_util.get_hidden_states_sdxl(
                args.max_token_length,
                input_ids1,
                input_ids2,
                tokenizer1_model_max_length,
                tokenizer2_model_max_length,
                tokenizer2_eos_token_id,
                text_encoder1,
                text_encoder2,
                None if not args.full_fp16 else weight_dtype,
            )
        else:
            encoder_hidden_states1 = text_encoder_outputs1_list.to(weight_dtype)
            encoder_hidden_states2 = text_encoder_outputs2_list.to(weight_dtype)
            pool2 = text_encoder_pool2_list.to(weight_dtype)

        return encoder_hidden_states1, encoder_hidden_states2, pool2

    def call_unet(
        self,
        args,
        unet,
        noisy_latents,
        timesteps,
        text_conds,
        original_sizes_hw,
        crop_top_lefts,
        target_sizes_hw,
        weight_dtype,
    ):
        noisy_latents = noisy_latents.to(weight_dtype)

        # get size embeddings
        orig_size = original_sizes_hw
        crop_size = crop_top_lefts
        target_size = target_sizes_hw
        embs = sdxl_train_util.get_size_embeddings(orig_size, crop_size, target_size).to(weight_dtype)

        # concat embeddings
        encoder_hidden_states1, encoder_hidden_states2, pool2 = text_conds
        vector_embedding = ops.cat([pool2, embs], axis=1).to(weight_dtype)
        text_embedding = ops.cat([encoder_hidden_states1, encoder_hidden_states2], axis=2).to(weight_dtype)

        noise_pred = unet(noisy_latents, timesteps, text_embedding, vector_embedding)
        return noise_pred

    def sample_images(self, args, epoch, global_step, vae, tokenizer, text_encoder, unet):
        # not implement
        pass


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    sdxl_train_util.add_sdxl_training_arguments(parser)
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)
    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT, jit_config={"jit_level": "O1"})
    train_util.init_distributed_device(args)
    trainer = SdxlNetworkTrainer()
    trainer.train(args)
