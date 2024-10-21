import logging
import os.path
from typing import List

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
#from ...core_code.header import *
from examples.nextgpt.core_code.model.ImageBind.models import imagebind_model
#from ...core_code.model.ImageBind import data
from examples.nextgpt.core_code.model import modeling_llama
from mindnlp.transformers import StoppingCriteria, StoppingCriteriaList,LlamaTokenizer,LlamaForCausalLM
from mindone.diffusers import StableDiffusionPipeline
from examples.nextgpt.core_code.model.custom_vd import TextToVideoSDPipeline
from examples.nextgpt.core_code.model.custom_ad import AudioLDMPipeline
from examples.nextgpt.core_code.model.layers import *
from examples.nextgpt.core_code.model.common.utils import *
from examples.nextgpt.core_code.model.ImageBind.models.imagebind_model import *
from examples.nextgpt.core_code.model.ImageBind import data

import os
import re

from mindnlp.peft import TaskType
from mindone.diffusers._peft import LoraConfig, get_peft_model
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops: List = None, encounters: int = 1):
        super().__init__()
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: mindspore.Tensor, scores: mindspore.Tensor):
        stop_count = 0
        for stop in self.stops:
            _stop = mindspore.tensor(stop)
            condition = _stop[0] == input_ids
            for i,e in enumerate(condition):
                if e.any():
                    if ops.all(input_ids[0][i:i + len(_stop)] == _stop):
                        stop_count += 1
        if stop_count >= self.ENCOUNTERS:
            return True
        return False


class NextGPTModel(nn.Cell):
    """LoRA for LLaMa model"""

    def __init__(self, **args):
        super(NextGPTModel, self).__init__()
        self.args = args

        self.max_length = args['max_length']
        self.stage = args['stage']
        print('args max_length', args['max_length'])

        imagebind_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'imagebind_ckpt',
                                           self.args['imagebind_version'])
        print(f'Initializing visual encoder from {imagebind_ckpt_path} ...')
        # self.visual_encoder, self.visual_hidden_size = \
        #     imagebind_model.imagebind_huge(pretrained=True, store_path=imagebind_ckpt_path)
        ## debug
        self.visual_encoder, self.visual_hidden_size = \
            imagebind_model.imagebind_huge(pretrained=False)
        # free vision encoder
        for param in self.visual_encoder.trainable_params():
            param.requires_grad = False
        self.visual_encoder.set_train(False)
        print('Visual encoder initialized.')

        self.vicuna_ckpt_path = os.path.join(self.args['pretrained_ckpt_path'], 'vicuna_ckpt',
                                             self.args['vicuna_version'])
        print(f'Initializing language decoder from {self.vicuna_ckpt_path} ...')

        self.llama_model = modeling_llama.LlamaForCausalLM.from_pretrained(self.vicuna_ckpt_path)
        if self.args.get('freeze_lm'):
            print("Freezing the LLaMa ...")
            for param in self.llama_model.trainable_params():
                param.requires_grad = False
            self.llama_model.set_train(False)
        else:
            print("Instruct tuning the LLaMa ...")
            # add the lora module
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=self.args['lora_r'],
                lora_alpha=self.args['lora_alpha'],
                lora_dropout=self.args['lora_dropout'],
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
            )

            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
        print('Language decoder initialized.')

        # use the new trained tokenizer
        tokenizer_path = self.vicuna_ckpt_path
        print(f'Initializing tokenizer from {tokenizer_path} ...')
        self.llama_tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token
        self.llama_tokenizer.padding_side = "right"
        # self.llama_tokenizer.add_special_tokens({"mask_token": "[MASK]"})
        self._add_image_token()
        self._add_video_token()
        self._add_audio_token()
        self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))
        print('Tokenizer initialized.')

        self.llama_proj = nn.Dense(
            self.visual_hidden_size, self.llama_model.config.hidden_size
        )
        if self.args.get('freeze_input_proj'):
            for param in self.llama_proj.trainable_params():
                param.requires_grad = False

        self.input_embeddings = self.llama_model.get_input_embeddings()

        # the alignment module for LLM-TO-IMAGE
        self.sd_ckpt_path = self.args['image_diffusion']
        self.gen_text_hidden_fcs = nn.CellList([])
        for layer_idx in self.args['text_emb_to_img_layers']:
            if layer_idx == -1 or layer_idx == self.llama_model.config.num_hidden_layers:
                in_dim = self.llama_model.config.hidden_size

                self.gen_text_hidden_fcs.append(
                    TextFcLayer(in_dim, 768, num_input_tokens=self.args['num_gen_img_tokens'],
                                num_output_tokens=self.args['num_clip_tokens'],
                                mode=self.args['text_fc_to_img_mode']))
            # self.sd_pipe.text_encoder.config.hidden_size
            elif layer_idx < self.llama_model.config.num_hidden_layers:
                self.gen_text_hidden_fcs.append(
                    TextFcLayer(self.llama_model.config.hidden_size, 768,
                                num_input_tokens=self.args['num_gen_img_tokens'],
                                num_output_tokens=self.args['num_clip_tokens'],
                                mode=self.args['text_fc_to_img_mode']))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only has {self.llama_model.config.num_hidden_layers} layers.')

        # the alignment module for LLM-TO-VIDEO
        self.vd_ckpt_path = self.args['video_diffusion']
        self.gen_text_hidden_fcs_video = nn.CellList([])
        for layer_idx in self.args['text_emb_to_video_layers']:
            if layer_idx == -1 or layer_idx == self.llama_model.config.num_hidden_layers:
                in_dim = self.llama_model.config.hidden_size  # 4096

                self.gen_text_hidden_fcs_video.append(
                    TextFcLayer(in_dim, 1024, num_input_tokens=self.args['num_gen_video_tokens'],
                                num_output_tokens=self.args['num_clip_tokens'],
                                mode=self.args['text_fc_to_video_mode']))
            # self.vd_pipe.text_encoder.config.hidden_size
            elif layer_idx < self.llama_model.config.num_hidden_layers:
                self.gen_text_hidden_fcs_video.append(
                    TextFcLayer(self.llama_model.config.hidden_size, 1024,
                                num_input_tokens=self.args['num_gen_video_tokens'],
                                num_output_tokens=self.args['num_clip_tokens'],
                                mode=self.args['text_fc_to_video_mode']))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only has {self.llama_model.config.num_hidden_layers} layers.')

        # the alignment module for LLM-TO-AUDIO
        self.ad_ckpt_path = self.args['audio_diffusion']
        self.gen_text_hidden_fcs_audio = nn.CellList([])
        for layer_idx in self.args['text_emb_to_audio_layers']:
            if layer_idx == -1 or layer_idx == self.llama_model.config.num_hidden_layers:
                in_dim = self.llama_model.config.hidden_size

                self.gen_text_hidden_fcs_audio.append(
                    TextFcLayer(in_dim, 512,
                                num_input_tokens=self.args['num_gen_audio_tokens'],
                                num_output_tokens=1,
                                mode=self.args['text_fc_to_audio_mode']))
            # self.ad_pipe.text_encoder.config.projection_dim
            elif layer_idx < self.llama_model.config.num_hidden_layers:
                self.gen_text_hidden_fcs_audio.append(
                    TextFcLayer(self.llama_model.config.hidden_size, 512,
                                num_input_tokens=self.args['num_gen_audio_tokens'],
                                num_output_tokens=1,
                                mode=self.args['text_fc_to_audio_mode']))
            else:
                raise ValueError(
                    f'Embedding of layer {layer_idx} was requested but model only has {self.llama_model.config.num_hidden_layers} layers.')

        if self.args.get('freeze_output_proj'):
            for param in self.gen_text_hidden_fcs.trainable_params():
                param.requires_grad = False
            for param in self.gen_text_hidden_fcs_video.trainable_params():
                param.requires_grad = False
            for param in self.gen_text_hidden_fcs_audio.trainable_params():
                param.requires_grad = False

    def _add_image_token(self):
        # Add an image token for loss masking (and visualization) purposes.
        self.llama_tokenizer.add_tokens(["<Img>"])  # add special image token to tokenizer
        self.llama_tokenizer.add_tokens(["</Img>"])  # add special image token to tokenizer

        # Add [IMG] tokens to the vocabulary.
        self.args['gen_img_token_idx'] = []
        for i in range(self.args['num_gen_img_tokens']):
            print(f'Adding [IMG{i}] token to vocabulary.')
            print(f'Before adding new token, tokenizer("[IMG{i}]") =',
                  self.llama_tokenizer(f'[IMG{i}]', add_special_tokens=False))
            num_added_tokens = self.llama_tokenizer.add_tokens(f'[IMG{i}]')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("[IMG{i}]") =',
                  self.llama_tokenizer(f'[IMG{i}]', add_special_tokens=False))
            gen_token_idx = self.llama_tokenizer(f'[IMG{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_img_token_idx'].append(gen_token_idx[0])

    def _add_video_token(self):
        # self.llama_tokenizer.add_tokens({"<Vid>"})  # add special video token to tokenizer
        # self.llama_tokenizer.add_tokens({"</Vid>"})  # add special video token to tokenizer

        # Add [VID] tokens to the vocabulary.
        self.args['gen_video_token_idx'] = []
        for i in range(self.args['num_gen_video_tokens']):
            print(f'Adding [VID{i}] token to vocabulary.')
            print(f'Before adding new token, tokenizer("[VID{i}]") =',
                  self.llama_tokenizer(f'[VID{i}]', add_special_tokens=False))
            num_added_tokens = self.llama_tokenizer.add_tokens(f'[VID{i}]')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("[VID{i}]") =',
                  self.llama_tokenizer(f'[VID{i}]', add_special_tokens=False))
            gen_token_idx = self.llama_tokenizer(f'[VID{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_video_token_idx'].append(gen_token_idx[0])

    def _add_audio_token(self):
        # self.llama_tokenizer.add_tokens({"<Aud>"})  # add special audio token to tokenizer
        # self.llama_tokenizer.add_tokens({"</Aud>"})  # add special audio token to tokenizer

        # Add [AUD] tokens to the vocabulary.
        self.args['gen_audio_token_idx'] = []
        for i in range(self.args['num_gen_audio_tokens']):
            print(f'Adding [AUD{i}] token to vocabulary.')
            print(f'Before adding new token, tokenizer("[AUD{i}]") =',
                  self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False))
            num_added_tokens = self.llama_tokenizer.add_tokens(f'[AUD{i}]')
            print(f'After adding {num_added_tokens} new tokens, tokenizer("[AUD{i}]") =',
                  self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False))
            gen_token_idx = self.llama_tokenizer(f'[AUD{i}]', add_special_tokens=False).input_ids
            assert len(gen_token_idx) == 1, gen_token_idx
            self.args['gen_audio_token_idx'].append(gen_token_idx[0])

    def encode_video(self, video_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_video_data(video_paths)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        embeddings = self.visual_encoder(inputs)
        video_embeds = embeddings[ModalityType.VISION]  # bsz x 1024
        inputs_llama = self.llama_proj(video_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = ops.ones(inputs_llama.shape[:-1], dtype=mindspore.int64) # bsz x 1
        return inputs_llama, atts_llama

    def encode_audio(self, audio_paths):
        inputs = {ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        embeddings = self.visual_encoder(inputs)
        audio_embeds = embeddings[ModalityType.AUDIO]  # bsz x 1024
        inputs_llama = self.llama_proj(audio_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = ops.ones(inputs_llama.shape[:-1], dtype=mindspore.int64)  # bsz x 1
        return inputs_llama, atts_llama

    def encode_image(self, image_paths):
        inputs = {ModalityType.VISION: data.load_and_transform_vision_data(image_paths)}
        # convert into visual dtype
        inputs = {key: inputs[key].to(self.llama_model.dtype) for key in inputs}
        embeddings = self.visual_encoder(inputs)
        image_embeds = embeddings['vision']  # bsz x 1024
        inputs_llama = self.llama_proj(image_embeds).unsqueeze(1)  # bsz x 1 x llama_size
        atts_llama = ops.ones(inputs_llama.shape[:-1], dtype=mindspore.int64)  # bsz x 1
        return inputs_llama, atts_llama

    def prompt_wrap(self, img_embeds, input_ids, target_ids, attention_mask):
        '''
            input_ids, target_ids, attention_mask: bsz x s2
        '''
        input_ids = input_ids  # bsz x s2
        target_ids = target_ids  # bsz x s2
        attention_mask = attention_mask  # bsz x s2

        batch_size = input_ids.shape[0]

        bos = ops.ones([batch_size, 1], dtype=input_ids.dtype) * self.llama_tokenizer.bos_token_id  # bsz x 1
        if self.args['freeze_lm']:
            p_after_embeds = self.llama_model.model.embed_tokens(input_ids).expand(batch_size, -1,
                                                                                   -1)  # bsz x s2 x embed_dim
            bos_embeds = self.llama_model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        else:
            p_after_embeds = self.llama_model.model.model.embed_tokens(input_ids).expand(batch_size, -1,
                                                                                         -1)  # bsz x s2 x embed_dim
            bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
        if img_embeds is not None:
            p_before = '### Human: <Img>'
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="np", add_special_tokens=False)
            # peft model need deeper call
            if self.args['freeze_lm']:
                p_before_embeds = self.llama_model.model.embed_tokens(mindspore.Tensor(p_before_tokens.input_ids)).expand(batch_size, -1,
                                                                                                        -1)  # bsz x s1 x embed_dim
            else:
                p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_tokens.input_ids).expand(
                    batch_size, -1, -1)  # bsz x s1 x embed_dim
            inputs_embeds = ops.cat([bos_embeds, p_before_embeds, img_embeds, p_after_embeds], axis=1)  # bsz x (1+s1+1+s2) x embed_dim

            # create targets
            # empty_targets = (
            #     ops.ones([batch_size, 1 + p_before_embeds.size()[1] + 1],  # 1 (bos) + s1 + 1
            #                dtype=mindspore.int64).fill_(-100)
            # )  # bsz x (1 + s1)
            empty_targets = ops.fill(shape=(batch_size, 1 + p_before_embeds.shape[1]), type=mindspore.int64, value=-100)
            targets = ops.cat([empty_targets, target_ids], axis=1)  # bsz x (1 + s1 + 1 + s2)
            assert inputs_embeds.shape[1] == targets.shape[1]

            atts_prefix = ops.ones([batch_size, 1 + p_before_embeds.shape[1] + 1], dtype=mindspore.int64)  # bsz x (1 + s1 + 1)
            attention_mask = ops.cat([atts_prefix, attention_mask], axis=1)
            assert attention_mask.shape == targets.shape  # bsz x (1 + s1 + 1 + s2)
        else:
            p_before = '### Human: '
            p_before_tokens = self.llama_tokenizer(p_before, return_tensors="np", add_special_tokens=False)
            # peft model need deeper call
            if self.args['freeze_lm']:
                p_before_embeds = self.llama_model.model.embed_tokens(mindspore.Tensor(p_before_tokens.input_ids)).expand(batch_size, -1,
                                                                                                        -1)  # bsz x s1 x embed_dim
            else:
                p_before_embeds = self.llama_model.model.model.embed_tokens(mindspore.Tensor(p_before_tokens.input_ids)).expand(
                    batch_size, -1, -1)  # bsz x s1 x embed_dim
            inputs_embeds = ops.cat([bos_embeds, p_before_embeds, p_after_embeds], axis=1)  # bsz x (1+s1+s2) x embed_dim

            # create targets
            # empty_targets = (
            #     ops.ones([batch_size, 1 + p_before_embeds.size()[1]],  # 1 (bos) + s1
            #                dtype=mindspore.int64).fill_(-100)
            # )  # bsz x (1 + s1)
            empty_targets = ops.fill(shape=(batch_size, 1+p_before_embeds.shape[1]),type=mindspore.int64, value=-100)
            targets = ops.cat([empty_targets, target_ids], axis=1)  # bsz x (1 + s1 + s2)
            assert inputs_embeds.shape[1] == targets.shape[1]

            atts_prefix = ops.ones([batch_size, 1 + p_before_embeds.shape[1]], dtype=mindspore.int64)  # bsz x (1 + s1)
            attention_mask = ops.cat([atts_prefix, attention_mask], axis=1)
            assert attention_mask.shape == targets.shape  # bsz x (1 + s1 + s2)
        return inputs_embeds, targets, attention_mask

    def _train_with_mode(self, texts, img_embeds=None, modality='text', num_gen_tokens='8',
                         text_hidden_fcs=None, gen_token_idx=None, text_emb_layers=None, text_prompt_embeddins=None,
                         loss_scale=1.0, stage=2):
        """
        :param num_gen_tokens: the number of generation tokens
        :param modality: mode can be 'image' / 'video' / 'audio' / 'text'
        :param text_hidden_fcs: alignment module
        :param gen_token_idx: List
        :param text_emb_layers: the layer index of LLM hidden states
        :param text_prompt_embeddins: the textual caption/prompt embeddings
        :param loss_scale: the scale on the mse loss for alignment
        :param stage: the training stage
        :param
        """
        if stage == 2:
            input_ids, target_ids, attention_mask = process_batch_stage_2(self.llama_tokenizer, texts,
                                                                          self.max_length,
                                                                          num_gen_tokens,
                                                                          modality
                                                                          )
        elif stage == 3:
            input_ids, target_ids, attention_mask = process_batch_stage_3(self.llama_tokenizer, texts, self.max_length,
                                                                          self.args['num_gen_img_tokens'],
                                                                          self.args['num_gen_video_tokens'],
                                                                          self.args['num_gen_audio_tokens']
                                                                          )
        else:
            raise NotImplementedError
        inputs_embeds, targets, attention_mask = self.prompt_wrap(img_embeds, input_ids, target_ids, attention_mask)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = ops.max(outputs.logits, axis=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(mindspore.int64)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)

        if modality == 'text':
            return loss, gen_acc, ops.zeros_like(loss)
        else:
            hidden_states = []
            # text_hidden_fcs = self.gen_text_hidden_fcs

            # based on the targets to obtain the hidden state, targets includes the [BOS] token
            start_pos = (targets == gen_token_idx[0]).nonzero(as_tuple=False)[:, 1].tolist()
            end_pos = (targets == gen_token_idx[-1]).nonzero(as_tuple=False)[:, 1].tolist()
            # logging.info(f'targets : {targets}')
            # logging.info(f'start_pos : {start_pos}')
            # logging.info(f'end_pos : {end_pos}')
            assert 0 < len(start_pos) == len(end_pos) == input_ids.shape[0] and len(end_pos) > 0, (start_pos, end_pos)
            for idx, fc_layer in zip(text_emb_layers, text_hidden_fcs):
                hidden_embedding = []
                input_embedding = []
                for b, (s, e) in enumerate(zip(start_pos, end_pos)):
                    assert e - s + 1 == num_gen_tokens, (s, e)
                    hidden_embedding.append(outputs.hidden_states[idx][b, s:e + 1, :])
                    input_embedding.append(self.input_embeddings(targets[b, s:e + 1]))
                hidden_embedding = ops.stack(hidden_embedding, axis=0)
                input_embedding = ops.stack(input_embedding, axis=0)
                hidden_states.append(fc_layer(hidden_embedding, input_embedding))  # (N, seq_len, 2048)
            embeddings = ops.stack(hidden_states, axis=-1).sum(axis=-1)  # (N, 77, 768)
            # embeddings = embeddings / embeddings.norm(axis=-1, keepaxis=True)  # (N, T_I_V_A.txt, 256)

            # Obtain the embeddings produced by the text encoder of a frozen text-to-image generation model
            input_text = [conversation for conversation in texts]

            if modality == 'image':
                mse_loss = l2_loss(embeddings, ops.stack(text_prompt_embeddins, axis=0))
            elif modality == 'video':
                mse_loss = l2_loss(embeddings, ops.stack(text_prompt_embeddins, axis=0))
            else:
                text_prompt_embeddins = ops.stack(text_prompt_embeddins, axis=0)
                assert len(text_prompt_embeddins.shape) == 2, text_prompt_embeddins.shape
                text_prompt_embeddins = text_prompt_embeddins.view(text_prompt_embeddins.size(0), 1,
                                                                   text_prompt_embeddins.size(1))
                mse_loss = l2_loss(embeddings, text_prompt_embeddins)
            mse_loss = mse_loss.mean()
            loss += loss_scale * mse_loss

            return loss, gen_acc, mse_loss

    def _enc_align_training_stage_1(self, inputs):
        """
        In the stage 1: training the encoding-side alignment via image/video/audio caption tasks
        modality: the input modality for each caption task, it could be 'image', 'video' or 'audio'.
        """
        dataset_type = inputs['dataset_types']
        if dataset_type == 'ImageToText':
            image_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_image(image_paths)
        elif dataset_type == 'VideoToText':
            video_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_video(video_paths)
        elif dataset_type == 'AudioToText':
            audio_paths = inputs['mm_paths']
            mm_embeds, _ = self.encode_audio(audio_paths)
        else:
            raise NotImplementedError
        input_ids, target_ids, attention_mask = process_batch_stage_1(self.llama_tokenizer,
                                                                      inputs['output_texts'],
                                                                      self.max_length,
                                                                      self.args['prompt'])
        # print(input_ids)
        inputs_embeds, targets, attention_mask = self.prompt_wrap(mm_embeds, input_ids, target_ids, attention_mask)
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )

        loss = outputs.loss
        # calculate the token accuracy
        chosen_tokens = ops.max(outputs.logits, axis=-1)[1][:, 1:-1]  # [B, S-1]
        labels = targets[:, 2:]
        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(mindspore.int64)  # [B*S]
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask  # [B*S]
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)
        return loss, gen_acc

    def _dec_align_training_stage_2(self, inputs):
        """
        In the stage 2: training the decoding-side alignment via minimize the distance between the
        representation of signal tokens and caption from text encoder within the respective diffusion models.
        modality: the output modality for each caption.
        """
        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'TextToImage':
            loss, gen_acc, mse_loss = self._train_with_mode(texts=inputs['output_texts'],
                                                            modality='image',
                                                            num_gen_tokens=self.args['num_gen_img_tokens'],
                                                            text_hidden_fcs=self.gen_text_hidden_fcs,
                                                            gen_token_idx=self.args['gen_img_token_idx'],
                                                            text_emb_layers=self.args['text_emb_to_img_layers'],
                                                            text_prompt_embeddins=inputs['caption_embs'],
                                                            stage=self.stage)
        elif dataset_type == 'TextToVideo':
            loss, gen_acc, mse_loss = self._train_with_mode(texts=inputs['output_texts'],
                                                            modality='video',
                                                            num_gen_tokens=self.args['num_gen_video_tokens'],
                                                            text_hidden_fcs=self.gen_text_hidden_fcs_video,
                                                            gen_token_idx=self.args['gen_video_token_idx'],
                                                            text_emb_layers=self.args['text_emb_to_video_layers'],
                                                            text_prompt_embeddins=inputs['caption_embs'],
                                                            stage=self.stage)
        elif dataset_type == 'TextToAudio':
            loss, gen_acc, mse_loss = self._train_with_mode(texts=inputs['output_texts'],
                                                            modality='audio',
                                                            num_gen_tokens=self.args['num_gen_audio_tokens'],
                                                            text_hidden_fcs=self.gen_text_hidden_fcs_audio,
                                                            gen_token_idx=self.args['gen_audio_token_idx'],
                                                            text_emb_layers=self.args['text_emb_to_audio_layers'],
                                                            text_prompt_embeddins=inputs['caption_embs'],
                                                            stage=self.stage)
        else:
            raise NotImplementedError

        return loss, gen_acc, mse_loss

    def _instruction_tuning_stage_3(self, inputs):
        """
        In the stage 3: instruction-following training via the instruction dataset.
        """
        loss = 0
        gen_acc = 0
        mse_loss = []

        dataset_type = inputs['dataset_types'][0]
        if dataset_type == 'TextToImage':
            loss, gen_acc, mse_loss = self._train_with_mode(inputs['output_texts'], None, 'image',
                                                            self.args['num_gen_img_tokens'],
                                                            self.gen_text_hidden_fcs,
                                                            self.args['gen_img_token_idx'],
                                                            self.args['text_emb_to_img_layers'],
                                                            inputs['caption_embs'], stage=self.stage)
        elif dataset_type == 'TextToVideo':
            loss, gen_acc, mse_loss = self._train_with_mode(inputs['output_texts'], None, 'video',
                                                            self.args['num_gen_video_tokens'],
                                                            self.gen_text_hidden_fcs_video,
                                                            self.args['gen_video_token_idx'],
                                                            self.args['text_emb_to_video_layers'],
                                                            inputs['caption_embs'], loss_scale=2,
                                                            stage=self.stage)
        elif dataset_type == 'TextToAudio':
            loss, gen_acc, mse_loss = self._train_with_mode(inputs['output_texts'], None, 'audio',
                                                            self.args['num_gen_audio_tokens'],
                                                            self.gen_text_hidden_fcs_audio,
                                                            self.args['gen_audio_token_idx'],
                                                            self.args['text_emb_to_audio_layers'],
                                                            inputs['caption_embs'], stage=self.stage)
        elif dataset_type == 'ImageToText':
            image_paths = inputs['mm_paths']
            img_embeds, _ = self.encode_image(image_paths)
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], img_embeds, modality='text',
                                                     stage=self.stage)
        elif dataset_type == 'TextToText':
            loss, gen_acc, _ = self._train_with_mode(inputs['output_texts'], None, modality='text',
                                                     stage=self.stage)
        else:
            raise NotImplementedError
        return loss, gen_acc, mse_loss

    def _stage_4_training(self, inputs):
        """
        In the stage 4, we employ the modality-switch dataset to instruction-tune the overall framework
        """
        pass

    def construct(self, inputs):
        loss = 0
        gen_acc = 0
        mse_loss = None

        if self.stage == 1:
            loss, gen_acc = self._enc_align_training_stage_1(inputs)
        elif self.stage == 2:
            loss, gen_acc, mse_loss = self._dec_align_training_stage_2(inputs)
        elif self.stage == 3:
            loss, gen_acc, mse_loss = self._instruction_tuning_stage_3(inputs)
        else:
            raise NotImplementedError(f"stage {self.stage} is not implemented, now it only support [1, 2, 3]")

        return loss, gen_acc, mse_loss

    def extract_multimodal_feature(self, inputs):
        features = []
        if inputs['image_paths']:
            image_embeds, _ = self.encode_image(inputs['image_paths'])
            features.append(image_embeds)
        if inputs['audio_paths']:
            audio_embeds, _ = self.encode_audio(inputs['audio_paths'])
            features.append(audio_embeds)
        if inputs['video_paths']:
            video_embeds, _ = self.encode_video(inputs['video_paths'])
            features.append(video_embeds)

        feature_embeds = ops.cat(features).sum(axis=0).unsqueeze(0)
        return feature_embeds

    def _prepare_image_embed(self, text, batch_size):
        pattern = r'Image>(.*?)<\/Image'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='np')
        p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='np')
        if self.args['freeze_lm']:
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                   -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                 -1)  # bsz x s2 x embed_dim
        else:
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                         -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                       -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('image path: ', m)
            if m.startswith('temp'):
                m = os.path.join('./', m)
                print('image path: ', m)
            _temp_embedding, _ = self.encode_image([m])
            features.append(_temp_embedding)
        feature_embeds = ops.cat(features).sum(axis=0).unsqueeze(0)
        return ops.cat([p_before_embeds, feature_embeds, p_after_embeds], axis=1)

    def _prepare_video_embed(self, text, batch_size):
        pattern = r'Video>(.*?)<\/Video'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='np')
        p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='np')
        if self.args['freeze_lm']:
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                   -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                 -1)  # bsz x s2 x embed_dim
        else:
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                         -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                       -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('Video path: ', m)
            if m.startswith('temp'):
                m = os.path.join('./', m)
                print('Video path: ', m)
            _temp_embedding, _ = self.encode_video([m])
            features.append(_temp_embedding)
        feature_embeds = ops.cat(features).sum(axis=0).unsqueeze(0)
        return ops.cat([p_before_embeds, feature_embeds, p_after_embeds], axis=1)

    def _prepare_audio_embed(self, text, batch_size):
        pattern = r'Audio>(.*?)<\/Audio'
        matches = re.findall(pattern, text)
        features = []
        p_before_token = self.llama_tokenizer('<Img>', add_special_tokens=False, return_tensors='np')
        p_after_token = self.llama_tokenizer('</Img>', add_special_tokens=False, return_tensors='np')
        if self.args['freeze_lm']:
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                   -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                 -1)  # bsz x s2 x embed_dim
        else:
            p_before_embeds = self.llama_model.model.model.embed_tokens(p_before_token.input_ids).expand(batch_size, -1,
                                                                                                         -1)  # bsz x s1 x embed_dim
            p_after_embeds = self.llama_model.model.model.embed_tokens(p_after_token.input_ids).expand(batch_size, -1,
                                                                                                       -1)  # bsz x s2 x embed_dim
        for m in matches:
            print('Audio path: ', m)
            if m.startswith('temp'):
                m = os.path.join('./', m)
                print('Video path: ', m)
            _temp_embedding, _ = self.encode_audio([m])
            features.append(_temp_embedding)
        feature_embeds = ops.cat(features).sum(axis=0).unsqueeze(0)
        return ops.cat([p_before_embeds, feature_embeds, p_after_embeds], axis=1)

    def prepare_generation_embedding(self, inputs):
        prompt = inputs['prompt']
        text = prompt + '\n### Assistant:'
        print("text prompt: ", text)
        batch_size = 1
        input_embeds = []
        split_text = re.split(r' <|> ', text)
        for st in split_text:
            if st.startswith('Image>'):
                input_embeds.append(self._prepare_image_embed(st, batch_size))
            elif st.startswith('Audio>'):
                input_embeds.append(self._prepare_audio_embed(st, batch_size))
            elif st.startswith('Video>'):
                input_embeds.append(self._prepare_video_embed(st, batch_size))
            else:
                text_tokens = self.llama_tokenizer(st, add_special_tokens=False, return_tensors='np')
                text_tokens.input_ids = ops.cast(mindspore.Tensor(text_tokens.input_ids),dtype=mindspore.int64)
                bos = ops.ones([batch_size, 1],
                                 dtype=text_tokens.input_ids.dtype) * self.llama_tokenizer.bos_token_id  # bsz x 1
                if self.args['freeze_lm']:
                    text_embeds = self.llama_model.model.embed_tokens(text_tokens.input_ids).expand(batch_size, -1, -1)
                    bos_embeds = self.llama_model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
                else:
                    text_embeds = self.llama_model.model.model.embed_tokens(text_tokens.input_ids).expand(batch_size,
                                                                                                          -1, -1)
                    bos_embeds = self.llama_model.model.model.embed_tokens(bos)  # bsz x 1 x embed_dim
                input_embeds.append(bos_embeds)
                input_embeds.append(text_embeds)
        inputs_embeds = ops.cat(input_embeds, axis=1)  # bsz x (1+s2) x embed_dim
        return inputs_embeds

    def generate_tokens_embeddings(self, inputs, input_embeds, temperature: float = 0.0, top_p: float = 1.0):
        """
        This function is used to generate the tokens and output embeddings that employed to generate images/videos/audios
        inputs: dict
        input_embeds: tensor
        return:
            out: the output tokens index
            output_embeddings: output embeddings for synthesizing images
            video_output_embedding: output embeddings for synthesizing video
            audio_output_embedding: output embeddings for synthesizing audio
        """
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=inputs['stops_id'], encounters=1)])

        outputs = self.llama_model.generate(
            inputs_embeds=input_embeds,
            max_new_tokens=inputs['max_tgt_len'],
            top_p=inputs['top_p'],
            temperature=inputs['temperature'],
            # repeat_pen,
            do_sample=True,
            use_cache=True,
            stopping_criteria=stopping_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            output_attentions=True
        )

        output_embeddings = []
        video_output_embedding = []
        audio_output_embedding = []
        out = outputs.sequences
        for _hidden_states in outputs.hidden_states[1:]:
            for idx in self.args['text_emb_to_img_layers']:
                output_embeddings.append(_hidden_states[idx])
            for idx in self.args['text_emb_to_video_layers']:
                video_output_embedding.append(_hidden_states[idx])
            for idx in self.args['text_emb_to_audio_layers']:
                audio_output_embedding.append(_hidden_states[idx])
        output_embeddings = ops.cat(output_embeddings, axis=1)
        video_output_embedding = ops.cat(video_output_embedding, axis=1)
        audio_output_embedding = ops.cat(audio_output_embedding, axis=1)

        return out, output_embeddings, video_output_embedding, audio_output_embedding

    def generate_images(self, generated_ids, embeddings, all_gen_idx, generation_model=None,
                        guidance_scale=7.5, num_inference_steps=40):
        """
        To generate the images based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing images
        all_gen_idx: the index of [IMG0] in the generated_ids
        """
        last_ret_idx = 0
        return_outputs = []
        generation_model = StableDiffusionPipeline.from_pretrained(self.sd_ckpt_path)
        for gen_idx in all_gen_idx:
            # assert generated_ids[0,
            #        gen_idx:gen_idx + self.args['num_gen_img_tokens']].cpu().detach().numpy().tolist() == self.args[
            #            'gen_img_token_idx'], (
            #     generated_ids[0, gen_idx:gen_idx + self.args['num_gen_img_tokens']], self.args['gen_img_token_idx'])
            raw_emb = embeddings[:, gen_idx - 1:gen_idx - 1 + self.args['num_gen_img_tokens'], :]  # (1, 8, 4096)

            # Produce generation embedding.
            gen_prefix = ''.join([f'[IMG{i}]' for i in range(self.args['num_gen_img_tokens'])])
            gen_prefx_ids = self.llama_tokenizer(gen_prefix, add_special_tokens=False,
                                                 return_tensors="np").input_ids
            gen_prefix_embs = self.input_embeddings(mindspore.Tensor(gen_prefx_ids))  # (1, T_I_V_A.txt, D)
            gen_emb = self.gen_text_hidden_fcs[-1](raw_emb, gen_prefix_embs)  # (1, 77, 768)

            if gen_emb.shape[1] != 77:
                bs = gen_emb.shape[0]
                clip_emb = 768
                gen_emb = gen_emb.reshape(bs, -1, clip_emb)  # (bs, T_I_V_A.txt, 768)
                seq_len = gen_emb.shape[1]
                gen_emb = ops.cat([gen_emb, ops.zeros((bs, 77 - seq_len, clip_emb),
                                                          dtype=gen_emb.dtype)], axis=1)

            image_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps,
                                             return_dict=True).images

            caption = \
                self.llama_tokenizer.batch_decode(generated_ids[:, last_ret_idx:gen_idx], skip_special_tokens=True)[
                    0]
            last_ret_idx = gen_idx + 1
            return_outputs.append(caption + f' {gen_prefix}')
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(image_outputs)
        return return_outputs

    def generate_videos(self, generated_ids, embeddings, all_gen_idx, generation_model=None,
                        guidance_scale=7.5, num_inference_steps=40, height=320, width=576, num_frames=16):
        """
        To generate videos based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing videos
        all_gen_idx: the index of [VID0] in the generated_ids
        """
        return_outputs = []
        last_ret_idx = 0
        generation_model = TextToVideoSDPipeline.from_pretrained(self.vd_ckpt_path)
        for gen_idx in all_gen_idx:
            # assert generated_ids[0,
            #        gen_idx:gen_idx + self.args['num_gen_video_tokens']].cpu().detach().numpy().tolist() == \
            #        self.args[
            #            'gen_video_token_idx'], (
            #     generated_ids[0, gen_idx:gen_idx + self.args['num_gen_video_tokens']],
            #     self.args['gen_video_token_idx'])
            raw_emb = embeddings[:, gen_idx - 1:gen_idx - 1 + self.args['num_gen_video_tokens'], :]  # (1, 8, 4096)
            # print(f'gen_idx: {gen_idx}')
            # print('4', raw_emb.size())
            # assert len(self.args['text_emb_to_video_layers']) == 1

            # Produce generation embedding.
            gen_prefix = ''.join([f'[VID{i}]' for i in range(self.args['num_gen_video_tokens'])])
            gen_prefx_ids = self.llama_tokenizer(gen_prefix, add_special_tokens=False,
                                                 return_tensors="np").input_ids
            gen_prefix_embs = self.input_embeddings(mindspore.Tensor(gen_prefx_ids))  # (1, T_I_V_A.txt, D)
            gen_emb = self.gen_text_hidden_fcs_video[-1](raw_emb, gen_prefix_embs)  # (1, 77, 768)

            if gen_emb.shape[1] != 77:
                print(f"Padding {gen_emb.shape} with zeros")
                bs = gen_emb.shape[0]
                clip_emb = 768
                gen_emb = gen_emb.reshape(bs, -1, clip_emb)  # (bs, T_I_V_A.txt, 768)
                seq_len = gen_emb.shape[1]
                gen_emb = ops.cat([gen_emb, ops.zeros((bs, 77 - seq_len, clip_emb),
                                                          dtype=gen_emb.dtype)], axis=1)
                print('Padded to', gen_emb.shape)

            video_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps, height=height,
                                             width=width, num_frames=num_frames,return_dict=True).frames
            caption = \
                self.llama_tokenizer.batch_decode(generated_ids[:, last_ret_idx:gen_idx], skip_special_tokens=True)[
                    0]
            last_ret_idx = gen_idx + 1
            return_outputs.append(caption + f' {gen_prefix}')
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(video_outputs)
        return return_outputs

    def generate_audios(self, generated_ids, embeddings, all_gen_idx, generation_model=None,
                        guidance_scale=7.5, num_inference_steps=40, audio_length_in_s=5.0):
        """
        To generate videos based on the embeddings
        generated_ids: the  index of the generated tokens
        embedding: the embeddings for synthesizing audios
        all_gen_idx: the index of [AUD0] in the generated_ids
        """
        return_outputs = []
        last_ret_idx = 0
        generation_model = AudioLDMPipeline.from_pretrained(self.ad_ckpt_path)
        for gen_idx in all_gen_idx:
            # assert generated_ids[0,
            #        gen_idx:gen_idx + self.args['num_gen_audio_tokens']].cpu().detach().numpy().tolist() == \
            #        self.args[
            #            'gen_audio_token_idx'], (
            #     generated_ids[0, gen_idx:gen_idx + self.args['num_gen_audio_tokens']],
            #     self.args['gen_audio_token_idx'])
            raw_emb = embeddings[:, gen_idx - 1:gen_idx - 1 + self.args['num_gen_audio_tokens'], :]  # (1, 8, 4096)
            # print(f'gen_idx: {gen_idx}')
            # print('raw_emb 4', raw_emb.size())
            # assert len(self.args['text_emb_to_video_layers']) == 1

            # Produce generation embedding.
            gen_prefix = ''.join([f'[AUD{i}]' for i in range(self.args['num_gen_audio_tokens'])])
            gen_prefx_ids = self.llama_tokenizer(gen_prefix, add_special_tokens=False,
                                                 return_tensors="np").input_ids
            gen_prefix_embs = self.input_embeddings(mindspore.Tensor(gen_prefx_ids))  # (1, T_I_V_A.txt, D)
            gen_emb = self.gen_text_hidden_fcs_audio[-1](raw_emb, gen_prefix_embs)  # (1, 77, 768)
            # print('gen_emb size:', gen_emb.size())
            bs = gen_emb.shape[0]
            hid_emb_size = gen_emb.shape[2]
            gen_emb = gen_emb.view(bs, hid_emb_size)

            audio_outputs = generation_model(prompt_embeds=gen_emb,
                                             guidance_scale=guidance_scale,
                                             num_inference_steps=num_inference_steps,
                                             audio_length_in_s=audio_length_in_s,
                                             return_dict=True).audios[0]
            caption = \
                self.llama_tokenizer.batch_decode(generated_ids[:, last_ret_idx:gen_idx], skip_special_tokens=True)[
                    0]
            last_ret_idx = gen_idx + 1
            return_outputs.append(caption + f' {gen_prefix}')
            # return_outputs.append(truncate_caption(caption) + f' {gen_prefix}')
            return_outputs.append(audio_outputs)
        return return_outputs

    def generate(self, inputs):
        """
            inputs = {
                'image_paths': optional,
                'audio_paths': optional
                'video_paths': optional
                'thermal_paths': optional
                'mode': generation mode,
                'prompt': human input prompt,
                'max_tgt_len': generation length,
                'top_p': top_p,
                'temperature': temperature, Used to modulate logit distribution.
                'modality_embeds': None or torch.tensor,
                'modality_cache': save the image cache,

                'filter_value': Value to assign to tokens that should never be generated,
                'min_word_tokens': Minimum number of words to generate before allowing a [IMG] output.
                'gen_scale_factor': float = 1.0,
                'stops_id': the default value is [[835], [2277, 29937]] the stop token is '###', which has two types of tokenization ways, [835] and [2277, 29937]
                'ENCOUNTERS': the times that the generated sentence will be ended.

                'load_sd': whether use SD for image generation
                'max_num_imgs': Maximum number of images to return in one generation pass.
                'guidance_scale_for_img': the guidance ratio of conditioner, if it is None, the default value will be applied in SD
                'num_inference_steps_for_img': the number of inference step for image generation in the stable diffusion model

                'load_vd': whether use VD for video generation
                'max_num_vids': Maximum number of videos to return in one generation pass.
                'guidance_scale_for_vid': the guidance ratio of conditioner, if it is None, the default value will be applied in VD
                'num_inference_steps_for_vid': the number of inference step for video generation in the stable diffusion model
                'height': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The height in pixels of the generated video.
                'width': (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                        The width in pixels of the generated video.
                'num_frames': (`int`, *optional*, defaults to 16):
                        The number of video frames that are generated. Defaults to 16 frames which at 8 frames per seconds
                        amounts to 2 seconds of video.

                'load_ad': whether use AD for audio generation
                'max_num_auds': Maximum number of audios to return in one generation pass.
                'guidance_scale_for_aud': the guidance ratio of conditioner, if it is None, the default value will be applied in AD
                'num_inference_steps_for_aud': the number of inference step for audio generation in the stable diffusion model
                'audio_length_in_s': the seconds for generated audio length
            }
        """
        # init output with image tokens

        input_embeds = self.prepare_generation_embedding(inputs)
        generated_ids, generated_image_embeddings, generated_video_embeddings, generated_audio_embeddings = self.generate_tokens_embeddings(
            inputs, input_embeds)

        return_outputs = []

        # Find up to max_num_rets [IMG] tokens, and their corresponding scores.
        all_gen_img_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_img_token_idx'][0]) if x][
                          :inputs['max_num_imgs']]
        print('all_gen_img_idx: ', all_gen_img_idx)

        # Find up to max_num_rest [VID] tokens, and their corresponding scores.
        all_gen_vid_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_video_token_idx'][0]) if x][
                          :inputs['max_num_vids']]
        print('all_gen_vid_idx: ', all_gen_vid_idx)

        # Find up to max_num_rest [AUD] tokens, and their corresponding scores.
        all_gen_aud_idx = [i for i, x in enumerate(generated_ids[0, :] == self.args['gen_audio_token_idx'][0]) if x][
                          :inputs['max_num_auds']]
        print('all_gen_aud_idx: ', all_gen_aud_idx)

        if len(all_gen_img_idx) == 0 and len(all_gen_vid_idx) == 0 and len(all_gen_aud_idx) == 0:
            # No [IMG], [VID], [AUD] tokens.
            caption = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            # return_outputs.append(truncate_caption(caption))
            return_outputs.append(caption)
        else:
            if len(all_gen_img_idx) > 0:
                img_outputs = self.generate_images(generated_ids, generated_image_embeddings, all_gen_img_idx, None,
                                                   guidance_scale=inputs['guidance_scale_for_img'],
                                                   num_inference_steps=inputs['num_inference_steps_for_img'],
                                                   )
                return_outputs.append({'img': img_outputs})
            if len(all_gen_vid_idx) > 0:
                vid_outputs = self.generate_videos(generated_ids, generated_video_embeddings, all_gen_vid_idx, None,
                                                   guidance_scale=inputs['guidance_scale_for_vid'],
                                                   num_inference_steps=inputs['num_inference_steps_for_vid'],
                                                   height=inputs['height'], width=inputs['width'],
                                                   num_frames=inputs['num_frames'])
                return_outputs.append({'vid': vid_outputs})

            if len(all_gen_aud_idx) > 0:
                aud_outputs = self.generate_audios(generated_ids, generated_audio_embeddings, all_gen_aud_idx, None,
                                                   guidance_scale=inputs['guidance_scale_for_aud'],
                                                   num_inference_steps=inputs['num_inference_steps_for_aud'],
                                                   audio_length_in_s=inputs['audio_length_in_s'])
                return_outputs.append({'aud': aud_outputs})

        return return_outputs
