import json
import math
from copy import deepcopy
from threading import Thread
from typing import List, Optional

from transformers import TextIteratorStreamer
from PIL import Image

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops

from ..qwen2 import Qwen2ForCausalLM, Qwen2PreTrainedModel
from .configuration_minicpm import MiniCPMVConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .processing_minicpmv import MiniCPMVProcessor
from .image_processing_minicpmv import MiniCPMVImageProcessor
from .resampler import Resampler
# from .tokenization_minicpmv_fast import MiniCPMVTokenizerFast

from mindspore import _no_grad

class MiniCPMVPreTrainedModel(Qwen2PreTrainedModel):
    config_class = MiniCPMVConfig


class MiniCPMV_v2_6(MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.llm = Qwen2ForCausalLM(config)
        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size
        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)
        self.processor = None

        self.terminators = ['<|im_end|>', '<|endoftext|>']

    def init_vision_module(self):
        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if self.config._attn_implementation == 'flash_attention_2':
            self.config.vision_config._attn_implementation = 'flash_attention_2'
        else:
            # not suport sdpa
            self.config.vision_config._attn_implementation = 'eager'
        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)

        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm

    def get_vllm_embedding(self, data):
        if 'vision_hidden_states' not in data:
            dtype = self.llm.model.embed_tokens.embedding_table.dtype
            device = None
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []
            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            # exist image
            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, ms.Tensor)]
                tgt_sizes = ops.vstack(tgt_sizes).astype(ms.int32)

                max_patches = ops.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])[0].asnumpy()

                # FIXME all_pixel_values
                # # all_pixel_values = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True,
                #                                                    padding_value=0.0)

                max_length_h = max([i.shape[0] for i in all_pixel_values])
                max_length_w = max([i.shape[1] for i in all_pixel_values])
                for i in range(len(all_pixel_values)):
                    if all_pixel_values[i].shape[0] < max_length_h or all_pixel_values[i].shape[1] < max_length_w:
                        all_pixel_values[i] = ops.pad(all_pixel_values[i], (0, max_length_w - all_pixel_values[i].shape[1], 0, max_length_h - all_pixel_values[i].shape[0]), value=0.0)
                all_pixel_values = ops.stack(all_pixel_values)

                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = ops.zeros(Tensor((B, 1, int(max_patches))), dtype=ms.bool_)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.astype(dtype)
                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(all_pixel_values[start_idx:end_idx], patch_attention_mask=patch_attn_mask[start_idx:end_idx], tgt_sizes=tgt_sizes[start_idx:end_idx]).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = ops.cat(hs, axis=0)
                else:
                    vision_embedding = self.vpm(all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes).last_hidden_state
                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    img_cnt = len(pixel_values)
                    if img_cnt > 0:
                        vision_hidden_states.append(vision_embedding[start: start + img_cnt])
                        start += img_cnt
                    else:
                        vision_hidden_states.append([])
            else: # no image
                if self.training:
                    dummy_image = ops.zeros(
                        (1, 3, 224, 224),
                        dtype=dtype
                    )
                    tgt_sizes = ms.Tensor([[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]).astype(ms.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)

        else:
            vision_hidden_states = data['vision_hidden_states']

        if hasattr(self.llm.config, 'scale_emb'):
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

        vision_hidden_states = [i.astype(vllm_embedding.dtype) if isinstance(
            i, ms.Tensor) else i for i in vision_hidden_states]

        # bs = len(data['input_ids'])
        # for i in range(bs):
        #     cur_vs_hs = vision_hidden_states[i]
        #     if len(cur_vs_hs) > 0:
        #         cur_vllm_emb = vllm_embedding[i]
        #         cur_image_bound = data['image_bound'][i]
        #         if len(cur_image_bound) > 0:
        #             image_indices = ops.stack(
        #                 [ops.arange(r[0], r[1], dtype=ms.int64) for r in cur_image_bound]
        #             )
        #
        #             cur_vllm_emb.scatter(0, image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
        #                                   cur_vs_hs.view(-1, cur_vs_hs.shape[-1]))
        #         elif self.training:
        #             cur_vllm_emb += cur_vs_hs[0].mean() * 0

        return vllm_embedding, vision_hidden_states

    def construct(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)
        position_ids = data["position_ids"]
        if position_ids.dtype != ms.int64:
            position_ids = position_ids.long()

        with _no_grad():
            return self.llm(
                input_ids=None,
                position_ids=position_ids,
                inputs_embeds=vllm_embedding,
                labels=data["labels"],
                **kwargs
            )

    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            **kwargs
        )
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()

        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        decode_text=False,
        **kwargs
    ):
        assert input_ids is not None
        assert len(input_ids) == len(pixel_values)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with ms._no_grad():
            (
                model_inputs["inputs_embeds"],
                vision_hidden_states,
            ) = self.get_vllm_embedding(model_inputs)

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
            else:
                result = self._decode(model_inputs["inputs_embeds"], tokenizer, attention_mask, decode_text=decode_text, **kwargs)

        if return_vision_hidden_states:
            return result, vision_hidden_states

        return result

    def chat(
        self,
        image,
        msgs,
        tokenizer,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=8192,
        system_prompt='',
        stream=False,
        max_slice_nums=None,
        use_image_id=None,
        **kwargs
    ):
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False
        msgs_list = msgs
        images_list = image

        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)
        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        image_processor = MiniCPMVImageProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)

        if processor is None:
            if self.processor is None:
                self.processor = MiniCPMVProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor

        assert self.config.query_num == processor.image_processor.image_feature_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.patch_size == processor.image_processor.patch_size, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.use_image_id == processor.image_processor.use_image_id, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."
        assert self.config.slice_mode == processor.image_processor.slice_mode, "These two values should be the same. Check `config.json` and `preprocessor_config.json`."

        prompts_lists = []
        input_images_lists = []
        for image, msgs in zip(images_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert sampling or not stream, "if use stream mode, make sure sampling=True"

            if image is not None and isinstance(copy_msgs[0]["content"], str):
                copy_msgs[0]["content"] = [image, copy_msgs[0]["content"]]

            images = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["user", "assistant"]
                if i == 0:
                    assert role == "user", "The role of first msg should be user"
                if isinstance(content, str):
                    content = [content]
                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                msg["content"] = "\n".join(cur_msgs)

            if system_prompt:
                sys_msg = {'role': 'system', 'content': system_prompt}
                copy_msgs = [sys_msg] + copy_msgs

            prompts_lists.append(processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True))
            input_images_lists.append(images)

        inputs = processor(
            prompts_lists,
            input_images_lists,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            return_tensors="ms",
            max_length=max_inp_length,
            image_processor=image_processor
        )

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config['min_new_tokens'] = min_new_tokens

        generation_config.update(
            (k, kwargs[k]) for k in generation_config.keys() & kwargs.keys()
        )

        inputs.pop("image_sizes")
        # with torch.inference_mode():
        res = self.generate(
            **inputs,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            vision_hidden_states=vision_hidden_states,
            stream=stream,
            decode_text=True,
            **generation_config
        )

        if stream:
            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, '')
                    yield text
            return stream_gen()

        else:
            if batched:
                answer = res
            else:
                answer = res[0]
            return answer
