# Tutorial: Qwen2-VL Implementation from Scratch <br> 从零开始实现 Qwen2-VL (MindSpore Version)
<!-- TODO: separate doc for CN ver. -->
<!-- [中文教程](tutorial_CN.md)  &nbsp;&nbsp;|&nbsp;&nbsp; English Tutorial -->

> This tutorial aims to re-implement Qwen2-VL based on [MindSpore](https://gitee.com/mindspore/mindspore) and [MindONE](https://github.com/mindspore-lab/mindone).
<br> 基于 [MindSpore](https://gitee.com/mindspore/mindspore) and [MindONE](https://github.com/mindspore-lab/mindone) 实现Qwen2-VL。

**Introduction:** Qwen2-VL is an advanced version of the [Qwen-VL](https://github.com/QwenLM/Qwen-VL) model, a large visual language model (LVLM). Key improvements include enhanced image comprehension, advanced video understanding, integrated visual agent functionality, and expanded multilingual support.
<br>
The model architecture has been optimized for handling arbitrary image resolutions through [Naive Dynamic Resolution](#naive-dynamic-resolution) support and utilizes [Multimodal Rotary Position Embedding (M-ROPE)](#multimodal-rotary-position-embedding-m-rope) to effectively process both 1D textual and multi-dimensional visual data. This updated model demonstrates competitive performance against leading AI systems like GPT-4o and Claude 3.5 Sonnet in vision-related tasks and ranks highly among open-source models in text capabilities. These advancements make Qwen2-VL a versatile tool for various applications requiring robust multimodal processing and reasoning abilities.

**Environment requirements:** Python, Mindspore, Mindone (todo), Transformers (latest)

## 1. Framework Overview 流程概览
Qwen2-VL adapted [ViT](https://github.com/google-research/vision_transformer#vision-transformer)'s encoder as Vision Encoder, and LLM [Qwen2](https://github.com/QwenLM/Qwen2)'s decoder as Decoder.

### Framework Overview 整体流程图

<div style="display: block; margin-left: auto;  margin-right: auto; width:80%" >
<img src="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/qwen2_vl.jpg" alt="Qwen2-VL Framework" width="100%" />

_Qwen2-VL architecture. Taken from [original paper](https://arxiv.org/abs/2409.12191)._
<br>Overall flow: Feeding multimodal inputs (vision and text) with M-ROPE into ViT visual encoder, LLM Qwen2 decode encoded input tokens and return textual reponses.

</div>

### Tasks
Qwen2-VL can support multimodal input for vision understanding.....

## 2. Model Architecture and Modules
### 2.1. Model Architecture
Here shows the components and architecture of the Qwen2-VL processor including image processor and tokenizer, as well as  the Qwen2-VL vision-and-text conditional generation model including vision encoder and LM decoder.

Taking "Qwen2-VL-7B-Instruct" as an example, Qwen2-VL contains the following components:


<details>
<summary>Chat template (Qwen's chatml format) </summary>
Convert the user input messsage into specific chatting template format (Qwen's chatml).

This step of chat template loading would be implemented by Processor's tokenizer.
For example:
```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "PATH",
                },
                {
                    "type": "text",
                    "text": "Describe this image."
                }],
        }]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# <|im_start|>system\n
# You are a helpful assistant.<|im_end|>\n
# <|im_start|>user\n
# <|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n
# <|im_start|>assistant\n
```
</details>

<details>
<summary>Image processor (Qwen2VLImageProcessor) </summary>
Preprocess input visual inputs into specific format that model can handle.
</details>

<details>
<summary>Model (visual: vision encoder; model: LM decoder)</summary>

- ``Qwen2VisionTransformerPretrainedModel``: vision model, with vision input embeddings
- ``Qwen2VLModel``: language model, generate response

```python
Qwen2VLForConditionalGeneration<
  (visual): Qwen2VisionTransformerPretrainedModel<
    (patch_embed): PatchEmbed<
      (proj): Conv3d<input_channels=3, output_channels=1280, kernel_size=(2, 14, 14), stride=(2, 14, 14), pad_mode=same, padding=0, dilation=(1, 1, 1), group=1, has_bias=False, weight_init=<mindspore.common.initializer.Normal object>, bias_init=None, format=NCDHW>
      >      >
    (rotary_pos_emb): VisionRotaryEmbedding<>
    (blocks): CellList<
    (0-31): 32 x Qwen2VLVisionBlock<
      (norm1): LayerNorm<normalized_shape=[1280], begin_norm_axis=-1, begin_params_axis=-1, gammaParameter (name=visual.blocks.XX.norm1.gamma, shape=(1280,), dtype=Float32, requires_grad=True), beta=Parameter (name=visual.blocks.XX.norm1.beta, shape=(1280,), dtype=Float32, requires_grad=True)>
      (norm2): LayerNorm<normalized_shape=[1280], begin_norm_axis=-1, begin_params_axis=-1, gammaParameter (name=visual.blocks.XX.norm2.gamma, shape=(1280,), dtype=Float32, requires_grad=True), beta=Parameter (name=visual.blocks.XX.norm2.beta, shape=(1280,), dtype=Float32, requires_grad=True)>
      (attn): VisionAttention<
        (qkv): Dense<input_channels=1280, output_channels=3840, has_bias=True>
        (proj): Dense<input_channels=1280, output_channels=1280, has_bias=True>
        >
      (mlp): VisionMlp<
        (fc1): Dense<input_channels=1280, output_channels=5120, has_bias=True>
        (act): QuickGELUActivation<>
        (fc2): Dense<input_channels=5120, output_channels=1280, has_bias=True>
        >
      >
    >
  (merger): PatchMerger<
    (ln_q): LayerNorm<normalized_shape=[1280], begin_norm_axis=-1, begin_params_axis=-1, gammaParameter (name=visual.merger.ln_q.gamma, shape=(1280,), dtype=Float32, requires_grad=True), beta=Parameter (name=visual.merger.ln_q.beta, shape=(1280,), dtype=Float32, requires_grad=True)>
    (mlp): SequentialCell<
      (0): Dense<input_channels=5120, output_channels=5120, has_bias=True>
      (1): GELU<>
      (2): Dense<input_channels=5120, output_channels=3584, has_bias=True>
      >
    >
  >
    >
  (model): Qwen2VLModel<
    (embed_tokens): Embedding<vocab_size=152064, embedding_size=3584, use_one_hot=False, embedding_table=Parameter (name=model.embed_tokens.embedding_table, shape=(152064, 3584), dtype=Float32, requires_grad=True), dtype=Float32, padding_idx=None>
    (layers): CellList<
      (0-27): 28 x Qwen2VLDecoderLayer<
        (self_attn): Qwen2VLAttention<
          (q_proj): Dense<input_channels=3584, output_channels=3584, has_bias=True>
          (k_proj): Dense<input_channels=3584, output_channels=512, has_bias=True>
          (v_proj): Dense<input_channels=3584, output_channels=512, has_bias=True>
          (o_proj): Dense<input_channels=3584, output_channels=3584>
          (rotary_emb): Qwen2VLRotaryEmbedding<>
          >
        (mlp): Qwen2MLP<
          (gate_proj): Dense<input_channels=3584, output_channels=18944>
          (up_proj): Dense<input_channels=3584, output_channels=18944>
          (down_proj): Dense<input_channels=18944, output_channels=3584>
          (act_fn): SiLU<>
          >
        (input_layernorm): Qwen2RMSNorm<>
        (post_attention_layernorm): Qwen2RMSNorm<>
        >
      >
    (norm): Qwen2RMSNorm<>
    (rotary_emb): Qwen2VLRotaryEmbedding<>
    >
  (lm_head): Dense<input_channels=3584, output_channels=152064>
  >
```
</details>


<details>
<summary>Processor (image processor; tokenizer)</summary>

- ``image_processor(Qwen2VLImageProcessor)``: handle input inputs
- ``tokenizer(Qwen2TokenizerFast)``: handle input text prompts, and placeholder vision tokens

```json
Qwen2VLProcessor:
- image_processor: Qwen2VLImageProcessor {
  "do_convert_rgb": true,
  "do_normalize": true,
  "do_rescale": true,
  "do_resize": true,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_processor_type": "Qwen2VLImageProcessor",
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "max_pixels": 1003520,
  "merge_size": 2,
  "min_pixels": 200704,
  "patch_size": 14,
  "processor_class": "Qwen2VLProcessor",
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "max_pixels": 12845056,
    "min_pixels": 3136
  },
  "temporal_patch_size": 2
}

- tokenizer: Qwen2TokenizerFast(name_or_path='Qwen/Qwen2-VL-7B-Instruct', vocab_size=151643, model_max_length=32768, is_fast=True, pa truncation_side='right', special_tokens={'eos_token': '<|im_end|>', 'pad_token': '<|endoftext|>', 'additional_special_tokens': ['<|iend|>', '<|object_ref_start|>', '<|object_ref_end|>', '<|box_start|>', '<|box_end|>', '<|quad_start|>', '<|quad_end|>', '<|vision_stand|>', '<|vision_pad|>', '<|image_pad|>', '<|video_pad|>']}, clean_up_tokenization_spaces=False),  added_tokens_decoder={
        151643: AddedToken("<|endoftext|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151644: AddedToken("<|im_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151645: AddedToken("<|im_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151646: AddedToken("<|object_ref_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151647: AddedToken("<|object_ref_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151648: AddedToken("<|box_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151649: AddedToken("<|box_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151650: AddedToken("<|quad_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151651: AddedToken("<|quad_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151652: AddedToken("<|vision_start|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151653: AddedToken("<|vision_end|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151654: AddedToken("<|vision_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151655: AddedToken("<|image_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
        151656: AddedToken("<|video_pad|>", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
}

{
  "processor_class": "Qwen2VLProcessor"
}
```
</details>

<br>

The Qwen2-VL model (Qwen2VLForConditionalGeneration) include a ViT encoder (Qwen2VisionTransformerPretrainedModel) and LM decoder (Qwen2VLModel).
<br>
Refer to [Position IDs](#position-ids) for details of ```get_rope_index()```.
<br>
Refer to [Visual Encoder](#22-visual-encoder) for detail of encoder
<br>
Refer to [LM decoder](#23-lm-decoder) for detail of decoder.
<br>
Refer to [IDs&tokens Generation](#input-ids-and-tokens-generation) for input IDs and tokens generation.
<br>
Running example in [Inference](#3-inference-pipelines).

```python
class Qwen2VLForConditionalGeneration():
    def __init__(self, config):
        super().__init__(config)
        self.visual = Qwen2VisionTransformerPretrainedModel._from_config(
            config.vision_config, attn_implementation=config._attn_implementation
        ) # Encoder
        self.model = Qwen2VLModel(config) # Decoder
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Dense(config.hidden_size, config.vocab_size, has_bias=False)
        self.padding_side = "left"

    ...

    # Hightlight! To get 3D ROPE index for multimodal input (image, video or text)
    def get_rope_index():  
        ...
        return position_ids, mrope_position_deltas
    def prepare_inputs_for_generation(...):
        ...
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs

   def construct(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds,
        labels, use_cache, output_attentions,  output_hidden_states, return_dict,
        pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw, rope_deltas,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:

        # Encode input
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                image_embeds = image_embeds.to(inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                video_embeds = video_embeds.to(inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        # Decode embeddings
        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # Training
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view((-1, self.config.vocab_size))
            shift_labels = shift_labels.view((-1))
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # TODO simplify
        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=rope_deltas,
        )
```


#### Input IDs and tokens Generation
Generate input IDs and tokens from inputs by tokenizer and generate().

Generate input IDs (``input_ids``), e.g.:
```python
# transformers/src/transformers/models/qwen2/tokenization_qwen2_fast.py
from transformers import Qwen2TokenizerFast
tokenizer = Qwen2TokenizerFast.from_pretrained("Qwen/Qwen-tokenizer")
tokenizer("Hello world")["input_ids"]
# [9707, 1879]
tokenizer(" Hello world")["input_ids"]
# [21927, 1879]
```

Text token generation (``generation_output``), e.g.: please refer to "mindone.transformers.generation" folder for more details.
```python
# mindone.transformers.MSGenerationMixin.generate()
from mindspore import Tensor
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
inputs = tokenizer("Hello world", return_tensors="np")
# {'input_ids': array([[9707, 1879]]), 'attention_mask': array([[1,1]])}
generation_output = model.generate(Tensor(inputs.input_ids)) # generated sequences of tokens
# Tensor(shape=[1, 20], dtype=Int64, value=[[9707, 1879, 0 ... 8405, 315, 279]])
```

### Naive Dynamic Resolution
Qwen2-VL can handle arbitrary image resolutions, mapping them into a dynamic number of visual tokens, offering a more human-like visual processing experience.

TODO: explain more

### Multimodal Rotary Position Embedding (M-ROPE)
Decomposes positional embedding into parts to capture 1D textual, 2D visual, and 3D video positional information, enhancing its multimodal processing capabilities.

![M-ROPE](https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/mrope.png)

#### Position IDs
Explanation:

    Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

    For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
    Examples:
        input_ids: [T T T T T], here T is for text.
        temporal position_ids: [0, 1, 2, 3, 4]
        height position_ids: [0, 1, 2, 3, 4]
        width position_ids: [0, 1, 2, 3, 4]

    For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
    and 1D rotary position embeddin for text part.
    Examples:
        Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
        input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
        vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        text temporal position_ids: [3, 4, 5, 6, 7]
        text height position_ids: [3, 4, 5, 6, 7]
        text width position_ids: [3, 4, 5, 6, 7]
        Here we calculate the text start position_ids as the max vision position_ids plus 1.

```python
class Qwen2VLForConditionalGeneration():
    ...
    # Hightlight! To get 3D ROPE index for multimodal input (image, video or text)
    def get_rope_index(
        self,
        input_ids: ms.Tensor,
        image_grid_thw: Optional[ms.Tensor] = None,
        video_grid_thw: Optional[ms.Tensor] = None,
        attention_mask: Optional[ms.Tensor] = None,
    ) -> Tuple[ms.Tensor, ms.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Args:
            input_ids (`ms.Tensor` of shape `(batch_size, sequence_length)`): Long
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`ms.Tensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`ms.Tensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`ms.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`ms.Tensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`ms.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if image_grid_thw is not None or video_grid_thw is not None: # For vision input
            total_input_ids = input_ids
            position_ids = ops.ones(
                (3, input_ids.shape[0], input_ids.shape[1]), dtype=input_ids.dtype
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                if attention_mask is not None:
                    input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = ops.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(ops.arange(text_len).view(1, -1).broadcast_to((3, -1)) + st_idx)

                    t_index = ops.arange(llm_grid_t).view((-1, 1)).broadcast_to((-1, llm_grid_h * llm_grid_w)).flatten()
                    h_index = ops.arange(llm_grid_h).view((1, -1, 1)).broadcast_to((llm_grid_t, -1, llm_grid_w)).flatten()
                    w_index = ops.arange(llm_grid_w).view((1, 1, -1)).broadcast_to((llm_grid_t, llm_grid_h, -1)).flatten()
                    llm_pos_ids_list.append(ops.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(ops.arange(text_len).view((1, -1)).broadcast_to((3, -1)) + st_idx)

                llm_positions = ops.cat(llm_pos_ids_list, axis=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = ms.Tensor(mrope_position_deltas).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.int().cumsum(-1) - 1
                position_ids = ops.masked_fill(position_ids.long(), attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).broadcast_to((3, -1, -1))
                max_position_ids = position_ids.max(0, keepdims=False)[0].max(-1, keepdims=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    ops.arange(input_ids.shape[1])
                    .view((1, 1, -1))
                    .broadcast_to((3, input_ids.shape[0], -1))
                )
                mrope_position_deltas = ops.zeros(
                    [input_ids.shape[0], 1],
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        rope_deltas = kwargs.get("rope_deltas", None)
        if attention_mask is not None and position_ids is None:
            if cache_position is None or (cache_position is not None and cache_position[0] == 0):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
            else:
                batch_size, seq_length = input_ids.shape
                delta = (
                    cache_position[0] + rope_deltas if cache_position is not None and rope_deltas is not None else 0
                )
                position_ids = ops.arange(seq_length)
                position_ids = position_ids.view((1, -1)).broadcast_to((batch_size, -1))
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).broadcast_to((3, -1, -1))

        if cache_position[0] != 0:
            pixel_values = None
            pixel_values_videos = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids, "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = inputs_embeds.shape  
            else:
                batch_size, sequence_length = input_ids.shape

            dtype = self.lm_head.weight.dtype
            min_dtype = dtype_to_min(dtype) # a function to get minimal value of the dtype

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "pixel_values_videos": pixel_values_videos,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "rope_deltas": rope_deltas,
            }
        )
        return model_inputs
```

<!-- #### Revisit: Rotary Position Embedding (1D-ROPE) -->


#### M-ROPE
```python
def _compute_default_rope_parameters(
    config = None
) -> Tuple["ms.Tensor", float]:
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Returns:
        Tuple of (`mindspore.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
    dim = int((config.hidden_size // config.num_attention_heads) * partial_rotary_factor)
    attention_factor = 1.0  # Unused in this type of RoPE
    # Compute the inverse frequencies
    inv_freq = 1.0 / (base ** (ops.arange(0, dim, 2, dtype=ms.int64).float() / dim))
    return inv_freq, attention_factor

# Hightlight !
class Qwen2VLRotaryEmbedding(nn.Cell):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        scaling_factor=1.0,
        rope_type="default",
        config = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        self.rope_type = rope_type
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config

        self.rope_init_fn = _compute_default_rope_parameters
        self.inv_freq, self.attention_scaling = self.rope_init_fn(self.config)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = ops.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            self.inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, seq_len=seq_len, **self.rope_kwargs
            )
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.inv_freq = self.original_inv_freq
            self.max_seq_len_cached = self.original_max_seq_len

    def construct(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids)

        # Core RoPE block. In contrast to other models, Qwen2_VL has different position ids for thw grids
        # So we expand the inv_freq to shape (3, ...)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().broadcast_to((3, position_ids.shape[1], -1, 1))
        position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).swapaxes(2, 3)
        emb = ops.cat((freqs, freqs), axis=-1)
        cos = emb.cos()
        sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return ops.cat((-x2, x1), axis=-1)

def apply_rotary_pos_emb_vision(tensor: ms.Tensor, freqs: ms.Tensor) -> ms.Tensor:
    orig_dtype = tensor.dtype
    tensor = tensor.float()
    cos = freqs.cos()
    sin = freqs.sin()
    cos = cos.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).unsqueeze(0).float()
    output = (tensor * cos) + (rotate_half(tensor) * sin)
    output = output.to(orig_dtype)
    return output
```

```python
def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section=[16,24,24], unsqueeze_dim=1):
    mrope_section = mrope_section * 2
    cos = ops.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, axis=-1))], axis=-1).unsqueeze(unsqueeze_dim)
    sin = ops.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, axis=-1))], axis=-1).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Qwen2VisionTransformerPretrainedModel:
    ...
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw:
            hpos_ids = ops.arange(h).unsqueeze(1).broadcast_to((-1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()

            wpos_ids = ops.arange(w).unsqueeze(0).broadcast_to((h, -1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(ops.stack([hpos_ids, wpos_ids], axis=-1).repeat(t, 1))
        pos_ids = ops.cat(pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size)
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb
```

### 2.2. Visual Encoder
<!-- #### Revisit Vision Transformer (ViT)

<div style="display: block; margin-left: auto;  margin-right: auto; width:80%" >
<img src="https://github.com/google-research/vision_transformer/raw/main/vit_figure.png" alt="ViT architecture" width="100%"/>

<br> _ViT architecture. Taken from [original paper](https://arxiv.org/abs/2010.11929)._
</div>

```python
To supplement
``` -->

#### Qwen2-VL ViT
Architecture: TODO: descrption + figure
- Conv3D => RotaryEmbedding => ViTEnocder

TODO: simplify code
```python
import mindspore as ms
from mindspore import nn

class Qwen2VisionTransformerPretrainedModel():

    def __init__(self, config) -> None:
        super().__init__(config)
        self.spatial_merge_size = config.spatial_merge_size

        self.patch_embed = PatchEmbed(
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        head_dim = config.embed_dim // config.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.CellList(
            [Qwen2VLVisionBlock(config, config._attn_implementation) for _ in range(config.depth)]
        )
        self.merger = PatchMerger(
            dim=config.hidden_size, context_dim=config.embed_dim, spatial_merge_size=config.spatial_merge_size
        )

    def rot_pos_emb(self, grid_thw):
        ...

    def construct(self, hidden_states: ms.Tensor, grid_thw: ms.Tensor) -> ms.Tensor:
        '''
            grid_thw: input position ids for (time, height, width) dimensions
        '''
        hidden_states = self.patch_embed(hidden_states)
        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        cu_seqlens = ops.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(axis=0, dtype=ms.int32)
        cu_seqlens = ops.pad(cu_seqlens, (1, 0), value=None)
        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

        return self.merger(hidden_states)
```
```python
# Conv3D, depth=2 for 2 frames
class PatchEmbed(nn.Cell):
    def __init__(self, patch_size: int = 14, temporal_patch_size: int = 2, in_channels: int = 3, embed_dim: int = 1152) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        kernel_size = (temporal_patch_size, patch_size, patch_size)
        self.proj = nn.Conv3d(in_channels, embed_dim, kernel_size=kernel_size, stride=kernel_size, has_bias=False)

    def construct(self, hidden_states: ms.Tensor) -> ms.Tensor:
        hidden_states = hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        hidden_states = self.proj(hidden_states).view(-1, self.embed_dim)
        return hidden_states

# Original Rotary Embedding
class VisionRotaryEmbedding(nn.Cell):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (theta ** (ops.arange(0, dim, 2, dtype=ms.float32) / dim))

    def construct(self, seqlen: int) -> ms.Tensor:
        seq = ops.arange(seqlen, dtype=self.inv_freq.dtype)
        freqs = ops.outer(seq, self.inv_freq)
        return freqs

class Qwen2VisionTransformerPretrainedModel:
    ...
    def rot_pos_emb(self, grid_thw):
        pos_ids = []
        for t, h, w in grid_thw: # 3D IDs
            hpos_ids = ops.arange(h).unsqueeze(1).broadcast_to((-1, w))
            hpos_ids = hpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            hpos_ids = hpos_ids.permute(0, 2, 1, 3)
            hpos_ids = hpos_ids.flatten()
            wpos_ids = ops.arange(w).unsqueeze(0).broadcast_to((h, -1))
            wpos_ids = wpos_ids.reshape(
                h // self.spatial_merge_size,
                self.spatial_merge_size,
                w // self.spatial_merge_size,
                self.spatial_merge_size,
            )
            wpos_ids = wpos_ids.permute(0, 2, 1, 3)
            wpos_ids = wpos_ids.flatten()
            pos_ids.append(ops.stack([hpos_ids, wpos_ids], axis=-1).repeat(t, 1))
        pos_ids = ops.cat(pos_ids, axis=0)
        max_grid_size = grid_thw[:, 1:].max()
        rotary_pos_emb_full = self.rotary_pos_emb(max_grid_size) #i.e.,VisionRotaryEmbedding
        rotary_pos_emb = rotary_pos_emb_full[pos_ids].flatten(1)
        return rotary_pos_emb
```

```python
class Qwen2VLAttention(nn.Cell):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout
        self.rope_scaling = config.rope_scaling

        self.q_proj = nn.Dense(self.hidden_size, self.num_heads * self.head_dim, has_bias=True)
        self.k_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True)
        self.v_proj = nn.Dense(self.hidden_size, self.num_key_value_heads * self.head_dim, has_bias=True)
        self.o_proj = nn.Dense(self.num_heads * self.head_dim, self.hidden_size, has_bias=False)

        self.rotary_emb = Qwen2VLRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def construct(
        self,
        hidden_states,
        attention_mask = None,
        position_ids = None,
        past_key_value = None,
        output_attentions = False,
        use_cache = False,
        cache_position = None,
        position_embeddings: = None,  
    ) -> Tuple[ms.Tensor, Optional[ms.Tensor], Optional[Tuple[ms.Tensor]]]:
        bsz, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).swapaxes(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).swapaxes(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += cache_position[0] + 1

        cos, sin = position_embeddings
        query_states, key_states = apply_multimodal_rotary_pos_emb(
            query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
        )

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = ops.matmul(query_states, key_states.swapaxes(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Fix precision issues in Qwen2-VL float16 inference
        # Replace inf values with zeros in attention weights to prevent NaN propagation
        if query_states.dtype == ms.float16:
            attn_weights = ops.where(ops.isinf(attn_weights), ops.zeros_like(attn_weights), attn_weights)

        # upcast attention to fp32
        attn_weights = ops.softmax(attn_weights, axis=-1).to(query_states.dtype)
        attn_weights = ops.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = ops.matmul(attn_weights, value_states)
        attn_output = attn_output.swapaxes(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

def scaled_dot_product_attention():
    ...

class VisionSdpaAttention(nn.Cell):
    def __init__(self, dim: int, num_heads: int = 16) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def construct(
        self, hidden_states: ms.Tensor, cu_seqlens: ms.Tensor, rotary_pos_emb: ms.Tensor = None
    ) -> ms.Tensor:
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary_pos_emb).squeeze(0)

        attention_mask = ops.zeros([1, seq_length, seq_length], dtype=ms.bool)
        for i in range(1, len(cu_seqlens)):
            attention_mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = True
        q = q.swapaxes(0, 1)
        k = k.swapaxes(0, 1)
        v = v.swapaxes(0, 1)
        attn_output = scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.swapaxes(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        attn_output = self.proj(attn_output)
        return attn_output

class VisionMlp(nn.Cell):
    def __init__(self, dim: int, hidden_dim: int, hidden_act: str) -> None:
        super().__init__()
        self.fc1 = nn.Dense(dim, hidden_dim)
        self.act = nn.ACT2FN[hidden_act]
        self.fc2 = nn.Dense(hidden_dim, dim)
    def construct(self, x) -> ms.Tensor:
        return self.fc2(self.act(self.fc1(x)))

class Qwen2VLVisionBlock(nn.Cell):
    def __init__(self, config) -> None:
        super().__init__()
        self.norm1 = LayerNorm((config.embed_dim,), epsilon=1e-6)
        self.norm2 = LayerNorm((config.embed_dim,), epsilon=1e-6)
        mlp_hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.attn = VisionSdpaAttention(config.embed_dim, num_heads=config.num_heads)
        self.mlp = VisionMlp(dim=config.embed_dim, hidden_dim=mlp_hidden_dim, hidden_act=config.hidden_act)

    def construct(self, hidden_states, cu_seqlens, rotary_pos_emb) -> ms.Tensor:
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states), cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states
```
```python
class PatchMerger(nn.Cell):
    def __init__(self, dim: int, context_dim: int, spatial_merge_size: int = 2) -> None:
        super().__init__()
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = LayerNorm((context_dim,), epsilon=1e-6)
        self.mlp = nn.SequentialCell(
            nn.Dense(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Dense(self.hidden_size, dim),
        )
    def construct(self, x: ms.Tensor) -> ms.Tensor:
        x = self.mlp(self.ln_q(x).view(-1, self.hidden_size))
        return x
```


### 2.3. LM Decoder
```python
class Qwen2VLModel(Qwen2VLPreTrainedModel):
    def __init__(self, config: Qwen2VLConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Embedding
        self.embed_tokens = nn.Embedding(vocab_size=config.vocab_size, embedding_size=config.hidden_size, padding_idx=self.padding_idx)
        self.layers = nn.CellList(
            [Qwen2VLDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2VLRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

    def construct(
        self,
        input_ids: ms.Tensor = None,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_values: Optional[List[ms.Tensor]] = None,
        inputs_embeds: Optional[ms.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[ms.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = ops.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )
        # the hard coded `3` is for temporal, height and width.
        if position_ids is None:
            position_ids = cache_position.view((1, 1, -1)).broadcast_to((3, inputs_embeds.shape[0], -1))
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].broadcast_to((3, position_ids.shape[0], -1))

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

```
```python
# Copied from transformers.models.qwen2.modeling_qwen2.Qwen2MLP
class Qwen2MLP(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.up_proj = nn.Dense(self.hidden_size, self.intermediate_size, has_bias=False)
        self.down_proj = nn.Dense(self.intermediate_size, self.hidden_size, has_bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def construct(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class Qwen2VLDecoderLayer(nn.Cell):
    def __init__(self, config: Qwen2VLConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2VLSdpaAttention(config, layer_idx)
        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def construct(
        self,
        hidden_states: ms.Tensor,
        attention_mask: Optional[ms.Tensor] = None,
        position_ids: Optional[ms.Tensor] = None,
        past_key_value: Optional[Tuple[ms.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[ms.Tensor] = None,
        position_embeddings: Optional[Tuple[ms.Tensor, ms.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[ms.Tensor, Optional[Tuple[ms.Tensor, ms.Tensor]]]:

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
```

### 2.4. [Optional] Vision Preprocessing
Convert all images or video frames into a list of Pillow Image.
Can downsample total frames by setting smaller FPS (i.e., 'fps' in config).
For example, for a video with 24FPS, total 1200 frame, with new 1 FPS. We can get a list of

Refer to qwen_vl_utils.


## 3. Inference Pipelines
### Single Media inference
The model can accept both images and videos as input.  Here is an example without using qwen_vl_utils.

```python
from PIL import Image
import requests
import mindspore
from mindspore import Tensor
from typing import Dict
from transformers import AutoProcessor
from mindone.transformers import Qwen2VLForConditionalGeneration

# Load the model and processor
model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct") # Qwen2VLProcessor


#################################
# Single Image

# Input
messages = [
    {
        "role":"user",
        "content":[
            {
                "type":"image",
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {
                "type":"text",
                "text":"Describe this image."
            }
        ]
    }
]

# Prepraration for inference
text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# Excepted output:
# '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n'
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
image_inputs = [Image.open(requests.get(url, stream=True).raw)]
video_inputs = None
inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="np")

# Inference: Generation of the output
output_ids = model.generate(Tensor(inputs.input_ids), max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text[0])
# Expected output:
# 'The image depicts a serene beach scene at sunset. A woman and her dog are sitting on the sand, enjoying each other's company. The woman is wearing a plaid shirt and dark pants, and she is sitting cross-legged with her dog. The dog, which appears to be a large breed, is wearing a colorful harness and is giving a high-five to the woman. The beach is relatively empty, with gentle waves in the background. The lighting is warm and golden, indicating that the photo was taken during the golden hour, just before sunset. The overall atmosphere of the image is peaceful and joyful.'

######################################################################################
# Single Video

from mindspore.dataset import vision
import numpy as np

def fetch_video(ele: Dict, nframe_factor=2):
    if isinstance(ele['video'], str):
        def round_by_factor(number: int, factor: int) -> int:
            return round(number / factor) * factor

        video_path = ele["video"]
        if video_path.startswith("file://"):
            video_path = video_path[7:]
        video, audio, info = vision.read_video(
            video_path,
            start_pts=ele.get("video_start", 0.0),
            end_pts=ele.get("video_end", None),
            pts_unit="sec",
        )
        assert not ("fps" in ele and "nframes" in ele), "Only accept either `fps` or `nframes`"
        if "nframes" in ele:
            nframes = round_by_factor(ele["nframes"], nframe_factor)
        else:
            fps = ele.get("fps", 1.0)
            nframes = round_by_factor(video.size(0) / info["video_fps"] * fps, nframe_factor)

        idx = np.linspace(0, total_frames - 1, nframes).round().astype(np.int32)
        video = video[idx] #video in [T, H, W, C] numpy

        ## If need resizing
        # images = [
        #     vision.Resize(
        #         size=[resized_height, resized_width],
        #         interpolation=Inter.BICUBIC,
        #     )(image).astype(np.uint8)
        #     for image in video
        # ]
        images = [Image.fromarray(image) for image in images]

        return images # list of Pillow Image


messages = [
    {
        "role": "user",
        "content": [
            # {"type": "video", "video": "/path/to/video.mp4", "fps": 1.0}
            {"type": "text", "text": "Describe the video."},
        ],
    }
]
video_info = {"type": "video", "video": "/path/to/video.mp4", "fps": 1.0}
video_inputs = fetch_video(video_info)
image_inputs = None
inputs = processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="np")
output_ids = model.generate(Tensor(inputs.input_ids), max_new_tokens=128)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text[0])

```

### Multiple Media inference
TODO: add another example

## 4. Finetune
TODO
