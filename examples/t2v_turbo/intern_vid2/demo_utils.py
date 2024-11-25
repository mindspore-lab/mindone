import gc
import os

import cv2
import numpy as np
from intern_vid2.models.backbones.bert.builder import build_bert
from intern_vid2.models.backbones.bert.tokenization_bert import BertTokenizer
from intern_vid2.models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224
from intern_vid2.models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new

import mindspore as ms
from mindspore import mint, nn

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def frames2tensor(vid_list, fnum=8, target_size=(224, 224)):
    assert len(vid_list) >= fnum
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = ms.Tensor.from_numpy(vid_tube).float()
    return vid_tube


def get_text_feat_dict(texts, clip, text_feat_d={}):
    for t in texts:
        feat = clip.get_txt_feat(t)
        text_feat_d[t] = feat
    return text_feat_d


def get_vid_feat(frames, vlm):
    return vlm.get_vid_features(frames)


def retrieve_text(frames, texts, model, topk: int = 5, config: dict = {}):
    vlm = model

    fn = config.get("num_frames", 8)
    size_t = config.get("size_t", 224)
    frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t))
    vid_feat = vlm.get_vid_feat(frames_tensor)

    text_feat_d = {}
    text_feat_d = get_text_feat_dict(texts, vlm, text_feat_d)
    text_feats = [text_feat_d[t] for t in texts]
    text_feats_tensor = mint.cat(text_feats, 0)

    probs, idxs = vlm.predict_label(vid_feat, text_feats_tensor, top=topk)

    ret_texts = [texts[i] for i in idxs.long().numpy()[0].tolist()]
    return ret_texts, probs.float().numpy()[0]


def setup_internvideo2(config: dict, dtype=ms.float32):
    if "bert" in config.model.text_encoder.name:
        tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained)
        model = InternVideo2_Stage2(
            config=config,
            tokenizer=tokenizer,
            is_pretrain=False,
            dtype=dtype,
        )
    else:
        model = InternVideo2_Stage2(config=config, is_pretrain=False, dtype=dtype)
        tokenizer = model.tokenizer

    # if config.get("compile_model", False):
    #     ms.set_float32_matmul_precision("high")
    #     model = ms.compile(model)

    model_without_ddp = model

    if config.pretrained_path.strip() and (os.path.isfile(config.pretrained_path)) or "s3://" in config.pretrained_path:
        state_dict = ms.load_checkpoint(config.pretrained_path)

        text_encoder_state_dict = {}
        vision_proj_state_dict = {}
        text_proj_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("text_encoder.bert"):
                text_encoder_state_dict["text_encoder." + k[len("text_encoder.bert.") :]] = v
            elif k.startswith("vision_proj."):
                vision_proj_state_dict[k] = v
            elif k.startswith("text_proj."):
                text_proj_state_dict[k] = v

        ms.load_param_into_net(model_without_ddp.text_encoder, text_encoder_state_dict)
        ms.load_param_into_net(model_without_ddp.vision_proj, vision_proj_state_dict)
        ms.load_param_into_net(model_without_ddp.text_proj, text_proj_state_dict)

        if config.get("origin_num_frames", None) is not None:
            a = len(state_dict)
            interpolate_pos_embed_internvideo2_new(
                state_dict,
                model_without_ddp.vision_encoder,
                orig_t_size=config.origin_num_frames,
            )
            assert a == len(state_dict), state_dict.keys()

        del state_dict
        gc.collect()

    if config.get("use_bf16", False):
        model_without_ddp = model_without_ddp.to_float(ms.bfloat16)
    elif config.get("use_half_precision", False):
        model_without_ddp = model_without_ddp.to_float(ms.float16)
    else:
        model_without_ddp = model_without_ddp.to_float(ms.float32)

    return (
        model_without_ddp,
        tokenizer,
    )


class InternVideo2_Stage2(nn.Cell):
    """docstring for InternVideo2_Stage2"""

    def __init__(self, config, tokenizer, is_pretrain: bool = True, dtype=ms.float32):
        super(InternVideo2_Stage2, self).__init__()

        self.config = config
        self.tokenizer = tokenizer

        self.is_pretrain = is_pretrain
        self.vision_width = config.model.vision_encoder.clip_embed_dim
        self.text_width = config.model.text_encoder.d_model
        self.embed_dim = config.model.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder(dtype=dtype)
        self.freeze_vision()

        self.text_encoder = self.build_text_encoder(dtype=dtype)
        self.freeze_text()

        self.vision_proj = nn.Dense(self.vision_width, self.embed_dim)
        self.text_proj = nn.Dense(self.text_width, self.embed_dim)

    def freeze_vision(self):
        """freeze vision encoder"""
        for p in self.vision_encoder.get_parameters():
            p.requires_grad = False

    def freeze_text(self):
        """freeze text encoder"""
        for p in self.text_encoder.get_parameters():
            p.requires_grad = False

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    def encode_vision(self, image: ms.Tensor, test: bool = False):
        """encode image / videos as features.

        Args:
            image (ms.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (ms.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (ms.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (ms.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (ms.Tensor): The features of clip. Shape: [K,B,N,C].

        """

        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        # keep_temporal=self.config.model.vision_encoder.keep_temporal
        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(image, None, use_image)
            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(image)
            # if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
            #     keep_temporal = False
            # print(f"\033[31mmask is {type(mask)}\033[0m")
            (
                vision_embeds,
                pooled_vision_embeds,
                student_output,
                student_output_final,
            ) = self.vision_encoder(image, mask, use_image)
            return (
                vision_embeds,
                pooled_vision_embeds,
                student_output,
                student_output_final,
                targets_clip_middle_vis,
                targets_clip_final_vis,
            )

    def encode_text(self, text: dict):
        """encode text.
        Args:
            text (dict): The output of huggingface's `PreTrainedTokenizer`. contains keys:
                - input_ids (ms.Tensor): Token ids to be fed to a model. Shape: [B,L].
                - attention_mask (ms.Tensor): The mask indicate padded tokens. Shape: [B,L]. 0 is padded token.
                - other keys refer to "https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__".
        Returns: tuple.
            - text_embeds (ms.Tensor): The features of all tokens. Shape: [B,L,C].
            - pooled_text_embeds (ms.Tensor): The pooled features. Shape: [B,C].

        """
        encoder = self.get_text_encoder()
        text_output = encoder(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            return_dict=True,
            mode="text",
        )
        text_embeds = text_output.last_hidden_state
        pooled_text_embeds = text_embeds[:, 0]
        return text_embeds, pooled_text_embeds

    def build_vision_encoder(self, dtype):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Cell`.

        """
        encoder_name = self.config.model.vision_encoder.name

        if encoder_name == "pretrain_internvideo2_1b_patch14_224":
            vision_encoder = pretrain_internvideo2_1b_patch14_224(self.config.model)
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        # parameters for mask
        img_size = self.config.model.vision_encoder.img_size
        num_frames = self.config.model.vision_encoder.num_frames
        tublet_size = self.config.model.vision_encoder.tubelet_size
        patch_size = self.config.model.vision_encoder.patch_size
        self.clip_img_size = self.config.model.vision_encoder.clip_input_resolution
        self.video_mask_type = self.config.model.vision_encoder.video_mask_type
        self.video_window_size = (
            num_frames // tublet_size,
            img_size // patch_size,
            img_size // patch_size,
        )
        self.video_mask_ratio = self.config.model.vision_encoder.video_mask_ratio
        self.image_mask_type = self.config.model.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.model.vision_encoder.image_mask_ratio

        return vision_encoder

    def build_text_encoder(self, dtype):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Cell. The text encoder

        """
        encoder_name = self.config.model.text_encoder.name

        if "bert" in encoder_name:
            text_encoder = build_bert(
                self.config.model,
                self.is_pretrain,
                self.config.gradient_checkpointing,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        return text_encoder

    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder

    def get_vid_feat(self, frames: ms.Tensor):
        """get the video features for the given frames.

        Args:
            frames (ms.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (ms.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (ms.Tensor): The pooled output features. Shape: [B,1,C].

        """
        with ms._no_grad():
            _, vfeat = self.encode_vision(frames, test=True)
            vfeat = self.vision_proj(vfeat)
            vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vfeat

    def get_vid_feat_with_grad(self, frames: ms.Tensor):
        """get the video features for the given frames with grad.

        Args:
            frames (ms.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (ms.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (ms.Tensor): The pooled output features. Shape: [B,1,C].

        """
        _, vfeat = self.encode_vision(frames, test=True)
        vfeat = self.vision_proj(vfeat)
        vfeat = vfeat / vfeat.norm(dim=-1, keepdim=True)
        return vfeat

    def get_txt_feat(self, text: str):
        """get the text features for the given text."""
        with ms._no_grad():
            text = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.config.max_txt_l,
            )
            _, tfeat = self.encode_text(text)
            tfeat = self.text_proj(tfeat)
            tfeat /= tfeat.norm(dim=-1, keepdim=True)
        return tfeat

    def predict_label(self, vid_feat: ms.Tensor, txt_feat: ms.Tensor, top: int = 5):
        label_probs = (100.0 * vid_feat @ txt_feat.T).softmax(dim=-1)
        top_probs, top_labels = label_probs.float().topk(top, dim=-1)
        return top_probs, top_labels
