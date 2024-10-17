from share4v.transformers.models.clip import CLIPVisionModel
from transformers import CLIPImageProcessor, CLIPVisionConfig

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class CLIPVisionTower(nn.Cell):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args["mm_vision_select_layer"]
        self.select_feature = args.get("mm_vision_select_feature", "patch")
        self.model_path = args["mm_vision_tower_path"]
        self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        if not delay_load:
            self.load_model()

    def load_model(self):
        print(f"Load vision tower from {self.vision_tower_name}")
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        if "eva" in self.vision_tower_name.lower():
            raise NotImplementedError("Not support eva models")
        else:
            if self.model_path is not None:
                print("load vision tower using ms.load_param_into_net function")
                self.vision_tower = CLIPVisionModel(self.cfg_only)
                ms_vit_source_data = ms.load_checkpoint(self.model_path)
                # modify dict keys
                ms_vit_source_data = {
                    k.replace("vision_model", "vision_tower.vision_model"): v for k, v in ms_vit_source_data.items()
                }
                params_not_load = ms.load_param_into_net(self.vision_tower, ms_vit_source_data, strict_load=False)
                print(f"Params not loaded: {params_not_load}")
            else:
                print("load vision tower using CLIPVisionModel.from_pretrained function")
                self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        # freeze the vision tower
        self.set_train(False)
        print("vision tower set_train to False ")

        self.is_loaded = True

    def set_train(self, mode: bool):
        for param in self.vision_tower.get_parameters():
            param.requires_grad = mode
        return self

    def set_dtype(self, dtype):
        for param in self.vision_tower.get_parameters():
            param.set_dtype(dtype)

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs[2][self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def construct(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return ops.zeros(1, self.hidden_size, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
