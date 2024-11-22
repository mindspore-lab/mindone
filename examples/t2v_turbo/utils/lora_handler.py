from types import SimpleNamespace

from mindspore import nn

from .lora import extract_lora_ups_down, inject_trainable_lora_extended, monkeypatch_or_replace_lora_extended

CLONE_OF_SIMO_KEYS = ["model", "loras", "target_replace_module", "r"]

lora_versions = dict(stable_lora="stable_lora", cloneofsimo="cloneofsimo")

lora_func_types = dict(loader="loader", injector="injector")

lora_args = dict(
    model=None,
    loras=None,
    target_replace_module=[],
    target_module=[],
    r=4,
    search_class=[nn.Dense],
    dropout=0,
    lora_bias="none",
)

LoraVersions = SimpleNamespace(**lora_versions)
LoraFuncTypes = SimpleNamespace(**lora_func_types)

LORA_VERSIONS = [LoraVersions.stable_lora, LoraVersions.cloneofsimo]
LORA_FUNC_TYPES = [LoraFuncTypes.loader, LoraFuncTypes.injector]


def filter_dict(_dict, keys=[]):
    if len(keys) == 0:
        assert "Keys cannot empty for filtering return dict."

    for k in keys:
        if k not in lora_args.keys():
            assert f"{k} does not exist in available LoRA arguments"

    return {k: v for k, v in _dict.items() if k in keys}


class LoraHandler(object):
    def __init__(
        self,
        version: str = LoraVersions.cloneofsimo,
        use_unet_lora: bool = False,
        use_text_lora: bool = False,
        save_for_webui: bool = False,
        only_for_webui: bool = False,
        lora_bias: str = "none",
        unet_replace_modules: list = ["UNet3DConditionModel"],
    ):
        self.version = version
        assert self.is_cloneofsimo_lora()

        self.lora_loader = self.get_lora_func(func_type=LoraFuncTypes.loader)
        self.lora_injector = self.get_lora_func(func_type=LoraFuncTypes.injector)
        self.lora_bias = lora_bias
        self.use_unet_lora = use_unet_lora
        self.use_text_lora = use_text_lora
        self.save_for_webui = save_for_webui
        self.only_for_webui = only_for_webui
        self.unet_replace_modules = unet_replace_modules
        self.use_lora = any([use_text_lora, use_unet_lora])

        if self.use_lora:
            print(f"Using LoRA Version: {self.version}")

    def is_cloneofsimo_lora(self):
        return self.version == LoraVersions.cloneofsimo

    def get_lora_func(self, func_type: str = LoraFuncTypes.loader):
        if func_type == LoraFuncTypes.loader:
            return monkeypatch_or_replace_lora_extended

        if func_type == LoraFuncTypes.injector:
            return inject_trainable_lora_extended

        assert "LoRA Version does not exist."

    def get_lora_func_args(self, lora_path, use_lora, model, replace_modules, r, dropout, lora_bias):
        return_dict = lora_args.copy()

        return_dict = filter_dict(return_dict, keys=CLONE_OF_SIMO_KEYS)
        return_dict.update(
            {
                "model": model,
                "loras": lora_path,
                "target_replace_module": replace_modules,
                "r": r,
            }
        )

        return return_dict

    def do_lora_injection(
        self,
        model,
        replace_modules,
        bias="none",
        dropout=0,
        r=4,
        lora_loader_args=None,
    ):
        REPLACE_MODULES = replace_modules

        params = None
        negation = None

        injector_args = lora_loader_args

        params, negation = self.lora_injector(**injector_args)
        for _up, _down in extract_lora_ups_down(model, target_replace_module=REPLACE_MODULES):
            if all(x is not None for x in [_up, _down]):
                print(f"Lora successfully injected into {model.__class__.__name__}.")

            break

        return params, negation

    def add_lora_to_model(self, use_lora, model, replace_modules, dropout=0.0, lora_path=None, r=16):
        params = None
        negation = None

        lora_loader_args = self.get_lora_func_args(
            lora_path, use_lora, model, replace_modules, r, dropout, self.lora_bias
        )

        if use_lora:
            params, negation = self.do_lora_injection(
                model,
                replace_modules,
                bias=self.lora_bias,
                lora_loader_args=lora_loader_args,
                dropout=dropout,
                r=r,
            )

        params = model if params is None else params
        return params, negation
