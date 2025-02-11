import os

from hydit.config import get_args
from hydit.inference import _to_tuple
from hydit.modules.models import HUNYUAN_DIT_MODELS

import mindspore as ms

args = get_args()

image_size = _to_tuple(args.image_size)
latent_size = (image_size[0] // 8, image_size[1] // 8)

model = HUNYUAN_DIT_MODELS[args.model](
    args,
    input_size=latent_size,
    log_fn=print,
)
model_path = os.path.join(args.model_root, "t2i", "model", f"pytorch_model_{args.load_key}.ckpt")
state_dict = ms.load_checkpoint(model_path)

print(f"Loading model from {model_path}")
model.load_state_dict(state_dict)

print(f"Loading lora from {args.lora_ckpt}")
model.load_adapter(args.lora_ckpt)
model.merge_and_unload()

ms.save_checkpoint(model, args.output_merge_path)
print(f"Model saved to {args.output_merge_path}")
