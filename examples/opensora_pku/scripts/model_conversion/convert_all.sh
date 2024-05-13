python tools/model_converion/convert_t5.py \
  -s DeepFloyd/t5-v1_1-xxl/pytorch_model-00001-of-00002.bin models/t5-v1_1-xxl/pytorch_model-00002-of-00002.bin \
  -t DeepFloyd/t5-v1_1-xxl/model.ckpt \

python tools/model_conversion/convert_vae.py \
  --src LanguageBind/Open-Sora-Plan-v1.0.0/vae/diffusion_pytorch_model.safetensors \
  --target LanguageBind/Open-Sora-Plan-v1.0.0/vae/model.ckpt \

python tools/model_conversion/convert_latte.py \
  --src LanguageBind/Open-Sora-Plan-v1.0.0/17x256x256/diffusion_pytorch_model.safetensors \
  --target LanguageBind/Open-Sora-Plan-v1.0.0/17x256x256/model.ckpt

python tools/model_conversion/convert_latte.py \
  --src LanguageBind/Open-Sora-Plan-v1.0.0/65x256x256/diffusion_pytorch_model.safetensors \
  --target LanguageBind/Open-Sora-Plan-v1.0.0/65x256x256/model.ckpt

python tools/model_conversion/convert_latte.py \
  --src LanguageBind/Open-Sora-Plan-v1.0.0/65x512x512/diffusion_pytorch_model.safetensors \
  --target LanguageBind/Open-Sora-Plan-v1.0.0/65x512x512/model.ckpt
