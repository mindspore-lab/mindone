python tools/model_conversion/convert_t5.py \
  -s DeepFloyd/t5-v1_1-xxl/pytorch_model-00001-of-00002.bin DeepFloyd/t5-v1_1-xxl/pytorch_model-00002-of-00002.bin \
  -t DeepFloyd/t5-v1_1-xxl/t5-v1_1-xxl.ckpt \

python tools/model_conversion/convert_vae.py \
  --src LanguageBind/Open-Sora-Plan-v1.1.0/vae/diffusion_pytorch_model.safetensors \
  --target LanguageBind/Open-Sora-Plan-v1.1.0/vae/causalvae_488.ckpt \

python tools/model_conversion/convert_latte.py \
  --src LanguageBind/Open-Sora-Plan-v1.1.0/65x512x512/diffusion_pytorch_model.safetensors \
  --target LanguageBind/Open-Sora-Plan-v1.1.0/65x512x512/LatteT2V-65x512x512.ckpt

python tools/model_conversion/convert_latte.py \
  --src LanguageBind/Open-Sora-Plan-v1.1.0/221x512x512/diffusion_pytorch_model.safetensors \
  --target LanguageBind/Open-Sora-Plan-v1.1.0/221x512x512/LatteT2V-221x512x512.ckpt

# python tools/model_conversion/convert_latte.py \
#   --src LanguageBind/Open-Sora-Plan-v1.1.0/513x512x512/diffusion_pytorch_model.safetensors \
#   --target LanguageBind/Open-Sora-Plan-v1.1.0/513x512x512/LatteT2V-513x512x512.ckpt
