python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base_controlnet.yaml \
  --weight checkpoints/sd_xl_base_1.0_controlnet_canny_ms.ckpt \
  --guidance_scale 9.0 \
  --controlnet_mode canny \
  --control_image_path /PATH TO/dog2.png \
  --prompt "cute dog, best quality, extremely detailed"   \
