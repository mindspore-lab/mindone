python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/inference/sd_xl_base_controlnet.yaml \
  --weight checkpoints/sd_xl_base_1.0_controlnet_canny_ms.ckpt \
  --guidance_scale 9.0 \
  --device_target Ascend \
  --controlnet_mode canny \
  --image_path /PATH TO/dog2.png \
  --prompt "cute dog, best quality, extremely detailed"   \
  --negative_prompt "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"  \
  # --control_path /PATH TO/dog2_canny_edge.png  \
