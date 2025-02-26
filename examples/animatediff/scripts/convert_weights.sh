wget -c https://download.mindspore.cn/toolkits/mindone/stable_diffusion/sd_v1.5-d0ab7146.ckpt

# Convert SD dreambooth model
wget https://civitai.com/api/download/models/78755 -P models/torch_ckpts/ --content-disposition --no-check-certificate
cp -r ../stable_diffusion_v2/tools/model_conversion ./tools/
python tools/model_conversion/convert_weights.py  --source models/torch_ckpts/toonyou_beta3.safetensors   --target models/dreambooth_lora/toonyou_beta3.ckpt  --model sdv1  --source_version pt

wget https://civitai.com/api/download/models/130072 -P models/torch_ckpts/ --content-disposition --no-check-certificate
python tools/model_conversion/convert_weights.py  --source models/torch_ckpts/realisticVisionV60B1_v51VAE.safetensors   --target models/dreambooth_lora/realisticVisionV51_v51VAE.ckpt  --model sdv1  --source_version pt

# Convert Motion Module
python tools/motion_module_convert.py --src models/torch_ckpts/mm_sd_v15_v2.ckpt --tar models/motion_module

# Convert the animatediff v3 motion module checkpoint
python tools/motion_module_convert.py -v v3 --src models/torch_ckpts/v3_sd15_mm.ckpt  --tar models/motion_module

# Convert Motion LoRA
python tools/motion_lora_convert.py --src models/torch_ckpts/v2_lora_ZoomIn.ckpt --tar models/motion_lora

# Convert Domain Adapter LoRA
python tools/domain_adapter_lora_convert.py --src models/torch_ckpts/v3_sd15_adapter.ckpt --tar models/domain_adapter_lora

# Convert SparseCtrl Encoder
python tools/sparsectrl_encoder_convert.py --src models/torch_ckpts/v3_sd15_sparsectrl_rgb.ckpt --tar models/sparsectrl_encoder
python tools/sparsectrl_encoder_convert.py --src models/torch_ckpts/v3_sd15_sparsectrl_scribble.ckpt --tar models/sparsectrl_encoder
