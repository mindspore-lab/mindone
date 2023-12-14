export DEVICE_ID=$1
export MS_PYNATIVE_GE=1
export PYTHONPATH=$(pwd):$PYTHONPATH

base_ckpt_path='models/sd_xl_base_1.0_ms.ckpt'
ckpt_path="lora_ft_pokemon/SDXL-base-1.0_40000_lora.ckpt"

init_latent_path='sdxl_latent_noise/prompt_bird.npy'

#init_latent_path='sdxl_latent_noise/prompt_dragon.npy'
#--prompt "a drawing of a black and gray dragon" \

# prompt=datasets/pokemon_blip/test/prompts.txt

python demo/sampling_without_streamlit.py \
  --task txt2img \
  --config configs/training/sd_xl_base_finetune_lora_910b.yaml \
  --save_path outputs/lora_pokemon_fix_noise \
  --weight $base_ckpt_path,$ckpt_path \
  --prompt "a small bird with a black and white tail" \
  --init_latent_path $init_latent_path \
  --device_target Ascend \
  --discretization "DiffusersDDPMDiscretization" \
  --precision_keep_origin_dtype True \
  # --seed 43 \

  #init_latent_path='sdxl_latent_noise/prompt_butterfly.npy'
  # --prompt "a black and white photo of a butterfly" \
