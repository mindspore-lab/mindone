export DEVICE_ID=0

# Test case 1: Replace background with mountains
python main.py \
  --image_path './assets/imgs/girl33.jpeg' \
  --image_whratio_unchange \
  --save_folder './results/' \
  --prompt_local "A mountain." \
  --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
  --prompt_target "A young girl with red hair smiles brightly, wearing a red and white checkered shirt, sitting on a bench with mountains in the background." \
  --mask_path "./assets/mask_temp/mask_209_inverse.png"


## Test case 2: Replace the girl with a boy
#python main.py \
#  --image_path './assets/imgs/girl33.jpeg' \
#  --image_whratio_unchange \
#  --save_folder './results/' \
#  --prompt_local "A boy smiling." \
#  --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
#  --prompt_target "A young boy with red hair smiles brightly, wearing a red and white checkered shirt." \
#  --mask_path "./assets/mask_temp/mask_208.png"


## Test case 3: Add a monkey
#python main.py \
#--image_path './assets/imgs/girl33.jpeg' \
#--image_whratio_unchange \
#--save_folder './results/' \
#--prompt_local "A monkey playing." \
#--prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
#--prompt_target "A young girl with red hair smiles brightly, wearing a red and white checkered shirt, a monkey playing nearby." \
#--mask_path "./assets/mask_temp/mask_213.png"


## Test case 4 Remove the girl
#python main.py \
#  --image_path './assets/imgs/girl33.jpeg' \
#  --image_whratio_unchange \
#  --save_folder './results/' \
#  --prompt_local '[remove]' \
#  --mask_path "./assets/mask_temp/mask_208.png" \
#  --dilate_mask \


## Test case 5: Replace the girl with a boy + add a monkey
#python main.py \
#  --image_path './assets/imgs/girl33.jpeg' \
#  --image_whratio_unchange \
#  --save_folder './results/' \
#  --prompt_source "A young girl with red hair smiles brightly, wearing a red and white checkered shirt." \
#  --prompt_local "A boy smiling." \
#  --prompt_local "A monkey playing." \
#  --mask_path "./assets/mask_temp/mask_208.png" \
#  --mask_path "./assets/mask_temp/mask_215.png" \
#  --prompt_target "A young boy wearing a red and white checkered shirt, a monkey playing nearby."
