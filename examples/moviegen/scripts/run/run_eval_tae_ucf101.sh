python scripts/inference_vae.py \
--mode=0 \
--jit_level=O1 \
--num_frames=32  \
--batch_size 1 \
--image_size 256 \
--ckpt_path  outputs/train_tae/ckpt/tae-e25.ckpt   \
--csv_path datasets/ucf101_test.csv  \
--video_folder datasets/UCF-101 \
