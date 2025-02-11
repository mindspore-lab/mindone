python scripts/eval_tae.py \
--mode=0 \
--jit_level=O1 \
--sample_n_frames=32 \
--batch_size 1 \
--size 256 \
--pretrained outputs/train_tae/ckpt/tae-e25.ckpt \
--csv_path datasets/ucf101_test.csv \
--folder datasets/UCF-101 \
