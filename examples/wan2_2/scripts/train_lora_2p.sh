msrun --worker_num=2 --local_worker_num=2 train.py --task ti2v-5B --size 1280*704 --t5_zero3 \
    --ckpt_dir ./model/Wan2.2-TI2V-5B \
    --data_root ./data/Disney-VideoGeneration-Dataset \
    --caption_column prompt.txt \
    --video_column videos.txt
