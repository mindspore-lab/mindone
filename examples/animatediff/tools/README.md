## Convert motion module checkpoint from Torch to MindSpore

```
cd tools/
python motion_module_convert.py --src {path to torch motion module ckpt} --tar {output folder}
```

The converted checkpoint will be saved in {output folder}


## Convert motion lora checkpoint

```
cd tools/
python motion_lora_convert.py --src {path to torch motion lora ckpt} --tar {output folder}
```

The converted checkpoint will be saved in {output folder}

## Save original dataset files as embedding cache to accelerate training

1. MindRecord:
```
python tools/embedding_cache.py --config configs/training/mmv2_train.yaml \
                                --train_data_type mindrecord \
                                --cache_folder /path/to/save_mindrecord \
                                --image_size 512
```

2. npz:
```
python tools/embedding_cache.py --config configs/training/mmv2_train.yaml \
                                --train_data_type npz \
                                --cache_folder /path/to/save_mindrecord \
                                --image_size 512
```
