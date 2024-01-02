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
