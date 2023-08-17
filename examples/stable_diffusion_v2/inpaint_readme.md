## SD 2 inpainting
python inpaint.py 
    --image datasets/inpaint/overture-creations-5sI6fQgYIuo.png \
    --mask datasets/inpaint/overture-creations-5sI6fQgYIuo_mask.png \
    --prompt "Face of a yellow cat, high resolution, sitting on a park bench"

## SD 1.5 CN
python inpaint.py 
    --image {path to input image} \
    --mask {path to mask image where white regions are to be edited} \
    --prompt {YouChinesePrompt} \
    -v 1.5

