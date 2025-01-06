# Example usage to load pretrained ms checkpoint
EXPORT_VIDEO=true
EXPORT_MESH=true
INFER_CONFIG="./configs/infer-s.yaml"
MODEL_NAME="./outputs/5data_sample3/2024-12-30T18-24-53"
IMAGE_INPUT="./openlrm/eval"

# epoch=None
# MODEL_CKPT=None
# or
epoch=6850
MODEL_CKPT=openlrm-e${epoch}.ckpt
DEVICE_ID=0 python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME model_ckpt=$MODEL_CKPT epoch=$epoch image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
