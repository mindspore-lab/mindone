# Example usage to load pretrained ms checkpoint
EXPORT_VIDEO=true
EXPORT_MESH=true
INFER_CONFIG="./configs/infer-s.yaml" # <REPLACE_WITH_EXACT_CONFIG>
MODEL_NAME="outputs/small/2024-12-30T18-24-53" # <REPLACE_WITH_TRAINED_MODEL_DIR>
IMAGE_INPUT="sample_input" # <REPLACE_WITH_INPUT_IMAGE_DIR>

# epoch=None
# MODEL_CKPT=None
# or
epoch=100000
MODEL_CKPT=openlrm-e${epoch}.ckpt # <REPLACE_WITH_EXACT_CKPT_NAME>
DEVICE_ID=0 python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME model_ckpt=$MODEL_CKPT epoch=$epoch image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
