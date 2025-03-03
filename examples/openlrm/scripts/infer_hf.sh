# Example usage to load official HF released checkpoints
EXPORT_VIDEO=true
EXPORT_MESH=true
INFER_CONFIG="./configs/infer-b.yaml"
MODEL_NAME="zxhezexin/openlrm-mix-base-1.1"
IMAGE_INPUT="./assets/sample_input/owl.png"
DEVICE_ID=0 python -m openlrm.launch infer.lrm --infer $INFER_CONFIG model_name=$MODEL_NAME image_input=$IMAGE_INPUT export_video=$EXPORT_VIDEO export_mesh=$EXPORT_MESH
