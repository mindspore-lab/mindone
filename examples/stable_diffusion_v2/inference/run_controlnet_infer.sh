python  sd_infer.py  \
  --device_target=Ascend \
  --task=controlnet  \
  --model=./config/model/v1-inference-controlnet.yaml  \
  --sampler=./config/schedule/ddim.yaml  \
  --sampling_steps=50  \
  --n_iter=5  \
  --n_samples=1  \
  --scale=9.0