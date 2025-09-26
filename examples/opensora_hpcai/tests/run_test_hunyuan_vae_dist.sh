export HF_HUB_OFFLINE=1
export PYTHONPATH=../..:$PYTHONPATH
msrun --master_port=8200 --worker_num=2 --local_worker_num=2 --log_dir="logs" --join=True pytest tests/test_hunyuan_vae_dist.py
