# Revise tests/test_dataloader.py MockConfig.training.distributed to True before running this script
msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=9000 --log_dir=./parallel_logs --join True \
python tests/test_dataloader.py
