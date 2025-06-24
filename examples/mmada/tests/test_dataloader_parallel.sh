msrun --bind_core=True --worker_num=2 --local_worker_num=2 --master_port=9000 --log_dir=./parallel_logs \
python tests/test_dataloader.py
