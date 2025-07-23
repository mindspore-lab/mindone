import os
import shlex
import socket
import subprocess
import sys

"""
On modelarts, usage example:
python /home/ma-user/modelarts/user-job-dir/mindone/tools/modelarts/msrun/msrun.py mindone/examples/opensora_hpcai/scripts train.py
"""


def query_host_ip(host_addr):
    try:
        ip = socket.gethostbyname(host_addr)
    finally:
        return ip


def run():
    ip_addr = os.getenv("VC_WORKER_HOSTS").split(",")
    print("host names:", ip_addr)
    print(os.environ)
    ip_addr_list = []
    for i in ip_addr:
        host_addr_ip = query_host_ip(i)
        ip_addr_list.append(host_addr_ip)
    print("ip address list:", ip_addr_list)
    master_addr = ip_addr_list[0]
    node_rank = int(os.getenv("VC_TASK_INDEX"))
    print(f"=======> {sys.argv}", flush=True)
    work_dir = sys.argv[1]  # e.g. mindone/examples/opensora_hpcai/scripts
    script_name = sys.argv[2]  # e.g. train.py
    args = " ".join(sys.argv[3:])
    print("job start with ")

    # install packages before launching training on modelarts
    # os.system(f"bash /home/ma-user/modelarts/user-job-dir/tools/modelarts/msrun/ma-pre-start.sh")

    command = f"bash /home/ma-user/modelarts/user-job-dir/mindone/tools/modelarts/msrun/run_train_modelarts.sh \
        {master_addr} {node_rank} {work_dir} {script_name} {args}"
    print("Running command:", command)
    subprocess.run(shlex.splt(command), check=True, shell=False)


if __name__ == "__main__":
    run()
