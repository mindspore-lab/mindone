import functools
import logging
import multiprocessing as mp
import os
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable, Dict, List, Union

sys.path.append("../../")
from tools.data_utils.laion_gen_csv_annot import gen_csv

LOCAL_RANK = int(os.getenv("RANK_ID", 0))
_logger = logging.getLogger("mindone.tools.modelarts")

_global_sync_count = 0


def get_device_id():
    device_id = os.getenv("DEVICE_ID", "0")
    return int(device_id)


def get_device_num():
    device_num = os.getenv("RANK_SIZE", "1")
    return int(device_num)


def get_rank_id():
    global_rank_id = os.getenv("RANK_ID", "0")
    return int(global_rank_id)


def extract_tar(file_path):
    try:
        with tarfile.open(file_path, "r") as archive:
            if "/" not in archive.getnames()[1]:
                subfolder_path = file_path[:-4]
                os.makedirs(subfolder_path, exist_ok=True)
                archive.extractall(subfolder_path)
            else:
                archive.extractall(os.path.dirname(file_path))
        os.remove(file_path)
        _logger.info("finish extract: " + file_path)
        return True
    except Exception as e:
        _logger.error(f"Error in extracting {file_path} : {e}")
        return False


def obs_copy(f_t):
    import moxing as mox

    f, t = f_t
    mox.file.copy_parallel(f, t)
    _logger.info("finish copy: " + t)
    return True


def sync_data(from_path, to_path, start_sample_idx, end_sample_idx, json_data_path):
    """
    Copy data from `from_path` to `to_path`.
    1) if `from_path` is remote url and `to_path` is local path, download data from remote obs to local directory
    2) if `from_path` is local path and `to_path` is remote url, upload data from local directory to remote obs .
    """
    import time

    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    print("get_device_id()", get_device_id(), flush=True)
    print("min(get_device_num(), 8)", min(get_device_num(), 8), flush=True)
    print("get_device_id() % min(get_device_num(), 8)", get_device_id() % min(get_device_num(), 8), flush=True)
    print("not os.path.exists(sync_lock)", not os.path.exists(sync_lock))

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        _logger.info(f"from path: {from_path}")
        _logger.info(f"to path: {to_path}")
        obs_args = list(zip(from_path, to_path))
        pool = mp.Pool(20)  # number of workers
        pool.map(obs_copy, obs_args, chunksize=1)
        pool.close()
        pool.join()
        _logger.info("===finish data synchronization===")
        _logger.info("Finish sync data from {} to {}.".format(from_path, to_path))

        pool = mp.Pool(min(mp.cpu_count(), len(to_path)))  # number of workers
        pool.map(extract_tar, to_path, chunksize=1)
        pool.close()
        pool.join()
        _logger.info("===finish data unzip===")

        gen_csv(
            "/cache",
            one_csv_per_part=True,
            folder_prefix="part",
            start_sample_idx=start_sample_idx,
            end_sample_idx=end_sample_idx,
            del_part_csvs=True,
            json_data_path=json_data_path,
        )
        _logger.info("===finish generate summary csv, path: /cache/merged_imgp_text.csv ===")
        _logger.info("===copy /cache/merged_imgp_text.csv back to obs===")

        try:
            os.mknod(sync_lock)
        except IOError:
            pass
        _logger.info("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)


def run_with_single_rank(local_rank: int = 0, signal: str = "/tmp/SUCCESS") -> Callable[..., Any]:
    """Run the task on 0th rank, perform synchronzation before return"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if local_rank in [0, None]:
                result = func(*args, **kwargs)
                with open(signal, "w") as f:
                    f.write("\n")
                return result
            else:
                while not os.path.isfile(signal):
                    time.sleep(1)

        return wrapper

    return decorator


@run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/INSTALL_SUCCESS")
def install_packages(req_path: str = "requirements.txt") -> None:
    url = "https://pypi.tuna.tsinghua.edu.cn/simple"
    # requirement_txt = os.path.join(project_dir, "requirements.txt")
    _logger.info(f"Packages to be installed: {req_path}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", url, "--upgrade", "pip"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-i", url, "-r", req_path])


# @run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/DOWNLOAD_DATA_SUCCESS")
# def download_data(s3_path: str, dest: str) -> None:
#    if not os.path.isdir(dest):
#        os.makedirs(dest)
#    DownLoad().download_url(url=s3_path, path=dest)


@run_with_single_rank(local_rank=LOCAL_RANK, signal="/tmp/DOWNLOAD_CKPT_SUCCESS")
def download_ckpt(s3_path: str, dest: str) -> str:
    if not os.path.isdir(dest):
        os.makedirs(dest)

    filename = os.path.basename(s3_path)
    dst_url = os.path.join(dest, filename)

    import moxing as mox

    mox.file.copy(src_url=s3_path, dst_url=dst_url)
    return dst_url


def upload_data(src: str, s3_path: str) -> None:
    abs_src = os.path.abspath(src)
    _logger.info(f"Uploading data from {abs_src} to s3")
    import moxing as mox

    mox.file.copy_parallel(src_url=abs_src, dst_url=s3_path)


def modelarts_setup(args):
    if args.enable_modelarts:
        cur_dir = os.path.dirname(os.path.abspath(__file__))

        # change relative path of configure file to absolute
        if not os.path.isabs(args.config):
            args.config = os.path.abspath(os.path.join(cur_dir, "../../", args.config))

        req_path = os.path.abspath(os.path.join(cur_dir, "../../requirements/modelarts.txt"))
        install_packages(req_path)
        return True
    return False


def update_config_value_by_key(config: Union[Dict, List], key: str, value: Any):
    if isinstance(config, dict):
        if key in config:
            config[key] = value
        for subconfig in config.values():
            update_config_value_by_key(subconfig, key, value)
    elif isinstance(config, list):
        for subconfig in config:
            update_config_value_by_key(subconfig, key, value)
