import os

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


def sync_data(from_path, to_path):
    """
    Download data from remote obs to local directory if the first url is remote url and the second one is local path
    Upload data from local directory to remote obs in contrast.
    """
    import time

    import moxing as mox

    global _global_sync_count
    sync_lock = "/tmp/copy_sync.lock" + str(_global_sync_count)
    _global_sync_count += 1

    # Each server contains 8 devices as most.
    if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
        print("from path: ", from_path)
        print("to path: ", to_path)
        mox.file.copy_parallel(from_path, to_path)
        print("===finish data synchronization===")
        try:
            os.mknod(sync_lock)
        except IOError:
            pass
        print("===save flag===")

    while True:
        if os.path.exists(sync_lock):
            break
        time.sleep(1)

    print("Finish sync data from {} to {}.".format(from_path, to_path))
