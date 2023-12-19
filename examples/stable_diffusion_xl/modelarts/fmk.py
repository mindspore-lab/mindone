import os
import pathlib
import subprocess
from contextlib import contextmanager

from common import ModelArts, RankTableEnv, RunAscendLog

log = RunAscendLog.get_run_ascend_logger()


class FMK:
    def __init__(self, index, device):
        self.job_id = ModelArts.get_job_id()
        self.rank_id = device.rank_id
        self.device_id = str(index)

    def gen_env_for_fmk(self, rank_size):
        current_envs = os.environ.copy()

        current_envs["JOB_ID"] = self.job_id

        current_envs["ASCEND_DEVICE_ID"] = self.device_id
        current_envs["DEVICE_ID"] = self.device_id

        current_envs["RANK_ID"] = self.rank_id
        current_envs["RANK_SIZE"] = str(rank_size)

        FMK.set_env_if_not_exist(current_envs, RankTableEnv.HCCL_CONNECT_TIMEOUT, str(1800))

        log_dir = FMK.get_log_dir()
        process_log_path = os.path.join(log_dir, self.job_id, "ascend", "process_log", "rank_" + self.rank_id)
        FMK.set_env_if_not_exist(current_envs, "ASCEND_PROCESS_LOG_PATH", process_log_path)
        pathlib.Path(current_envs["ASCEND_PROCESS_LOG_PATH"]).mkdir(parents=True, exist_ok=True)

        return current_envs

    @contextmanager
    def switch_directory(self, directory):
        owd = os.getcwd()
        try:
            os.chdir(directory)
            yield directory
        finally:
            os.chdir(owd)

    def get_working_dir(self):
        fmk_workspace_prefix = ModelArts.get_parent_working_dir()
        return os.path.join(os.path.normpath(fmk_workspace_prefix), "device%s" % self.device_id)

    @staticmethod
    def get_log_dir():
        parent_path = os.getenv(ModelArts.MA_MOUNT_PATH_ENV)
        if parent_path:
            log_path = os.path.join(parent_path, "log")
            if os.path.exists(log_path):
                return log_path

        return ModelArts.TMP_LOG_DIR

    @staticmethod
    def set_env_if_not_exist(envs, env_name, env_value):
        if env_name in os.environ:
            log.info("env already exists. env_name: %s, env_value: %s " % (env_name, env_value))
            return

        envs[env_name] = env_value

    def run(self, rank_size, command):
        envs = self.gen_env_for_fmk(rank_size)
        log.info("bootstrap proc-rank-%s-device-%s" % (self.rank_id, self.device_id))

        log_dir = FMK.get_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_file = "%s-proc-rank-%s-device-%s.txt" % (self.job_id, self.rank_id, self.device_id)
        log_file_path = os.path.join(log_dir, log_file)

        working_dir = self.get_working_dir()
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

        with self.switch_directory(working_dir):
            # os.setsid: change the process(forked) group id to itself
            training_proc = subprocess.Popen(
                command, env=envs, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )

            log.info("proc-rank-%s-device-%s (pid: %d)", self.rank_id, self.device_id, training_proc.pid)

            # https://docs.python.org/3/library/subprocess.html#subprocess.Popen.wait
            subprocess.Popen(["tee", log_file_path], stdin=training_proc.stdout)

            return training_proc
