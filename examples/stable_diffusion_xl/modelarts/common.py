import logging
import os

logger_name = "Training"


# Rank Table Constants
class RankTableEnv:
    RANK_TABLE_FILE = "RANK_TABLE_FILE"

    RANK_TABLE_FILE_V1 = "RANK_TABLE_FILE_V_1_0"

    HCCL_CONNECT_TIMEOUT = "HCCL_CONNECT_TIMEOUT"

    # jobstart_hccl.json is provided by the volcano controller of Cloud-Container-Engine(CCE)
    HCCL_JSON_FILE_NAME = "jobstart_hccl.json"

    RANK_TABLE_FILE_DEFAULT_VALUE = "/user/config/%s" % HCCL_JSON_FILE_NAME

    @staticmethod
    def get_rank_table_template1_file_dir():
        parent_dir = os.environ[ModelArts.MA_MOUNT_PATH_ENV]
        return os.path.join(parent_dir, "rank_table")

    @staticmethod
    def get_rank_table_template2_file_path():
        rank_table_file_path = os.environ.get(RankTableEnv.RANK_TABLE_FILE)
        if rank_table_file_path is None:
            return RankTableEnv.RANK_TABLE_FILE_DEFAULT_VALUE

        return os.path.join(os.path.normpath(rank_table_file_path), RankTableEnv.HCCL_JSON_FILE_NAME)

    @staticmethod
    def set_rank_table_env(path):
        os.environ[RankTableEnv.RANK_TABLE_FILE] = path

    @staticmethod
    def unset_rank_table_env():
        del os.environ[RankTableEnv.RANK_TABLE_FILE]


class ModelArts:
    MA_MOUNT_PATH_ENV = "MA_MOUNT_PATH"
    MA_CURRENT_INSTANCE_NAME_ENV = "MA_CURRENT_INSTANCE_NAME"
    MA_VJ_NAME = "MA_VJ_NAME"

    MA_CURRENT_HOST_IP = "MA_CURRENT_HOST_IP"

    CACHE_DIR = "/cache"

    TMP_LOG_DIR = "/tmp/log/"

    FMK_WORKSPACE = "workspace"

    @staticmethod
    def get_current_instance_name():
        return os.environ[ModelArts.MA_CURRENT_INSTANCE_NAME_ENV]

    @staticmethod
    def get_current_host_ip():
        return os.environ.get(ModelArts.MA_CURRENT_HOST_IP)

    @staticmethod
    def get_job_id():
        ma_vj_name = os.environ[ModelArts.MA_VJ_NAME]
        return ma_vj_name.replace("ma-job", "modelarts-job", 1)

    @staticmethod
    def get_parent_working_dir():
        if ModelArts.MA_MOUNT_PATH_ENV in os.environ:
            return os.path.join(os.environ.get(ModelArts.MA_MOUNT_PATH_ENV), ModelArts.FMK_WORKSPACE)

        return ModelArts.CACHE_DIR


class RunAscendLog:
    @staticmethod
    def setup_run_ascend_logger():
        name = logger_name
        formatter = logging.Formatter(fmt="[run ascend] %(asctime)s - %(levelname)s - %(message)s")

        handler = logging.StreamHandler()
        handler.setFormatter(formatter)

        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    @staticmethod
    def get_run_ascend_logger():
        return logging.getLogger(logger_name)
