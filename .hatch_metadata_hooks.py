import os
import subprocess

from hatchling.metadata.plugin.interface import MetadataHookInterface


def get_git_info():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    git_root = (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=script_dir, stderr=subprocess.STDOUT)
        .decode()
        .strip()
    )

    commit_id = subprocess.check_output(["git", "log", "-1", "--pretty=format:%H"], cwd=git_root).decode().strip()[:7]

    commit_date = (
        subprocess.check_output(["git", "log", "-1", "--pretty=format:%cd", "--date=iso"], cwd=git_root)
        .decode()
        .strip()
        .split()[0]
        .replace("-", "")
    )

    return commit_id, commit_date


class MetadataHookInterface(MetadataHookInterface):
    PLUGIN_NAME = "version_with_latest_commit_info"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.__config_base_version = None
        self.__config_type = None

    @property
    def config_build_type(self):
        if self.__config_type is None:
            build_type = self.config.get("build-type", "dev")
            if not isinstance(build_type, str):
                raise TypeError("option `type` must be a string with options ['dev', 'release'].")

            self.__config_build_type = build_type

        return self.__config_build_type

    @property
    def config_base_version(self):
        if self.__config_base_version is None:
            base_version = self.config.get("base-version", "0.0.1")
            if not isinstance(base_version, str):
                raise TypeError("option `base-version` must be a string")

            self.__config_base_version = base_version

        return self.__config_base_version

    def update_version(self, metadata: dict) -> None:
        if self.config_build_type == "dev":
            commit_id, commit_date = get_git_info()
            metadata["version"] = f"{self.config_base_version}+g{commit_id}.d{commit_date}"
        else:
            metadata["version"] = self.config_base_version

    def update(self, metadata: dict) -> None:
        self.update_version(metadata)
