# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
setup packpage
"""
import os
import shlex
import shutil
import stat
import subprocess

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info

exec(open("mindone/version.py").read())

version = __version__
package_name = "mindone"
cur_dir = os.path.dirname(os.path.realpath(__file__))
pkg_dir = os.path.join(cur_dir, "build")


def clean():
    # pylint: disable=unused-argument
    def readonly_handler(func, path, execinfo):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if os.path.exists(os.path.join(cur_dir, "build")):
        shutil.rmtree(os.path.join(cur_dir, "build"), onerror=readonly_handler)
    if os.path.exists(os.path.join(cur_dir, "mindone.egg-info")):
        shutil.rmtree(os.path.join(cur_dir, "mindone.egg-info"), onerror=readonly_handler)


clean()


def update_permissions(path):
    """
    Update permissions.

    Args:
        path (str): Target directory path.
    """
    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dir_fullpath = os.path.join(dirpath, dirname)
            os.chmod(dir_fullpath, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC | stat.S_IRGRP | stat.S_IXGRP)
        for filename in filenames:
            file_fullpath = os.path.join(dirpath, filename)
            os.chmod(file_fullpath, stat.S_IREAD)


def get_description():
    """
    Get description.

    Returns:
        str, wheel package description.
    """
    cmd = "git log --format='[sha1]:%h, [branch]:%d' -1"
    process = subprocess.Popen(
        shlex.split(cmd), shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    stdout, _ = process.communicate()
    if not process.returncode:
        git_version = stdout.decode().strip()
        return "An open source natural language processing research tool box. Git version: %s" % (git_version)
    return "An open source natural language processing research tool box."


class EggInfo(egg_info):
    """Egg info."""

    def run(self):
        super().run()
        egg_info_dir = os.path.join(cur_dir, "mindone.egg-info")
        update_permissions(egg_info_dir)


class BuildPy(build_py):
    """BuildPy."""

    def run(self):
        super().run()
        mindarmour_dir = os.path.join(pkg_dir, "lib", "mindone")
        update_permissions(mindarmour_dir)


setup(
    name="mindone",
    version=version,
    author="MindSpore Team",
    url="https://github.com/mindspore-lab/mindone/tree/master",
    project_urls={
        "Sources": "https://github.com/mindspore-lab/mindone",
        "Issue Tracker": "https://github.com/mindspore-lab/mindone/issues",
    },
    description=get_description(),
    license="Apache 2.0",
    packages=find_packages(exclude=("example")),
    include_package_data=True,
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},
    cmdclass={
        "egg_info": EggInfo,
        "build_py": BuildPy,
    },
    install_requires=[
        "tqdm",
        # 'requests',
        # 'datasets',
        # 'tokenizers'
    ],
    classifiers=["License :: OSI Approved :: Apache Software License"],
)
print(find_packages())
