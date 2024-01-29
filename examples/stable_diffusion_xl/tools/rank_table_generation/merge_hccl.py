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
# ============================================================================
"""generate hccl config file script"""
import json
import os
import sys
from argparse import ArgumentParser


def parse_args():
    """
    parse args .

    Args:

    Returns:
        args.

    Examples:
        >>> parse_args()
    """
    parser = ArgumentParser(
        description="Merge several hccl config json files" "of single server into one config file of the whole cluster"
    )
    parser.add_argument("file_list", type=str, nargs="+", help="Hccl file lists")
    arg = parser.parse_args()
    return arg


if __name__ == "__main__":
    args = parse_args()
    print(args.file_list)

    server_count = 0
    json_list = []

    for f_name in args.file_list:
        with open(f_name) as f:
            f_json = json.load(f)
            json_list.append(f_json)
            server_count += int(f_json["server_count"])

    hccl_table = {"version": "1.0", "server_count": f"{server_count}", "server_list": []}

    rank_id = 0
    for j in json_list:
        server_list = j["server_list"]
        for server in server_list:
            for device in server["device"]:
                device["rank_id"] = str(rank_id)
                rank_id += 1
        hccl_table["server_list"].extend(server_list)

    hccl_table["status"] = "completed"

    table_path = os.getcwd()
    table_name = os.path.join(table_path, "hccl_{}s_{}p.json".format(server_count, rank_id))
    with open(table_name, "w") as table_fp:
        json.dump(hccl_table, table_fp, indent=4)
    sys.stdout.flush()
    print("Completed: hccl file was save in :", table_name)
