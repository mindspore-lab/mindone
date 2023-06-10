# Copyright 2021 Huawei Technologies Co., Ltd
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
"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

Misc utilities
"""
import json
import os
import sys
import random
import numpy as np
import mindspore as ms

class NoOp:
    """ useful for distributed training No-Ops """

    def __getattr__(self, name):
        return self.noop

    def noop(self, *args, **kwargs):
        return


def parse_with_config(args):
    """Parse With Config"""
    if args.train_config is not None:
        abs_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../.."))
        args.train_config = os.path.join(abs_path, args.train_config)
        config_args = json.load(open(args.train_config))
        override_keys = {arg[2:].split('=')[0] for arg in sys.argv[1:]
                         if arg.startswith('--')}
        for k, v in config_args.items():
            if k not in override_keys:
                setattr(args, k, v)
    return args


def set_random_seed(seed):
    """Set Random Seed"""
    print("random seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)

class Struct:
    def __init__(self, dict_):
        self.__dict__.update(dict_)
