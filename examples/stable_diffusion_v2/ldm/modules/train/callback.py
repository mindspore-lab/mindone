# Copyright 2022 Huawei Technologies Co., Ltd
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

import mindspore as ms


class OverflowMonitor(ms.Callback):
    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        overflow = cb_params.net_outputs[1]
        if overflow:
            print(f"overflow detected in epoch {cur_epoch_num} step {cur_step_in_epoch}")
        return super().step_end(run_context)