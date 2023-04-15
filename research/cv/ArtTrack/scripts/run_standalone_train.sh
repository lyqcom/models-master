#!/bin/bash
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
echo "========================================================================"
echo "Please run the script as: "
echo "bash scripts/run_standalone_train.sh DEVICE_ID"
echo "For example: bash scripts/run_standalone_train.sh 0"
echo "It is better to use the absolute path."
echo "========================================================================"
export DEVICE_ID=$1
echo "start training for device $DEVICE_ID"
python -u train.py --device_target 'Ascend' --device_id ${DEVICE_ID} > train${DEVICE_ID}.log 2>&1 &
tail -f train${DEVICE_ID}.log
echo "finish"
