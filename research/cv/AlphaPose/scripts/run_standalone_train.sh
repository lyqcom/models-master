#!/bin/bash
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
echo "========================================================================"
echo "Please run the script as: "
echo "bash run_standalone_train.sh"
echo "For example: bash run_standalone_train.sh"
echo "It is better to use the absolute path."
echo "========================================================================"
echo "start training for device $DEVICE_ID"
export DEVICE_ID=$1
python -u ../train.py --DEVICE_TARGET Ascend --DEVICE_ID ${DEVICE_ID} > train${DEVICE_ID}.log 2>&1 &
echo "finish"
