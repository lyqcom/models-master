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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval_onnx.sh  [DEVICE_ID] [CKPT_PATH]"
echo "For example: bash run_eval_onnx.sh device_id onnx_path"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

if [ $# != 2 ]
then
    echo "Usage: bash run_eval_onnx.sh device_id onnx_path"
exit 1
fi

set -e
DEVICE_ID=$1
CKPT_PATH=$2
export DEVICE_ID=$DEVICE_ID
export CKPT_PATH

EXEC_PATH=$(pwd)
echo "$EXEC_PATH"

python eval_onnx.py --ckpt_url CKPT_PATH > ./eval_onnx.log 2>&1 &
