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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_standalone_train.sh DATA_PATH DEVICE_ID CONFIG_PATH"
echo "For example: bash run_standalone_train.sh /path/dataset 0 ../config/config_resnet50_ascend.yaml"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

if [ $# != 3 ]
then
    echo "Usage: bash run_standalone_train.sh [DATASET_PATH] [DEVICE_ID] [CONFIG_PATH]"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

set -e
DATA_PATH=$(get_real_path $1)
DEVICE_ID=$2
export DATA_PATH=${DATA_PATH}
CONFIG_PATH=$(get_real_path $3)
EXEC_PATH=$(pwd)

echo "$EXEC_PATH"

export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

cd ../
export DEVICE_ID=$2
export RANK_ID=$2
env > env0.log
python3 train.py --data_url $1 --isModelArts False --run_distribute False --device_id $2 --config_path $CONFIG_PATH> train.log 2>&1

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
echo "finish"
cd ../
