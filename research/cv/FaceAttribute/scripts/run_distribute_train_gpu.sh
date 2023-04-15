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

if [ $# != 3 ] && [ $# != 4 ]
then
    echo "Usage: sh run_distribute_train_gpu [DEVICE_NUM] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [MINDRECORD_FILE] [PRETRAINED_BACKBONE]"
    echo "   or: sh run_distrubute_train_gpu.sh [DEVICE_NUM] [CUDA_VISIBLE_DEVICES(0,1,2,3,4,5,6,7)] [MINDRECORD_FILE]"
exit 1
fi

current_exec_path=$(pwd)
echo ${current_exec_path}

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}

export PYTHONPATH=${dirname_path}:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=$2
export RANK_SIZE=$1

SCRIPT_NAME='train.py'
ulimit -c unlimited

echo 'start training'
export RANK_ID=0
rm -rf train_distribute_gpu
mkdir train_distribute_gpu
cd train_distribute_gpu

if [ $# == 3 ]
then
  mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
  python ${dirname_path}/${SCRIPT_NAME} \
      --world_size=$1 \
      --device_target='GPU' \
      --mindrecord_path=$3 > train.log 2>&1 &
else
  mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
  python ${dirname_path}/${SCRIPT_NAME} \
      --world_size=$1 \
      --device_target='GPU' \
      --mindrecord_path=$3 \
      --pretrained=$4 > train.log  2>&1 &
fi
echo 'running'
