#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
echo "$1 $2 $3"

if [ $# != 3 ]; then
  echo "Usage: bash scripts/run_distribute_train.sh  [MINDSPORE_HCCL_CONFIG_PATH] [CKPT_SAVE_DIR] [RANK_SIZE]"
  exit 1
fi
if [ ! -f $1 ]; then
  echo "error: MINDSPORE_HCCL_CONFIG_PATH=$1 is not a file"
  exit 1
fi
export RANK_TABLE_FILE=$1
echo "RANK_TABLE_FILE=$RANK_TABLE_FILE"
export RANK_SIZE=$3
CKPT_SAVE_DIR=$2

for((i=0;i<RANK_SIZE;i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$i

    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    echo "start training for rank $i, device $DEVICE_ID"

    cd ./train_parallel$i ||exit
    env > env.log
    cd ../
    python train.py  \
    --ckpt_save_dir=$CKPT_SAVE_DIR \
    --run_distribute=True > ./train_parallel$i/log.txt 2>&1 &
    echo "python train.py  \
    --ckpt_save_dir=$CKPT_SAVE_DIR \
    --run_distribute=True > ./train_parallel$i/log.txt 2>&1 &"
done
