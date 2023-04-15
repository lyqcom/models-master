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
if [ $# != 6 ]
then
    echo "Usage: sh run_standalone_train_gpu.sh [TRAIN_DATA_PATH] [ANNO_DATA_PATH] [PRETRAIN_PATH] [CKPT_SAVE_PATH] [DEVICE_ID] [DEVICE_NUM]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}
TRAIN_PATH=$(get_real_path $1)
ANNO_PATH=$(get_real_path $2)
PRETRAIN_PATH=$(get_real_path $3)
SAVE_PATH=$(get_real_path $4)


if [ ! -d $PATH1 ]
then
    echo "error: TRAIN_DATA_PATH=$PATH1 is not a directory"
exit 1
fi

if [ ! -d $PATH2 ]
then
    echo "error: ANNO_DATA_PATH=$PATH2 is not a directory"
exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: PRETRAIN_PATH=$PATH3 is not a directory"
exit 1
fi

if [ ! -d $PATH4 ]
then
    echo "error: CKPT_SAVE_PATH=$PATH4 is not a directory"
exit 1
fi

export DEVICE_ID=$5
export DEVICE_NUM=$6

rm -rf device
mkdir device
cp -r ../src/ ./device
cp ../train.py ./device
echo "start training"
cd ./device
export TRAIN_PATH=$1
export ANNO_PATH=$2
export PRETRAIN_PATH=$3
export SAVE_PATH=$4

python train.py --device_id=$DEVICE_ID --train_path=$TRAIN_PATH --device_num=$DEVICE_NUM --anno_path=$ANNO_PATH --pretrain_ckpt_path=$PRETRAIN_PATH --ckpt_save_path=$SAVE_PATH   > train.log 2>&1 &

