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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 4 ]
then
    echo "Usage: sh run_distribution_train_ascend.sh [DEVICENUM] [RANK_TABLE_FILE] [cifar10] [TRAIN_DATASET_PATH]"
exit 1
fi

#
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

#
if [ ! -f $2 ]
then
    echo "error: RANK_TABLE_FILE=$2 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=$1
export RANK_SIZE=$1
RANK_TABLE_FILE=$(get_real_path $2)
export RANK_TABLE_FILE
export DATASET_NAME=$3
export TRAIN_DATASET_PATH=$(get_real_path $4)
echo "RANK_TABLE_FILE=${RANK_TABLE_FILE}"

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=$i
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cp ../train.py ./train_parallel$i
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    cd ./train_parallel$i ||exit
    env > env.log
    python train.py --device_id=$i --dataset_name=$DATASET_NAME --train_dataset_path=$TRAIN_DATASET_PATH \
                    --run_cloudbrain=False --run_distribute=True > log 2>&1 &
    cd ..
done
