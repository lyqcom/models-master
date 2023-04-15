#!/usr/bin/env bash
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

if [ $# != 3 ]; then
  echo "Usage: bash run_distribute_train_gpu.sh [RANK_SIZE] [TRAIN_DATA_DIR] [cifar10|imagenet]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

dataset_type='imagenet'
if [ $# == 3 ]
then
    if [ $3 != "cifar10" ] && [ $3 != "imagenet" ]
    then
        echo "error: the selected dataset is neither cifar10 nor imagenet"
    exit 1
    fi
    dataset_type=$3
fi

export RANK_SIZE=$1
PROJECT_DIR=$(cd ./"`dirname $0`" || exit; pwd)
TRAIN_DATA_DIR=$(get_real_path $2)

if [ ! -d $TRAIN_DATA_DIR ]; then
  echo "error: TRAIN_DATA_DIR=$TRAIN_DATA_DIR is not a directory"
  exit 1
fi

if [ -d "distribute_train_gpu" ]; then
  rm -rf ./distribute_train_gpu
fi

mkdir ./distribute_train_gpu
cp ../*.py ./distribute_train_gpu
cp -r ../config ./distribute_train_gpu
cp -r ../src ./distribute_train_gpu
cd ./distribute_train_gpu || exit

if [ $3 == 'imagenet' ]; then
  CONFIG_FILE="$PROJECT_DIR/../config/imagenet_config_gpu.yaml"
elif [ $3 == 'cifar10' ]; then
  CONFIG_FILE="$PROJECT_DIR/../config/cifar10_config_gpu.yaml"
else
  echo "error: the selected dataset is neither cifar10 nor imagenet"
exit 1
fi

mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
nohup python train.py  \
    --config_path=$CONFIG_FILE \
    --dataset_name=$dataset_type \
    --train_data_dir=$TRAIN_DATA_DIR \
    --device_target=GPU > log.txt 2>&1 &
cd ..