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

if [ $# != 1 ] && [ $# != 2 ]
then
  echo "Usage bash run_train_cpu.sh [TRAIN_DATA_DIR] [cifar10|imagenet]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
if [ ! -d $PATH1 ]
then
  echo "error: TRAIN_DATA_DIR=$PATH1 is not a directory"
exit 1
fi

BASE_PATH=$(dirname "$(cd "$(dirname "$0")" || exit; pwd)")
if [ $2 == 'imagenet' ]; then
  CONFIG_FILE="${BASE_PATH}/config/imagenet_config.yaml"
elif [ $2 == 'cifar10' ]; then
  CONFIG_FILE="${BASE_PATH}/config/cifar10_config.yaml"
else
  echo "error: the selected dataset is neither cifar10 nor imagenet"
exit 1
fi

rm -rf ./train_cpu
mkdir ./train_cpu
cp ../train.py ./train_cpu
cp -r ../src ./train_cpu
cp -r ../config ./train_cpu
echo "start training for device CPU"
cd ./train_cpu || exit
env > env.log
python train.py --device_target=CPU --train_data_dir=$PATH1 --dataset_name=$2 \
    --config_path=$CONFIG_FILE> ./train.log 2>&1 &
cd ..
