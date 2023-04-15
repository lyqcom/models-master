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


if [ $# != 2 ] && [ $# != 3 ] && [ $# != 4 ]
then 
    echo "Usage: bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)"
    echo "bash run_standalone_train.sh [DATASET_PATH] [CONFIG_PATH] [PRETRAINED_CKPT_PATH](optional)"
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
CONFIG_FILE=$(get_real_path $2)

if [ ! -d $PATH1 ]
then 
    echo "error: DATASET_PATH=$PATH1 is not a directory"
exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=1
export RANK_ID=0
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../config/*.yaml ./train
cp ../*.py ./train
cp *.sh ./train
cp -r ../src ./train
cp -r ../model ./train
cd ./train || exit
echo "start training for device $DEVICE_ID"
env > env.log
if [ $# == 2 ]
then
    python train.py  --data_path=$PATH1 --config_path=$CONFIG_FILE --output_path './output' &> log &
fi

cd ..