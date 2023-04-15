#!/bin/bash
# Copyright 2023 Huawei Technologies Co., Ltd
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

if [ $# != 2 ] && [ $# != 3 ]
then 
    echo "Usage: sh run_standalone_train_gpu.sh [DATA_PATH] [DEVICE_ID] [PRETRAINED_PATH](optional)"
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
PATH2=$3
echo $PATH1

if [ $# == 3 ]
then
    echo $PATH2
fi

ulimit -u unlimited
export CUDA_VISIBLE_DEVICES=$2
export RANK_SIZE=1

if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp *.sh ./train
cp -r ../src ./train
cd ./train || exit

echo "======start training======"
env > env.log
if [ $# == 3 ]
then
    python train.py --coco_root=$PATH1 --do_train=True --device_target=GPU --pre_trained=$PATH2 &> log &
fi

if [ $# == 2 ]
then
    python train.py --coco_root=$PATH1 --do_train=True --device_target=GPU &> log &
fi

cd ..
