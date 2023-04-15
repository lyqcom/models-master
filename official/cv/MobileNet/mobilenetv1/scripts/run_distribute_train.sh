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

if [ $# != 3 ] && [ $# != 4 ]
then 
    echo "Usage: sh run_distribute_train.sh [cifar10|imagenet2012] [RANK_TABLE_FILE] [DATASET_PATH] [PRETRAINED_CKPT_PATH](optional)"
    exit 1
fi

if [ $1 != "cifar10" ] && [ $1 != "imagenet2012" ]
then 
    echo "error: the selected dataset is neither cifar10 nor imagenet2012"
    exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $2)
PATH2=$(get_real_path $3)

if [ $# == 4 ]
then 
    PATH3=$(get_real_path $4)
fi

if [ ! -f $PATH1 ]
then 
    echo "error: RANK_TABLE_FILE=$PATH1 is not a file"
    exit 1
fi 

if [ ! -d $PATH2 ]
then 
    echo "error: DATASET_PATH=$PATH2 is not a directory"
    exit 1
fi 

if [ $# == 4 ] && [ ! -f $PATH3 ]
then
    echo "error: PRETRAINED_CKPT_PATH=$PATH3 is not a file"
    exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$PATH1

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
if [ $# -ge 1 ]; then
  if [ $1 == 'cifar10' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  elif [ $1 == 'imagenet2012' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_imagenet.yaml"
  else
    echo "Unrecognized parameter"
    exit 1
  fi
else
  CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
fi

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $DEVICE_NUM`
gap=`expr $avg \- 1`

for((i=0; i<${DEVICE_NUM}; i++))
do
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp ../*.py ./train_parallel$i
    cp ../*.yaml ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ../src ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    if [ $# == 3 ]
    then    
        taskset -c $cmdopt python train.py --config_path=$CONFIG_FILE --dataset=$1 --run_distribute=True --device_num=$DEVICE_NUM --dataset_path=$PATH2 &> log.txt &
    fi
    
    if [ $# == 4 ]
    then
        taskset -c $cmdopt python train.py --config_path=$CONFIG_FILE --dataset=$1 --run_distribute=True --device_num=$DEVICE_NUM --dataset_path=$PATH2 --pre_trained=$PATH3 &> log.txt &
    fi

    cd ..
done
