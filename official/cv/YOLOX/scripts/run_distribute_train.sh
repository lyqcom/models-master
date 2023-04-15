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
# ===========================================================================
if [[ $# -lt 3 || $# -gt 4 ]];then
    echo "Usage1: bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE] [BACKBONE]"
    echo "Usage2: bash run_distribute_train.sh [DATASET_PATH] [RANK_TABLE_FILE] [BACKBONE] [RESUME_CKPT] for resume"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_PATH=$(get_real_path $1)
RANK_TABLE_FILE=$(get_real_path $2)
BACKBONE=$3
if [ "$BACKBONE" = 'yolox_darknet53' ]
then
  CONFIG_PATH='yolox_darknet53.yaml'
else
  CONFIG_PATH='yolox_x.yaml'
fi
echo $DATASET_PATH
echo $RANK_TABLE_FILE
echo $BACKBONE
if [ $# == 4 ]
then
  RESUME_CKPT=$(get_real_path $4)
  if [ ! -f $RESUME_CKPT ]
    then
    echo "error: RESUME_CKPT=$RESUME_CKPT is not a file"
    exit 1
  fi
  echo $RESUME_CKPT
fi

if [ ! -d $DATASET_PATH ]
then
    echo "error: DATASET_PATH=$DATASET_PATH is not a directory"
exit 1
fi
if [ ! -f $RANK_TABLE_FILE ]
then
    echo "error: RANK_TABLE_FILE=$RANK_TABLE_FILE is not a file"
exit 1
fi

export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE_FILE

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`
if [ $# == 3 ]
then
  echo "Start to launch first data augment epochs..."
  for((i=0; i<${DEVICE_NUM}; i++))
  do
      start=`expr $i \* $avg`
      end=`expr $start \+ $gap`
      cmdopt=$start"-"$end
      export DEVICE_ID=$i
      export RANK_ID=$i
      rm -rf ./train_parallel$i
      mkdir ./train_parallel$i
      cp ../*.py ./train_parallel$i
      cp ../*.yaml ./train_parallel$i
      cp -r ../src ./train_parallel$i
      cp -r ../model_utils ./train_parallel$i
      cp -r ../third_party ./train_parallel$i
      cd ./train_parallel$i || exit
      echo "start training for rank $RANK_ID, device $DEVICE_ID"
      env > env.log
      taskset -c $cmdopt python train.py \
          --config_path=$CONFIG_PATH\
          --data_dir=$DATASET_PATH \
          --backbone=$BACKBONE \
          --is_distributed=1 > log.txt 2>&1 &
      cd ..
  done
fi
if [ $# == 4 ]
then
  echo "Start to resume train..."
  for((i=0; i<${DEVICE_NUM}; i++))
  do
      start=`expr $i \* $avg`
      end=`expr $start \+ $gap`
      cmdopt=$start"-"$end
      export DEVICE_ID=$i
      export RANK_ID=$i
      rm -rf ./train_parallel$i
      mkdir ./train_parallel$i
      cp ../*.py ./train_parallel$i
      cp ../*.yaml ./train_parallel$i
      cp -r ../src ./train_parallel$i
      cp -r ../model_utils ./train_parallel$i
      cp -r ../third_party ./train_parallel$i
      cd ./train_parallel$i || exit
      echo "start training for rank $RANK_ID, device $DEVICE_ID"
      env > env.log
      taskset -c $cmdopt python train.py \
          --config_path=$CONFIG_PATH\
          --data_dir=$DATASET_PATH \
          --backbone=$BACKBONE \
          --is_distributed=1 \
          --resume_yolox=$RESUME_CKPT > log.txt 2>&1 &
      cd ..
  done
fi
