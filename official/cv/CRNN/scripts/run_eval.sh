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

if [ $# != 5 ] && [ $# != 4 ]; then
  echo "Usage: bash run_eval.sh [DATASET_NAME] [DATASET_PATH] [CHECKPOINT_PATH] [DEVICE_ID] [PLATFORM](optional) "
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET_NAME=$1
PATH1=$(get_real_path $2)
PATH2=$(get_real_path $3)
export DEVICE_ID=$4
if [ $# == 5 ]; then
  PLATFORM=$5
else
  PLATFORM="Ascend"
fi

if [ ! -d $PATH1 ]; then
  echo "error: DATASET_PATH=$PATH1 is not a directory"
  exit 1
fi

if [ ! -f $PATH2 ]; then
  echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
  exit 1
fi

run_ascend() {
  ulimit -u unlimited
  export DEVICE_NUM=1
  export RANK_SIZE=$DEVICE_NUM
  export RANK_ID=0

  if [ -d "eval" ]; then
    rm -rf ./eval
  fi
  mkdir ./eval
  cp ./*.py ./eval
  cp -r ./src ./eval
  cp ./*yaml ./eval
  cd ./eval || exit
  env >env.log
  echo "start evaluation for device $DEVICE_ID"
  python eval.py --eval_dataset=$DATASET_NAME --eval_dataset_path=$PATH1 --checkpoint_path=$PATH2 --device_target=Ascend> log.txt 2>&1 &
  cd ..
}

run_gpu() {
  if [ -d "eval" ]; then
    rm -rf ./eval
  fi
  mkdir ./eval
  cp ./*.py ./eval
  cp -r ./src ./eval
  cp ./*yaml ./eval
  cd ./eval || exit
  env >env.log
  python eval.py --eval_dataset=$DATASET_NAME \
                 --eval_dataset_path=$PATH1 \
                 --checkpoint_path=$PATH2 \
                 --device_target=GPU > log.txt 2>&1 &
  cd ..
}

if [ "Ascend" == $PLATFORM ]; then
  run_ascend $PATH1 $PATH2
elif [ "GPU" == $PLATFORM ]; then
  run_gpu $PATH1 $PATH2
else
  echo "error: PLATFORM=$PLATFORM is not support, only support Ascend and GPU."
fi

