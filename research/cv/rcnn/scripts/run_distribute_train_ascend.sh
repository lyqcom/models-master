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
if [ $# -ne 1 ]; then
  echo "Usage: sh run_distribute_train_ascend.sh [RANK_TABLE_FILE]"
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

RANK_TABLE_PATH=$(get_real_path $1)
echo $RANK_TABLE_PATH

if [ ! -f $RANK_TABLE_PATH ]; then
  echo "error: RANK_TABLE_FILE=$RANK_TABLE_PATH is not a file"
  exit 1
fi

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=$RANK_TABLE_PATH

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  rm -rf ./train_parallel$i
  mkdir ./train_parallel$i
  cp ../*.py ./train_parallel$i
  cp ../*.yaml ./train_parallel$i
  cp -r ../src ./train_parallel$i
  cd train_parallel$i || exit
  ln -s ../../data data
  echo "start training finetune for rank $RANK_ID, device $DEVICE_ID"
  env >env.log
  python train.py --config_path=./default_config.yaml --device_id=${DEVICE_ID} --step 0 >log_train_finetune 2>&1 &
  cd ..
done

wait

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  cd ./train_parallel$i || exit
  echo "start training svm for rank $RANK_ID, device $DEVICE_ID"
  python train.py --config_path=./default_config.yaml --device_id=${DEVICE_ID} --step 1 >log_train_svm 2>&1 &
  cd ..
done

wait

for ((i = 0; i < ${DEVICE_NUM}; i++)); do
  export DEVICE_ID=$i
  export RANK_ID=$i
  cd ./train_parallel$i || exit
  echo "start training regression for rank $RANK_ID, device $DEVICE_ID"
  python train.py --config_path=./default_config.yaml --device_id=${DEVICE_ID} --step 2 >log_train_regression 2>&1 &
  cd ..
done
