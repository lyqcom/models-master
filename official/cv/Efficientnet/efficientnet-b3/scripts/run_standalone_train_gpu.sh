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
if [ $# != 2 ]; then
  echo "Usage: bash run_eval_gpu.sh [DEVICE_ID] [DATASET_PATH] "
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DEVICE_ID=$1
DATA_DIR=$(get_real_path $2)
export CUDA_VISIBLE_DEVICES=$DEVICE_ID

BASEPATH=$(cd "`dirname $0`" || exit; pwd)

if [ -d "./train_standalone" ]; then
  rm -rf ./train_standalone
fi

mkdir train_standalone
cp $BASEPATH/../*.py ./train_standalone
cp $BASEPATH/*.sh ./train_standalone
cp -r $BASEPATH/../src ./train_standalone
cd ./train_standalone || exit

python ./train.py  \
    --device_target=GPU  \
    --dataset_path=$DATA_DIR > log.txt 2>&1 &
cd ..