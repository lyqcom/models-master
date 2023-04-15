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
if [ $# != 3 ]; then
  echo "Usage: 
        bash run_standalone_train_gpu.sh [config_file] [dataset_dir] [pretrained_backbone]
       " 
  exit 1
fi

get_real_path() {
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)

CONFIG=$(get_real_path $1)
echo "CONFIG: "$CONFIG

DATASET=$(get_real_path $2)
echo "DATASET: "$DATASET

BACKBONE=$(get_real_path $3)
echo "BACKBONE: "$BACKBONE

if [ ! -f $CONFIG ]
then
    echo "error: config=$CONFIG is not a file."
exit 1
fi

if [ ! -d $DATASET ]
then
    echo "error: dataset_root=$DATASET is not a directory."
exit 1
fi

if [ ! -f $BACKBONE ]
then
    echo "error: pretrained_backbone=$BACKBONE is not a file."
exit 1
fi

if [ -d "$BASE_PATH/../train" ];
then
    rm -rf $BASE_PATH/../train
fi
mkdir $BASE_PATH/../train
cd $BASE_PATH/../train || exit

export PYTHONPATH=${BASE_PATH}:$PYTHONPATH

echo "start training on single GPU"
env > env.log
echo
python -u ${BASE_PATH}/../train.py --DEVICE_TARGET GPU --config_path $CONFIG \
  --DATASET_ROOT $DATASET --MODEL_PRETRAINED $BACKBONE &> train.log &
