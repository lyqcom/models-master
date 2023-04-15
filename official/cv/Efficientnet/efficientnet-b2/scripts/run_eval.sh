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
if [ $# != 3 ];then
  echo "usage sh run_eval.sh [device_id(0)] [checkpoint_path] [dataset_path]"
exit 1
fi

# check dataset file
if [ ! -d $3 ];then
    echo "error: DATASET_PATH=$3 is not a directory"
exit 1
fi

if [ ! -f $2 ];then
  echo "error: PATH_CHECKPOINT=$2 is not a file"
  exit 1
fi

export DEVICE_ID=$1
PATH_CHECKPOINT=$2
DATA_DIR=$3

python ./eval.py  \
    --device_id=$DEVICE_ID \
    --checkpoint_path=$PATH_CHECKPOINT \
    --dataset_path=$DATA_DIR > eval.log 2>&1 &
