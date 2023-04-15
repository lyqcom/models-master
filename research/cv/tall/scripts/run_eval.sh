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
export DEVICE_NUM=1
export DEVICE_ID=0
export RANK_SIZE=1
export RANK_ID=0

if [ $# != 3 ]
then
    echo "Usage: sh run_eval.sh [CHECKPOINT_PATH] [EVAL_DATA_DIR] [DEVICE_ID]."
    exit 1
fi

if [ ! -f $1 ]
then
    echo "error: CHECKPOINT_PATH=$1 is not a file"
    echo "Usage: sh run_eval.sh [CHECKPOINT_PATH] [EVAL_DATA_DIR] [DEVICE_ID]."
exit 1
fi

if [ ! -d $2 ]
then
    echo "error: EVAL_DATA_DIR=$2 is not a directory"
    echo "Usage: sh run_eval.sh [CHECKPOINT_PATH] [EVAL_DATA_DIR] [DEVICE_ID]."
exit 1
fi

python eval.py --checkpoint_path=$1 --eval_data_dir=$2 --device_id=$3 > eval.log 2>&1 &
