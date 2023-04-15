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

if [ $# != 4 ]
then
    echo "============================================================================================================"
    echo "Please run the script as: "
    echo "bash scripts/run_eval.sh [TASK_NAME] [DEVICE_TARGET] [MODEL_DIR] [DATA_DIR]"
    echo "============================================================================================================"
exit 1
fi

echo "===============================================start evaling================================================"

task_name=$1
device_target=$2
model_dir=$3
data_dir=$4

if [ -z $DEVICE_ID ]
then
    DEVICE_ID=0
fi

mkdir -p ms_log
PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
CUR_DIR=`pwd`
export GLOG_log_dir=${CUR_DIR}/ms_log
export GLOG_logtostderr=0
python ${PROJECT_DIR}/../eval.py \
    --task_name=$task_name \
    --device_target=$device_target \
    --device_id=$DEVICE_ID \
    --model_dir=$model_dir \
    --data_dir=$data_dir > eval_log.txt 2>&1 &
