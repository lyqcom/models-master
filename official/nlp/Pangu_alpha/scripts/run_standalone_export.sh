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

execute_path=$(pwd)
self_path=$(cd "$(dirname "$0")" || exit; pwd)
export PARAM_INIT_TYPE=fp32
export STRATEGY=$1
export CKPT_PATH=$2
export MODE=$3
export DEVICE_TARGET=$4 # or GPU
export EVAL_TASK=$5
export CKPT_NAME='filerted'

rm -rf ${execute_path}/device0/
mkdir ${execute_path}/device0/
cd ${execute_path}/device0/ || exit
python -s ${self_path}/../predict.py --strategy_load_ckpt_path=$STRATEGY --load_ckpt_path=$CKPT_PATH \
                --load_ckpt_name=$CKPT_NAME --mode=$MODE --run_type=predict --param_init_type=$PARAM_INIT_TYPE \
                --export=1 --distribute=false --eval_task=$EVAL_TASK\
                --device_target=$DEVICE_TARGET >log0.log 2>&1 &
