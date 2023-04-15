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
# an simple tutorial as follows, more parameters can be setting
if [ $# != 5 ]
then
    echo "Usage: sh run_standalone_eval_gpu.sh [cifar10] [DEVICE_ID] [SIMCLR_MODEL_PATH] [TRAIN_DATASET_PATH] [EVAL_DATASET_PATH]"
exit 1
else

self_path=$(cd "$(dirname "$0")" || exit; pwd)
export DATASET_NAME=$1
export DEVICE_ID=$2
export SIMCLR_MODEL_PATH=$3
export TRAIN_DATASET_PATH=$4
export EVAL_DATASET_PATH=$5


python ${self_path}/../linear_eval.py --dataset_name=$DATASET_NAME \
               --encoder_checkpoint_path=$SIMCLR_MODEL_PATH \
               --train_dataset_path=$TRAIN_DATASET_PATH \
               --eval_dataset_path=$EVAL_DATASET_PATH \
               --device_id=$DEVICE_ID --device_target="GPU" \
               --run_distribute=False --run_cloudbrain=False > eval_log 2>&1 &
fi
