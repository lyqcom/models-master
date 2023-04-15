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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval_gpu.sh DEVICE_ID DATASET_NAME LOAD_CHECKPOINT_PATH"
echo "if eval with up-3d dataset, just run: bash run_eval_gpu.sh 0 up-3d /path/load_ckpt"
echo "=============================================================================================================="


if [ $# != 3 ]
then
    echo "Please specify device id, the eval dataset name or checkpoint path"
    echo "Please try again"
    exit 1
fi

DEVICE_ID=$1
DATASET_NAME=$2
CHECKPOINT=$3

PROJECT_DIR=$(cd "$(dirname "$0")" || exit; pwd)
LOG_DIR=$PROJECT_DIR/../logs

export DEVICE_ID=$DEVICE_ID

python $PROJECT_DIR/../eval.py \
    --checkpoint $CHECKPOINT \
    --batch_size 32 \
    --dataset $DATASET_NAME \
    --num_workers 2 \
    --log_freq 10 \
    --device_target GPU > $LOG_DIR/eval_gpu.log 2>&1 &

echo "The eval log is at /logs/eval_gpu.log"
