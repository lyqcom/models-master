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

[ $# -ne 1 ] && {
    echo "Usage: bash scripts/run_distribute_train_gpu.sh [DATASET_PATH]"
    exit 1
}

DATA_DIR=$1
export DEVICE_ID=0
export RANK_SIZE=8

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
CONFIG_FILE="${BASE_PATH}/../default_config_gpu.yaml"

echo "start training"

mpirun -n $RANK_SIZE --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python train.py --config_path=$CONFIG_FILE --dataset_path=$DATA_DIR \
--platform='GPU' > train.log 2>&1 &
