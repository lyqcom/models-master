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

if [ $# != 4 ]
then
    echo "Usage: sh eval_gpu.sh [DEVICE_ID] [CHECKPOINTPATH] [DATA_DIR] [MODE]"
    exit 1
fi

export DEVICE_ID=$1
export CHECKPOINTPATH=$2
export MODE=$4
echo "start training for device $DEVICE_ID"
env > env.log
cd ..
python eval.py --mode=$4 --data_url=$3 --device_target="GPU" --device_id=$DEVICE_ID --ckpt_url=$CHECKPOINTPATH > log_eval.txt 2>&1 &

cd ..
