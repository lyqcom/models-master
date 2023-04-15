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

if [ $# != 2 ]
then
    echo "Usage: bash run_export_gpu.sh [DATASET] [CKPT_FILE]"
exit 1
fi

DATASET=$1
CKPT_FILE=$2

export GLOG_v=3

echo "DATASET: $DATASET   CKPT_FILE: $CKPT_FILE"

if [ ! -e $CKPT_FILE ]
then
    echo "ckpt file not exists"
exit 1
fi

if [ ! -d "mindir" ]
then
    mkdir mindir
fi

if [ ! -d "logs" ]
then
    mkdir logs
fi

echo "Start exporting..."

nohup python -u export.py --dataset $DATASET --ckpt_file $CKPT_FILE --file_name mindir/$DATASET > logs/export_$DATASET.log 2>&1 &