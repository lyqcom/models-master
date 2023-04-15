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

if [ $# -lt 2 ]
then
  echo "Usage: sh run_export_gpu.sh [PRE_TRAINED] [BATCH_SIZE] [FILE_NAME](optional)"
  exit 1
fi

BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH

cd ..

if [ $3 ] #file name
then
  python3 ${BASEPATH}/../export.py \
            --pretrained=$1 \
            --device_target='GPU' \
            --batch_size=$2 \
            --file_format=MINDIR \
            --file_name=$3
else
  python3 ${BASEPATH}/../export.py \
            --pretrained=$1 \
            --device_target='GPU' \
            --batch_size=$2 \
            --file_format=MINDIR
fi
