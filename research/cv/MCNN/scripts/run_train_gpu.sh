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

export DEVICE_ID=0
export DEVICE_NUM=$1
export RANK_SIZE=$1
export CUDA_VISIBLE_DEVICES="$2"

if [ $1 -lt 1 ] && [ $1 -gt 8 ]
then
    echo "error: DEVICE_NUM=$1 is not in (1-8)"
exit 1
fi


if [ $1 -gt 1 ]
then
    echo "----Multi card task start----"
    mpirun -n $1 --allow-run-as-root --output-filename log_output --merge-stderr-to-stdout \
    python train.py &> log &
else
    python train.py &> log &
fi