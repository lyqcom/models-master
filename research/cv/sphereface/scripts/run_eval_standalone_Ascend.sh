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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_train_Ascend.sh [DEVICE_ID] [CKPT_FILES]"
echo "=============================================================================================================="

if [ $# != 2 ]
then
    echo "Usage: sh run_train_cpu.sh [DEVICE_ID] [CKPT_FILES]"
    exit 1
fi

export DEVICE_ID=$1
BASEPATH=$(cd "`dirname $0`" || exit; pwd)
export PYTHONPATH=${BASEPATH}:$PYTHONPATH
if [ -d "./eval_Ascend" ];
then
    rm -rf ./eval_Ascend
fi
mkdir ./eval_Ascend
cd ./eval_Ascend || exit
mkdir src
cd ../
cp ../*.py ./eval_Ascend
cp ../*.yaml ./eval_Ascend
cp -r ../src ./eval_Ascend/
cd ./eval_Ascend

nohup python ${BASEPATH}/../eval.py \
    --is_distributed=0 \
    --device_target='Ascend' \
    --device_id=$1 \
    --ckpt_files=$2 > eval.log 2>&1 &
