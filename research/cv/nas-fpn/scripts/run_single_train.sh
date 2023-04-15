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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash scripts/run_single_train.sh DEVICE_ID MINDRECORD_DIR PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "for example: bash scripts/run_single_train.sh 0 /cache/mindrecord_dir/ /opt/retinanet_nasfpn-300_916.ckpt(optional) 200(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 2 ] && [ $# != 4 ]
then
    echo "Usage: bash scripts/run_single_train.sh [DEVICE_ID] [MINDRECORD_DIR] \
[PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

export DEVICE_ID=$1
MINDRECORD_DIR=$2
PRE_TRAINED=$3
PRE_TRAINED_EPOCH_SIZE=$4

rm -rf LOG$1
mkdir ./LOG$1
cp ./*.py ./LOG$1
cp -r ./src ./LOG$1
cp ./*yaml ./LOG$1
cd ./LOG$1 || exit
echo "start training for device $1"
env > env.log
if [ $# == 2 ]
then
    python train.py  \
    --distribute=False  \
    --mindrecord_dir=$MINDRECORD_DIR  > log.txt 2>&1 &
fi

if [ $# == 4 ]
then
    python train,py  \
    --distribute=False  \
    --mindrecord_dir=$MINDRECORD_DIR \
    --pre_trained=$PRE_TRAINED \
    --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE > log.txt 2>&1 &
fi

cd ../

