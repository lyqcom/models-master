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
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_1p_train.sh DEVICE_ID EPOCH_SIZE LR DATASET RANK_TABLE_FILE PRE_TRAINED PRE_TRAINED_EPOCH_SIZE"
echo "for example: sh run_1p_train.sh 8 500 0.2 coco /opt/ssd-300.ckpt(optional) 200(optional)"
echo "It is better to use absolute path."
echo "================================================================================================================="

if [ $# != 4 ] && [ $# != 6 ]
then
    echo "Usage: sh run_1p_train.sh [DEVICE_ID] [EPOCH_SIZE] [LR] [DATASET] \
[PRE_TRAINED](optional) [PRE_TRAINED_EPOCH_SIZE](optional)"
    exit 1
fi

# Before start 1pc train, first create mindrecord files.
BASE_PATH=$(cd "`dirname $0`" || exit; pwd)
cd $BASE_PATH/../ || exit
python train.py --only_create_dataset=True --dataset=$4

echo "After running the script, the network runs in the background. The log will be generated in LOGx/log.txt"

DEVICE_ID=$1
EPOCH_SIZE=$2
LR=$3
DATASET=$4
PRE_TRAINED=$5
PRE_TRAINED_EPOCH_SIZE=$6

rm -rf LOG$1
mkdir ./LOG$1
cp ./*.py ./LOG$1
cp -r ./src ./LOG$1
cp -r ./scripts ./LOG$1
cd ./LOG$1 || exit

echo "start training for device $1"
env > env.log
if [ $# == 4 ]
then
    python train.py  \
    --lr=$LR \
    --dataset=$DATASET \
    --device_id=$DEVICE_ID  \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
fi

if [ $# == 6 ]
then
    python train.py  \
    --lr=$LR \
    --dataset=$DATASET \
    --device_id=$DEVICE_ID  \
    --pre_trained=$PRE_TRAINED \
    --pre_trained_epoch_size=$PRE_TRAINED_EPOCH_SIZE \
    --epoch_size=$EPOCH_SIZE > log.txt 2>&1 &
fi

cd ../
