#!/bin/bash
# Copyright 2020-22 Huawei Technologies Co., Ltd
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
if [ $# != 5 ] ; then
echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh scripts/run_standalone_train.sh DEVICE_TARGET DEVICE_ID EPOCH_SIZE GRADIENT_ACCUMULATE_STEP DATA_PATH"
echo "for example: sh run_standalone_train.sh Ascend 0 52 8 /path/ende-l128-mindrecord00"
echo "It is better to use absolute path."
echo "=============================================================================================================="
exit 1;
fi

rm -rf run_standalone_train
mkdir run_standalone_train
cp -rf ./src/ train.py ./*.yaml ./run_standalone_train
cd run_standalone_train || exit

FILE_LIMIT=$(ulimit -n)
if [ $FILE_LIMIT -lt 2048 ] ; then
ulimit -n 2048;
fi

export DEVICE_TARGET=$1
EPOCH_SIZE=$3
GRADIENT_ACCUMULATE_STEP=$4
DATA_PATH=$5

if [ $DEVICE_TARGET == 'Ascend' ];then
    export DEVICE_ID=$2
    python train.py  \
        --config_path="./default_config_large.yaml" \
        --distribute="false" \
        --epoch_size=$EPOCH_SIZE \
        --accumulation_steps=$GRADIENT_ACCUMULATE_STEP \
        --device_target=$DEVICE_TARGET \
        --enable_save_ckpt="true" \
        --enable_lossscale="true" \
        --do_shuffle="true" \
        --checkpoint_path="" \
        --save_checkpoint_steps=2500 \
        --save_checkpoint_num=30 \
        --data_path=$DATA_PATH > log.txt 2>&1 &
elif [ $DEVICE_TARGET == 'GPU' ];then
    export CUDA_VISIBLE_DEVICES="$2"

    python train.py  \
        --config_path="./default_config_large_gpu.yaml" \
        --distribute="false" \
        --epoch_size=$EPOCH_SIZE \
        --device_target=$DEVICE_TARGET \
        --enable_save_ckpt="true" \
        --enable_lossscale="true" \
        --do_shuffle="true" \
        --checkpoint_path="" \
        --save_checkpoint_steps=2500 \
        --save_checkpoint_num=30 \
        --data_path=$DATA_PATH > log.txt 2>&1 &
else
    echo "Not supported device target."
fi

if [ $FILE_LIMIT -lt 2048 ] ; then
ulimit -n $FILE_LIMIT;
fi

cd ..
