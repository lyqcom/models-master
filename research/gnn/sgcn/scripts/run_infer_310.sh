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

if [[ $# -lt 4 || $# -gt 5 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [DATASET_NAME] [DATASET_PATH] [NEED_PREPROCESS] [DEVICE_ID]
    DATASET_NAME must be in ['bitcoin-alpha', 'bitcoin-otc'].
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

model=$1
if [ "$2" == "bitcoin-alpha" ] || [ "$2" == "bitcoin-otc" ];then
    dataset_name=$2
else
  echo "dataset must be in ['bitcoin-alpha', 'bitcoin-otc']"
  exit 1
fi

dataset_path=$3

if [ "$4" == "y" ] || [ "$4" == "n" ];then
    need_preprocess=$4
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 5 ]; then
    device_id=$5
fi

echo "mindir name: "$model
echo "dataset name: "$dataset_name
echo "dataset path: "$dataset_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d ./ascend310_infer/input ]; then
        rm -rf ./ascend310_infer/input
    fi
    mkdir ./ascend310_infer/input
    if [ -d ./ascend310_infer/input/00_data ]; then
        rm -rf ./ascend310_infer/input/00_data
    fi
    mkdir ./ascend310_infer/input/00_data
    if [ -d ./ascend310_infer/input/01_data ]; then
        rm -rf ./ascend310_infer/input/01_data
    fi
    mkdir ./ascend310_infer/input/01_data
    if [ -d ./ascend310_infer/input/02_data ]; then
        rm -rf ./ascend310_infer/input/02_data
    fi
    mkdir ./ascend310_infer/input/02_data
    python preprocess.py --edge_path=$dataset_path --result_path=./ascend310_infer/input/
}

function compile_app()
{
    cd ./ascend310_infer || exit
    bash build.sh &> build.log
}

function infer()
{
    cd - || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result

    ./ascend310_infer/out/main --mindir_path=$model --input0_path=./ascend310_infer/input/00_data --input1_path=./ascend310_infer/input/01_data --device_id=$device_id &> infer.log

}

function cal_acc()
{
    python postprocess.py --result_path=./ascend310_infer/input/ --dataset_name=$dataset_name &> acc.log
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess dataset failed"
        exit 1
    fi
fi
compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi
infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi
cal_acc
if [ $? -ne 0 ]; then
    echo "calculate accuracy failed"
    exit 1
fi