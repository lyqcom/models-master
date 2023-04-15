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

if [[ $# -lt 5 || $# -gt 6 ]]; then
    echo "Usage: bash run_infer_310.sh [MINDIR_PATH] [MODEL_NAME] [DATA_FILE_PATH] [EVAL_BATCH_SIZE] [NEED_PREPROCESS] [DEVICE_ID]（optional）
    NEED_PREPROCESS means weather need preprocess or not, it's value is 'y' or 'n'.
    DEVICE_ID is optional, it can be set by environment variable device_id, otherwise the value is zero"
exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}
model=$(get_real_path $1)
model_name=$2
eval_data_file_path=$(get_real_path $3)
eval_batch_size=$4

if [ "$5" == "y" ] || [ "$5" == "n" ];then
    need_preprocess=$5
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

device_id=0
if [ $# == 6 ]; then
    device_id=$6
fi

echo "mindir name: "$model
echo "model_name: "$model_name
echo "eval_data_file_path: "$eval_data_file_path
echo "eval_batch_size: "$eval_batch_size
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d preprocess_Result ]; then
        rm -rf ./preprocess_Result
    fi
    mkdir preprocess_Result
    python ../preprocess.py --eval_data_file_path=$eval_data_file_path --eval_batch_size=$eval_batch_size
}

function compile_app()
{
    cd ../ascend310_infer || exit
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

    ../ascend310_infer/out/main --mindir_path=$model \
                                --input0_path=./preprocess_Result/00_data \
                                --input1_path=./preprocess_Result/01_data \
                                --input2_path=./preprocess_Result/02_data \
                                --input3_path=./preprocess_Result/03_data \
                                --input4_path=./preprocess_Result/04_data \
                                --device_id=$device_id &> infer.log

}

function cal_acc()
{
    python ../postprocess.py --model_name=$model_name \
                             --batch_size=$eval_batch_size \
                             --label_dir=./preprocess_Result/04_data \
                             --result_dir=./result_Files &> acc.log
}

# start run
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