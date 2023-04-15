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

if [ $# != 3 ]; then
    echo "Usage: sh run_infer_310.sh [MODEL_PATH] [DATA_PATH] [DEVICE_ID]
    DEVICE_ID is optional, it can be set by environment variable DEVICE_ID, otherwise the value is zero"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

MODEL=$(get_real_path $1)
DATA_PATH=$(get_real_path $2)
DEVICE_ID=$3

echo "$MODEL"
echo "$DATA_PATH"
echo "$DEVICE_ID"

function compile_app()
{
    cd ../ascend310_infer || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log

    if [ $? -ne 0 ]; then
        echo "compile app code failed"
        exit 1
    fi
    cd - || exit
}

function infer()
{
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
     if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    img_path=$DATA_PATH/leftImg8bit/val
    ../ascend310_infer/out/main --model_path="$MODEL" --dataset_path="$img_path" --device_id=$DEVICE_ID &> infer.log

    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

function cal_acc()
{
    if [ -d output ]; then
        rm -rf ./output
    fi
     if [ -d output_img ]; then
        rm -rf ./output_img
    fi
    mkdir output
    mkdir output_img
    gt_path=$DATA_PATH
    RESULT_FILES=$(realpath -m "./result_Files")
    OUTPUT_PATH=$(realpath -m "./output")
    python ../postprocess.py --dataset_path="$gt_path" --result_path="${RESULT_FILES}" --output_path="${OUTPUT_PATH}" &> acc.log
    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi

}

compile_app
infer
cal_acc
