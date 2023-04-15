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

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: bash run_infer_310.sh [GEN_MINDIR_PATH] [DATA_PATH] [NEED_PREPROCESS] [DEVICE_ID]
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

gen_model=$(get_real_path $1)
data_path=$(get_real_path $2)
preprocess_data_path=$(get_real_path $2)/preprocess_Data

if [ "$3" == "y" ] || [ "$3" == "n" ];then
    need_preprocess=$3
else
  echo "weather need preprocess or not, it's value must be in [y, n]"
  exit 1
fi

if [ "$3" == "n" ]; then
    preprocess_data_path=$data_path
fi

device_id=0
if [ $# == 4 ]; then
    device_id=$4
fi

echo "generator mindir name: "$gen_model
echo "dataset path: "$data_path
echo "need preprocess: "$need_preprocess
echo "device id: "$device_id

function preprocess_data()
{
    if [ -d $preprocess_data_path ]; then
        rm -rf $preprocess_data_path
    fi
    mkdir $preprocess_data_path
    mkdir $preprocess_data_path/data
    mkdir $preprocess_data_path/label
    echo "Start to preprocess attr file..."
    python ../preprocess.py --dataroot=$data_path --experiment_name temp --test_int=1.0 --thres_int=0.5 &> preprocess.log
    echo "Attribute file generates successfully!"
}

function compile_app()
{
    echo "Start to compile source code..."
    cd ../ascend310_infer || exit
    bash build.sh &> build.log
    echo "Compile successfully."
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
    echo "Start to execute inference..."
    ../ascend310_infer/out/main --gen_mindir_path=$gen_model --dataset_path=$preprocess_data_path/ --device_id=$device_id --image_height=128 --image_width=128 &> infer.log
}

function postprocess_data()
{
    echo "Start to postprocess image file..."
    python ../postprocess.py --bin_path="./result_Files/" --target_path="./result_Files/"
    rm -rf ./result_Files/*.bin
}

if [ $need_preprocess == "y" ]; then
    preprocess_data
    if [ $? -ne 0 ]; then
        echo "preprocess attrs failed"
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
    echo "execute inference failed"
    exit 1
fi

postprocess_data
if [ $? -ne 0 ]; then
    echo "postprocess images failed"
    exit 1
fi
