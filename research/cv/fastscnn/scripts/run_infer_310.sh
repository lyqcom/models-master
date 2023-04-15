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

if [ $# != 6 ]; then
    echo "Usage: sh run_infer_310.sh [model_path] [data_path]" \
       "[out_image_path] [image_height] [image_width] [device_id]"
exit 1
fi

get_real_path_name() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

get_real_path() {
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)/"
    fi
}

model=$(get_real_path_name $1)
data_path=$(get_real_path $2)
out_image_path=$(get_real_path $3)
image_height=$4
image_width=$5
device_id=$6

echo "model path: "$model
echo "dataset path: "$data_path
echo "out image path: "$out_image_path
echo "image_height: "$image_height
echo "image_width: "$image_width 
echo "device id: "$device_id

function compile_app()
{
    cd ../ascend310_infer/ || exit
    if [ -f "Makefile" ]; then
        make clean
    fi
    sh build.sh &> build.log
}

function preprocess()
{
    cd ../ || exit
    echo "\nstart preprocess"
    echo "waitting for preprocess finish..."
    python preprocess.py --out_dir=$out_image_path --image_path=$data_path --image_height=$image_height --image_width=$image_width > preprocess.log 2>&1
    echo "preprocess finished! you can see the log in preprocess.log!"
}

function infer()
{
    cd ./scripts || exit
    if [ -d result_Files ]; then
        rm -rf ./result_Files
    fi
    if [ -d time_Result ]; then
        rm -rf ./time_Result
    fi
    mkdir result_Files
    mkdir time_Result
    echo "\nstart infer..."
    echo "waitting for infer finish..."
    ../ascend310_infer/out/main --model_path=$model --dataset_path=$out_image_path/images/ --device_id=$device_id > infer.log 2>&1
    echo "infer finished! you can see the log in infer.log!"
}

function cal_mIoU()
{
    echo "\nstart calculate mIoU..."
    echo "waitting for calculate finish..."
    python ../cal_mIoU.py --label_path=$out_image_path/labels/ --output_path=./result_Files --image_height=$image_height --image_width=$image_width --save_mask=0 >acc.log 2>&1
    echo "infer finished! you can see the log in acc.log\n"
}

compile_app
if [ $? -ne 0 ]; then
    echo "compile app code failed"
    exit 1
fi

preprocess
if [ $? -ne 0 ]; then
    echo "execute preprocess failed"
    exit 1
fi

infer
if [ $? -ne 0 ]; then
    echo " execute inference failed"
    exit 1
fi

cal_mIoU
if [ $? -ne 0 ]; then
    echo "calculate mIoU failed"
    exit 1
fi
