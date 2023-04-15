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
# an simple tutorial as follows, more parameters can be setting

function prepocess()
{
    cd ..
    python preprocess.py
    
}
function compile_app()
{   
    cd ascend310_infer || exit
    
    if [ -f "Makefile" ]; then
        make clean
    fi

    bash build.sh 

  
}

function infer()
{
    cd ./out
    if [ -d result_Files ]; then
        rm -rf ./result_Files
        mkdir result_Files
    else
        mkdir result_Files
    fi
    
    ./main --mindir_path ../../relationnet.mindir --input0_path ../../data
    if [ $? -ne 0 ]; then
        echo "execute inference failed"
        exit 1
    fi
}

function cal_acc()
{
    cd 。。、
    python postprocess.py
    if [ $? -ne 0 ]; then
        echo "calculate accuracy failed"
        exit 1
    fi

}
prepocess
compile_app
infer
cal_acc
