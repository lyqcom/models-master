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

model_path=$1
output_model_name=$2

export install_path=/usr/local/Ascend/

export ASCEND_ATC_PATH=${install_path}/atc
export LD_LIBRARY_PATH=${install_path}/atc/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/python3.7.5/bin:${install_path}/atc/ccec_compiler/bin:${install_path}/atc/bin:$PATH
export PYTHONPATH=${install_path}/atc/python/site-packages:${install_path}/latest/atc/python/site-packages/auto_tune.egg/auto_tune:${install_path}/atc/python/site-packages/schedule_search.egg
export ASCEND_OPP_PATH=${install_path}/opp
export ASCEND_SLOG_PRINT_TO_STDOUT=0

atc --model=$model_path \
--framework=1 \
--output=$output_model_name \
--input_format=NCHW --input_shape="actual_input_1:1,3,256,192" \
--enable_small_channel=1 \
--log=error \
--soc_version=Ascend310 \
--output_type=FP32 \
