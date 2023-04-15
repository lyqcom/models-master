#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
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
BASE_PATH=$(dirname "$(cd "$(dirname "$0")" || exit; pwd)")
dataset_type='MR'
CONFIG_FILE="${BASE_PATH}/mr_config.yaml"
if [ $# == 2 ]
then
    if [ $2 == "MR" ]; then
        CONFIG_FILE="${BASE_PATH}/mr_config.yaml"
    elif [ $2 == "SUBJ" ]; then
        CONFIG_FILE="${BASE_PATH}/subj_config.yaml"
    elif [ $2 == "SST2" ]; then
        CONFIG_FILE="${BASE_PATH}/sst2_config.yaml"
    else
        echo "error: the selected dataset is not in supported set{MR, SUBJ, SST2}"
    exit 1
    fi
    dataset_type=$2
fi
python eval.py --checkpoint_file_path="$1" --dataset=$dataset_type --config_path=$CONFIG_FILE > eval.log 2>&1 &
