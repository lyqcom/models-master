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

ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export NPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
RANK_TABLE_FILE=$(realpath $1)
export RANK_TABLE_FILE

python3 ../train.py --config_file=../src/spnas_distributed.yml