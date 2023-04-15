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


export DEVICE_NUM=1
export DEVICE_ID=0
export MODE='test'
echo "start training for device $DEVICE_ID"
env > env.log
python eval.py --run_distribute=0 --device_num=$DEVICE_NUM --device_id=$DEVICE_ID --mode=$MODE> log_eval.txt 2>&1 &

cd ..
