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
if [ $# != 3 ]
then
    echo "Please run the script as: "
    echo "bash scripts/run_export_gpu.sh [DEVICE_ID] [CFG_PATH] [CKPT_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CFG_PATH=$(get_real_path $2)
CKPT_PATH=$(get_real_path $3)

export DEVICE_NUM=1
export DEVICE_ID=$1

if [ ! -d "./logs" ]
then
  mkdir "./logs"
fi

echo "Start export for device $DEVICE_ID"

python export.py \
  --checkpoint_path=$CKPT_PATH \
  --device_target='GPU' \
  --device_num=$DEVICE_NUM \
  --config_path=$CFG_PATH \
  --device_id=$DEVICE_ID > ./logs/export_log.txt 2>&1 &
