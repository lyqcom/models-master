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

if [ $# != 3 ]
then 
    echo "Usage: bash run_eval.sh [DATA_URL] [CHECKPOINT_PATH / CHECKPOINT_DIR] [DEVICE_ID]"
exit 1
fi

ulimit -u unlimited
export DEVICE_ID=$3
export RANK_SIZE=1
export RANK_ID=0
export DATA_URL=$1
export CKPT_PATH=$2

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp *.sh ./eval
cp -r ../src ./eval
cd ./eval || exit
env > env.log
echo "start evaluation for device $DEVICE_ID"

python eval.py --data_url=$DATA_URL --checkpoint_path=$CKPT_PATH  --device_target=Ascend &> log &

cd ..
