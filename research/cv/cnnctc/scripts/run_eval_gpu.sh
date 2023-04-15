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

if [ $# -ne 2 ]
then
    echo "Usage: bash run_eval_gpu.sh [TRAINED_CKPT] [TEST_DATASET_PATH]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

PATH1=$(get_real_path $1)
PATH2=$(get_real_path $2)

echo $PATH1
echo $PATH2

if [ ! -f $PATH1 ]
then
    echo "error: TRAINED_CKPT=$PATH1 is not a file"
exit 1
fi

ulimit -u unlimited
export DEVICE_ID=0

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ./*.py ./eval
cp ./scripts/*.sh ./eval
cp -r ./src ./eval
cp ./*yaml ./eval
cd ./eval || exit
echo "start inferring for device $DEVICE_ID"
env > env.log
python eval.py --device_target="GPU" \
               --device_id=$DEVICE_ID \
               --CHECKPOINT_PATH=$PATH1 \
               --TEST_DATASET_PATH=$PATH2 &> eval.log
cd .. || exit
