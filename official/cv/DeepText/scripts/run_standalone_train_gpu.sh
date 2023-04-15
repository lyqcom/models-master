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

if [ $# -ne 5 ]
then 
    echo "Usage: sh run_distribute_train_gpu.sh [IMGS_PATH] [ANNOS_PATH] [PRETRAINED_PATH] [COCO_TEXT_PARSER_PATH] [DEVICE_ID]"
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
echo $PATH1
PATH2=$(get_real_path $2)
echo $PATH2
PATH3=$(get_real_path $3)
echo $PATH3
PATH4=$(get_real_path $4)
echo $PATH4
DEVICE_ID=0
if [ $# == 5 ]; then
    DEVICE_ID=$5
fi

if [ ! -f $PATH3 ]
then 
    echo "error: PRETRAINED_PATH=$PATH3 is not a file"
exit 1
fi

if [ ! -f $PATH4 ]
then 
    echo "error: COCO_TEXT_PARSER_PATH=$PATH4 is not a file"
exit 1
fi

cp $PATH4 ../src/
rm -rf ./train
mkdir ./train
cp ../*.py ./train
cp ../*.yaml ./train
cp *.sh ./train
cp -r ../src ./train
cp -r ../model_utils ./train
cd ./train || exit

export RANK_SIZE=1

echo "======start training======"

export CUDA_VISIBLE_DEVICES=$DEVICE_ID
python train.py \
  --imgs_path=$PATH1 \
  --annos_path=$PATH2 \
  --run_distribute=False \
  --device_target="GPU" \
  --export_device_target="GPU" \
  --pre_trained=$PATH3 > log.txt 2>&1 &
