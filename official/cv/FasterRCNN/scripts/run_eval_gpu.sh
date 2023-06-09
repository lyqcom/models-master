#!/bin/bash
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

if [ $# -lt 5 ]
then 
    echo "Usage: 
    bash run_eval_gpu.sh [VALIDATION_JSON_FILE] [CHECKPOINT_PATH] [BACKBONE] [COCO_ROOT] [DEVICE_ID] [MINDRECORD_DIR](option)
    "
exit 1
fi

if [ $3 != "resnet_v1_50" ] && [ $3 != "resnet_v1.5_50" ] && [ $3 != "resnet_v1_101" ] && [ $3 != "resnet_v1_152" ] && [ $3 != "inception_resnet_v2" ]
then 
  echo "error: the selected backbone must be resnet_v1_50, resnet_v1.5_50, resnet_v1_101, resnet_v1_152, inception_resnet_v2"
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
PATH3=$(get_real_path $4)
echo $PATH1
echo $PATH2
echo $PATH3

if [ ! -f $PATH1 ]
then 
    echo "error: ANN_FILE=$PATH1 is not a file"
exit 1
fi 

if [ ! -f $PATH2 ]
then 
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file"
exit 1
fi

if [ ! -d $PATH3 ]
then
    echo "error: COCO_ROOT=$PATH3 is not a dir"
exit 1
fi

mindrecord_dir=$PATH3/MindRecord_COCO_TRAIN/
if [ $# -eq 6 ]
then
    mindrecord_dir=$(get_real_path $6)
    if [ ! -d $mindrecord_dir ]
    then
        echo "error: mindrecord_dir=$mindrecord_dir is not a dir"
    exit 1
    fi
fi
echo $mindrecord_dir

BASE_PATH=$(cd ./"`dirname $0`" || exit; pwd)
if [ $# -ge 1 ]; then
  if [ $3 == 'resnet_v1.5_50' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  elif [ $3 == 'resnet_v1_101' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_101.yaml"
  elif [ $3 == 'resnet_v1_152' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_152.yaml"
  elif [ $3 == 'resnet_v1_50' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
  elif [ $3 == 'inception_resnet_v2' ]; then
    CONFIG_FILE="${BASE_PATH}/../default_config_InceptionResnetV2.yaml"
  else
    echo "Unrecognized parameter"
    exit 1
  fi
else
  CONFIG_FILE="${BASE_PATH}/../default_config.yaml"
fi

export DEVICE_NUM=1
export RANK_SIZE=$DEVICE_NUM
export CUDA_VISIBLE_DEVICES=$5
export RANK_ID=0

if [ -d "../eval" ];
then
    rm -rf ../eval
fi
mkdir ../eval
cd ../eval || exit

echo "start eval for device $CUDA_VISIBLE_DEVICES"
env > env.log
pwd
python ${BASE_PATH}/../eval.py --config_path=$CONFIG_FILE --coco_root=$PATH3 --mindrecord_dir=$mindrecord_dir \
--device_target="GPU" --anno_path=$PATH1 --checkpoint_path=$PATH2 --backbone=$3 &> eval.log &
