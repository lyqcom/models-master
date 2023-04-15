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
echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval_gpu.sh DATASET_PATH DATASET_NAME MODEL_CKPT"
echo "for example: bash run_eval_gpu.sh /home/workspace/ag/ ag device0/ckpt0/fasttext-5-118.ckpt"
echo "It is better to use absolute path."
echo "=============================================================================================================="

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASET=$(get_real_path $1)
echo $DATASET
DATANAME=$2
MODEL_CKPT=$(get_real_path $3)
echo "MODEL_CKPT:${MODEL_CKPT}"
echo "DATANAME: ${DATANAME}"
config_path="./${DATANAME}_config.yaml"
echo "config path is : ${config_path}"

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp ../*.yaml ./eval
cp -r ../src ./eval
cp -r ../model_utils ./eval
cp -r ../scripts/*.sh ./eval
cd ./eval || exit
echo "start eval on standalone GPU"

python ../../eval.py \
--config_path $config_path \
--device_target GPU \
--dataset_path $DATASET \
--data_name $DATANAME \
--model_ckpt $MODEL_CKPT> log_fasttext.log 2>&1 &
cd ..
