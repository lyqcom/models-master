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

execute_path=$(pwd)
echo ${execute_path}
self_path=$(cd "$(dirname "$0")" || exit; pwd)
echo ${self_path}

export RANK_SIZE=$1
RANK_START=$2
EPOCH_SIZE=$3
VOCAB_SIZE=$4
EMB_DIM=$5
DATASET=$6
ENV_SH=$7
MODE=$8
export RANK_TABLE_FILE=$9
DEVICE_START=0
# shellcheck source=/dev/null
source $ENV_SH

if [ $MODE == "forward_unique" ]; then
  export GRAPH_OP_RUN=1
  export HCCL_WHITELIST_DISABLE=1
  rm -rf ${execute_path}/train_parallel
  mkdir ${execute_path}/train_parallel
  cd ${execute_path}/train_parallel || exit
  mpirun --allow-run-as-root -n $RANK_SIZE --output-filename log_output --merge-stderr-to-stdout \
    python -s ${self_path}/../train_and_eval_auto_parallel.py --data_path=$DATASET --epochs=$EPOCH_SIZE --vocab_size=$VOCAB_SIZE --emb_dim=$EMB_DIM --dropout_flag=True --sparse=True >train_deep.log 2>&1 &
else
  for((i=0;i<=7;i++));
  do
    export RANK_ID=$[i+RANK_START]
    export DEVICE_ID=$[i+DEVICE_START]
    rm -rf ${execute_path}/device_$RANK_ID
    mkdir ${execute_path}/device_$RANK_ID
    cd ${execute_path}/device_$RANK_ID || exit
    if [ $MODE == "host_device_mix" ]; then
      python -s ${self_path}/../train_and_eval_auto_parallel.py --data_path=$DATASET --epochs=$EPOCH_SIZE --vocab_size=$VOCAB_SIZE --emb_dim=$EMB_DIM --dropout_flag=True --host_device_mix=1 >train_deep$i.log 2>&1 &
    elif [ $MODE == "field_slice_host_device_mix" ]; then
      python -s ${self_path}/../train_and_eval_auto_parallel.py --data_path=$DATASET --epochs=$EPOCH_SIZE --vocab_size=$VOCAB_SIZE --emb_dim=$EMB_DIM --dropout_flag=True --host_device_mix=1 --full_batch=True --field_slice=True >train_deep$i.log 2>&1 &
    else
      python -s ${self_path}/../train_and_eval_auto_parallel.py --data_path=$DATASET --epochs=$EPOCH_SIZE --vocab_size=$VOCAB_SIZE --emb_dim=$EMB_DIM --dropout_flag=True --host_device_mix=0 >train_deep$i.log 2>&1 &
    fi
  done
fi
