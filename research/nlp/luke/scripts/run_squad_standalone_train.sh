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

DATA=$1
MODEL_FILE=$2

python ./run_squad_train.py --data $DATA --model_file $MODEL_FILE  --warmup_proportion 0.09 --num_train_epochs 2 --train_batch_size 8 --learning_rate 12e-6 --dataset_sink_mode True