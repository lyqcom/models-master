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
if [ $# != 2 ]
then
    echo "Usage: bash scripts/run_train_distributed_gpu.sh [DATA_PATH] [EPOCH_SIZE]"
exit 1
fi

DATA_PATH=$1
EPOCH_SIZE=$2
mpirun -n 8 --output-filename log_output --merge-stderr-to-stdout --allow-run-as-root \
    python train.py --platform GPU --model ResNet --max_epoch $EPOCH_SIZE --n_epochs 300 \
    --device_num 8 --dataroot $DATA_PATH  > output.train.log 2>&1 &
