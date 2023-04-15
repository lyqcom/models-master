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

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run_eval.sh ckpt_filename eval_method pred_image_name(if eval_method is pred)"
echo "for example: bash run_eval.sh 0_8-24_1012.ckpt pred ./demo/001.png"
echo "for example: bash run_eval.sh 0_8-24_1012.ckpt score"
echo "=============================================================================================================="

CKPT=$1
METHOD=$2

if [ $# == 3 ] && [ $METHOD == "pred" ]
then
  PATH1=$3
  python eval.py \
      --device_target="GPU" \
      --ckpt=$CKPT \
      --method=$METHOD \
      --path=$PATH1 > output.eval.log 2>&1 &
else
  python eval.py \
      --device_target="GPU" \
      --ckpt=$CKPT \
      --method=$METHOD > output.eval.log 2>&1 &
fi
