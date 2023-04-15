#!/bin/bash
# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

if [ $# != 2 ] && [ $# != 3 ]
then
    echo "Usage: sh run_distribute_train.sh [DATA_DIR] [RANK_TABLE] [PRETRAINED_BACKBONE]"
    echo "   or: sh run_distribute_train.sh [DATA_DIR] [RANK_TABLE]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

current_exec_path=$(pwd)
echo ${current_exec_path}

dirname_path=$(dirname "$(pwd)")
echo ${dirname_path}

export PYTHONPATH=${dirname_path}:$PYTHONPATH

SCRIPT_NAME='train.py'

rm -rf ${current_exec_path}/device*

ulimit -c unlimited

DATA_DIR=$(get_real_path $1)
RANK_TABLE=$(get_real_path $2)
PRETRAINED_BACKBONE=''

if [ ! -d $DATA_DIR ]
then
    echo "error: DATA_DIR=$DATA_DIR is not a directory"
exit 1
fi

if [ $# == 3 ]
then
    PRETRAINED_BACKBONE=$(get_real_path $3)
    if [ ! -f $PRETRAINED_BACKBONE ]
    then
        echo "error: PRETRAINED_PATH=$PRETRAINED_BACKBONE is not a file"
    exit 1
    fi
fi

echo $DATA_DIR
echo $RANK_TABLE
echo $PRETRAINED_BACKBONE

export RANK_TABLE_FILE=$RANK_TABLE
export RANK_SIZE=8
export GLOG_v=3

cpus=`cat /proc/cpuinfo| grep "processor"| wc -l`
avg=`expr $cpus \/ $RANK_SIZE`
gap=`expr $avg \- 1`

config_path="${dirname_path}/reid_8p_ascend_config.yaml"
echo "config path is : ${config_path}"

echo 'start training'
for((i=0;i<=$RANK_SIZE-1;i++));
do
    echo 'start rank '$i
    start=`expr $i \* $avg`
    end=`expr $start \+ $gap`
    cmdopt=$start"-"$end
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i  || exit
    export RANK_ID=$i
    dev=`expr $i + 0`
    export DEVICE_ID=$dev
    taskset -c $cmdopt python ${dirname_path}/${SCRIPT_NAME} \
        --config_path=$config_path \
        --is_distributed=1 \
        --data_dir=$DATA_DIR \
        --pretrained=$PRETRAINED_BACKBONE > train.log  2>&1 &
done

echo 'running'
