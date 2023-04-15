#!/usr/bin/env bash
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

PWD_DIR=`pwd`
scripts_path=$(dirname $0)
DATA=$1
LOAD_CHECKPOINT_PATH=$2

BERT_DIR=$DATA/cased_L-24_H-1024_A-16
WN_CPT_EMBEDDING_PATH=$DATA/KB_embeddings/wn_concept2vec.txt
NELL_CPT_EMBEDDING_PATH=$DATA/KB_embeddings/nell_concept2vec.txt

if [ ! -d log ]; then
    mkdir log
fi

python3 $scripts_path/../run_KTNET_record_eval.py \
  --device_target "Ascend" \
  --device_id 0 \
  --batch_size 12 \
  --do_train false \
  --do_predict true \
  --do_lower_case false \
  --init_pretraining_params $BERT_DIR/params \
  --load_pretrain_checkpoint_path $BERT_DIR/roberta.ckpt \
  --load_checkpoint_path $LOAD_CHECKPOINT_PATH \
  --train_file $DATA/ReCoRD/train.json \
  --predict_file $DATA/ReCoRD/dev.json \
  --train_mindrecord_file $DATA/ReCoRD/train.mindrecord \
  --predict_mindrecord_file $DATA/ReCoRD/dev.mindrecord \
  --vocab_path $BERT_DIR/vocab.txt \
  --bert_config_path $BERT_DIR/bert_config.json \
  --freeze false \
  --save_steps 4000 \
  --weight_decay 0.01 \
  --warmup_proportion 0.1 \
  --learning_rate 6e-5 \
  --epoch 4 \
  --max_seq_len 384 \
  --doc_stride 128 \
  --wn_concept_embedding_path $WN_CPT_EMBEDDING_PATH \
  --nell_concept_embedding_path $NELL_CPT_EMBEDDING_PATH \
  --use_wordnet true \
  --use_nell true \
  --random_seed 45 \
  --save_finetune_checkpoint_path $PWD_DIR/output/finetune_checkpoint \
  --data_url $DATA \
  --checkpoints output/ 1>$PWD_DIR/log/eval_record.log 2>&1 &
