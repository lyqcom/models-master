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
"""Seq2seq Init."""
from config.config import Seq2seqConfig
from .seq2seq import Seq2seqModel
from .seq2seq_for_train import Seq2seqTraining, LabelSmoothedCrossEntropyCriterion, \
    Seq2seqNetworkWithLoss, Seq2seqTrainOneStepWithLossScaleCell
from .bleu_calculate import bleu_calculate
from .seq2seq_for_infer_onnx import infer_onnx

__all__ = [
    "Seq2seqTraining",
    "LabelSmoothedCrossEntropyCriterion",
    "Seq2seqTrainOneStepWithLossScaleCell",
    "Seq2seqNetworkWithLoss",
    "Seq2seqModel",
    "Seq2seqConfig",
    "bleu_calculate",
    "infer_onnx"
]
