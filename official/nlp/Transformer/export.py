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
""" export checkpoint file into models"""

import numpy as np

import mindspore as ms
from mindspore import Tensor

from src.transformer_model import TransformerModel
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id
from eval import load_weights


config.batch_size = config.batch_size_ev
config.hidden_dropout_prob = config.hidden_dropout_prob_ev
config.attention_probs_dropout_prob = config.attention_probs_dropout_prob_ev

ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    ms.set_context(device_id=get_device_id())

def modelarts_pre_process():
    pass

@moxing_wrapper(pre_process=modelarts_pre_process)
def export_transformer():
    """ export_transformer """
    tfm_model = TransformerModel(config=config, is_training=False, use_one_hot_embeddings=False)

    parameter_dict = load_weights(config.model_file)
    ms.load_param_into_net(tfm_model, parameter_dict)

    source_ids = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))
    source_mask = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))

    ms.export(tfm_model, source_ids, source_mask, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_transformer()
