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
"""
##############export checkpoint file into mindir model#################
python export.py
"""
import os
import numpy as np

from src.LSTM_CRF import Lstm_CRF
from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.imdb import ImdbParser

from mindspore import Tensor, context
from mindspore import export, load_checkpoint, load_param_into_net

def modelarts_process():
    config.ckpt_file = os.path.join(config.output_path, config.ckpt_file)

def export_lstm_crf():
    """ export lstm """
    context.set_context(
        mode=context.GRAPH_MODE,
        save_graphs=False,
        device_target=config.device_target,
        enable_graph_kernel=False,
        device_id=get_device_id())

    embeddings_size = config.embed_size
    parser = ImdbParser(config.data_CoNLL_path,
                        config.glove_path,
                        config.data_CoNLL_path,
                        embed_size=config.embed_size)

    embeddings, sequence_length, _, _, _, _, tags_to_index_map \
        = parser.get_datas_embeddings(seg=['test'], build_data=False)
    embeddings_table = embeddings.astype(np.float32)

    # DynamicRNN in this network on Ascend platform only support the condition that the shape of input_size
    # and hiddle_size is multiples of 16, this problem will be solved later.
    if config.device_target == 'Ascend':
        pad_num = int(np.ceil(config.embed_size / 16) * 16 - config.embed_size)
        if pad_num > 0:
            embeddings_table = np.pad(embeddings_table, [(0, 0), (0, pad_num)], 'constant')
        embeddings_size = int(np.ceil(config.embed_size / 16) * 16)

    network = Lstm_CRF(vocab_size=embeddings_table.shape[0],
                       tag_to_index=tags_to_index_map,
                       embedding_size=embeddings_size,
                       hidden_size=config.num_hiddens,
                       num_layers=config.num_layers,
                       weight=Tensor(embeddings_table),
                       bidirectional=config.bidirectional,
                       batch_size=config.batch_size,
                       seq_length=sequence_length,
                       is_training=True)

    param_dict = load_checkpoint(os.path.join(config.ckpt_save_path, config.ckpt_path))
    load_param_into_net(network, param_dict)

    input_arr_features = Tensor(np.random.uniform(0.0, 1e5, size=[config.batch_size, sequence_length]).astype(np.int32))
    input_arr_labels = Tensor(np.random.uniform(0.0, 1e5, size=[config.batch_size, sequence_length]).astype(np.int32))
    export(network, input_arr_features, input_arr_labels, file_name=config.file_name, file_format=config.file_format)


if __name__ == '__main__':
    export_lstm_crf()
