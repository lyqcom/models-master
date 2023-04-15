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
"""Transformer evaluation script."""

import os
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.parameter import Parameter
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
import mindspore.dataset as ds
import mindspore.dataset.transforms as deC

from src.transformer_model import TransformerModel
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id

config.dtype = ms.float32
config.compute_type = ms.float16
config.batch_size = config.batch_size_ev
config.hidden_dropout_prob = config.hidden_dropout_prob_ev
config.attention_probs_dropout_prob = config.attention_probs_dropout_prob_ev

def load_test_data(batch_size=1, data_file=None):
    """
    Load test dataset
    """
    data_set = ds.MindDataset(data_file,
                              columns_list=["source_eos_ids", "source_eos_mask",
                                            "target_sos_ids", "target_sos_mask",
                                            "target_eos_ids", "target_eos_mask"],
                              shuffle=False)
    type_cast_op = deC.TypeCast(ms.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="source_eos_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="source_eos_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="target_sos_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="target_sos_mask")
    data_set = data_set.map(operations=type_cast_op, input_columns="target_eos_ids")
    data_set = data_set.map(operations=type_cast_op, input_columns="target_eos_mask")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set.channel_name = 'transformer'
    return data_set


class TransformerInferCell(nn.Cell):
    """
    Encapsulation class of transformer network infer.
    """
    def __init__(self, network):
        super(TransformerInferCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self,
                  source_ids,
                  source_mask):
        predicted_ids = self.network(source_ids, source_mask)
        return predicted_ids


def load_weights(model_path):
    """
    Load checkpoint as parameter dict, support both npz file and mindspore checkpoint file.
    """
    if model_path.endswith(".npz"):
        ms_ckpt = np.load(model_path)
        is_npz = True
    else:
        ms_ckpt = ms.load_checkpoint(model_path)
        is_npz = False

    weights = {}
    for msname in ms_ckpt:
        infer_name = msname
        if "tfm_decoder" in msname:
            infer_name = "tfm_decoder.decoder." + infer_name
        if is_npz:
            weights[infer_name] = ms_ckpt[msname]
        else:
            weights[infer_name] = ms_ckpt[msname].data.asnumpy()
    weights["tfm_decoder.decoder.tfm_embedding_lookup.embedding_table"] = \
        weights["tfm_embedding_lookup.embedding_table"]

    parameter_dict = {}
    for name in weights:
        parameter_dict[name] = Parameter(Tensor(weights[name]), name=name)
    return parameter_dict


def modelarts_pre_process():
    config.output_file = os.path.join(config.output_path, config.output_file)
    config.data_file = os.path.join(config.data_file, config.data_file_name)

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_transformer_eval():
    """
    Transformer evaluation.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, reserve_class_name_in_scope=False,
                   device_id=get_device_id())

    dataset = load_test_data(batch_size=config.batch_size, data_file=config.data_file)
    tfm_model = TransformerModel(config=config, is_training=False, use_one_hot_embeddings=False)

    parameter_dict = load_weights(config.model_file)
    ms.load_param_into_net(tfm_model, parameter_dict)

    tfm_infer = TransformerInferCell(tfm_model)
    model = Model(tfm_infer)

    predictions = []
    source_sents = []
    target_sents = []
    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        source_sents.append(batch["source_eos_ids"])
        target_sents.append(batch["target_eos_ids"])
        source_ids = Tensor(batch["source_eos_ids"], ms.int32)
        source_mask = Tensor(batch["source_eos_mask"], ms.int32)
        predicted_ids = model.predict(source_ids, source_mask)
        predictions.append(predicted_ids.asnumpy())

    # decode and write to file
    f = open(config.output_file, 'w')
    for batch_out in predictions:
        for i in range(config.batch_size):
            if batch_out.ndim == 3:
                batch_out = batch_out[:, 0]
            token_ids = [str(x) for x in batch_out[i].tolist()]
            f.write(" ".join(token_ids) + "\n")
    f.close()


if __name__ == "__main__":
    run_transformer_eval()
