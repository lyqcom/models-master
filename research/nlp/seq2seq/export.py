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
"""export checkpoint file into air models"""

import argparse
import numpy as np
import mindspore.nn as nn

from mindspore import Tensor, context, Parameter
from mindspore.common import dtype as mstype
from mindspore.train.serialization import export

from config import Seq2seqConfig
from src.seq2seq_model.seq2seq import Seq2seqModel
from src.utils import zero_weight
from src.utils.load_weights import load_infer_weights

parser = argparse.ArgumentParser(description="seq2seq export")
parser.add_argument("--file_name", type=str, default="seq2seq", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument('--infer_config', type=str, required=True, help='seq2seq config file')
parser.add_argument("--existed_ckpt", type=str, required=True, help="existed checkpoint address.")
parser.add_argument('--vocab_file', type=str, required=True, help='vocabulary file')
parser.add_argument("--bpe_codes", type=str, required=True, help="bpe codes to use.")
parser.add_argument("--device_target", type=str, required=True, default="GPU")
args = parser.parse_args()

context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target=args.device_target,
    reserve_class_name_in_scope=False)

class Seq2seqInferCell(nn.Cell):
    """
    Encapsulation class of Seq2seqModel network infer.

    Args:
        network (nn.Cell): Seq2seqModel model.

    Returns:
        Tuple[Tensor, Tensor], predicted_ids and predicted_probs.
    """

    def __init__(self, network):
        super(Seq2seqInferCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self,
                  source_ids_,
                  source_mask_):
        """Defines the computation performed."""

        predicted_ids = self.network(source_ids_,
                                     source_mask_)

        return predicted_ids

def get_config(config_file):
    tfm_config = Seq2seqConfig.from_json_file(config_file)
    tfm_config.compute_type = mstype.float16
    tfm_config.dtype = mstype.float32
    return tfm_config


if __name__ == '__main__':
    config = get_config(args.infer_config)
    config.existed_ckpt = args.existed_ckpt
    vocab = args.vocab_file
    bpe_codes = args.bpe_codes

    tfm_model = Seq2seqModel(
        config=config,
        is_training=False,
        use_one_hot_embeddings=False)

    params = tfm_model.trainable_params()
    weights = load_infer_weights(config)

    for param in params:
        value = param.data
        weights_name = param.name
        if weights_name not in weights:
            raise ValueError(f"{weights_name} is not found in weights.")
        if isinstance(value, Tensor):
            if weights_name in weights:
                assert weights_name in weights
                if isinstance(weights[weights_name], Parameter):
                    if param.data.dtype == "Float32":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float32))
                    elif param.data.dtype == "Float16":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float16))

                elif isinstance(weights[weights_name], Tensor):
                    param.set_data(Tensor(weights[weights_name].asnumpy(), config.dtype))
                elif isinstance(weights[weights_name], np.ndarray):
                    param.set_data(Tensor(weights[weights_name], config.dtype))
                else:
                    param.set_data(weights[weights_name])
            else:
                print("weight not found in checkpoint: " + weights_name)
                param.set_data(zero_weight(value.asnumpy().shape))

    print(" | Load weights successfully.")
    tfm_infer = Seq2seqInferCell(tfm_model)
    tfm_infer.set_train(False)

    source_ids = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))
    source_mask = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))

    export(tfm_infer, source_ids, source_mask, file_name=args.file_name, file_format=args.file_format)
