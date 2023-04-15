# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""Eval."""
import json
import numpy as np

from mindspore import context, load_distributed_checkpoint
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication import management as MultiAscend
from mindspore.context import ParallelMode
from mindspore.parallel import set_algo_parameters

from src.cpm import CPMModel
from src.cpm_train import VirtualDatasetOneInputCell
from src.cpm_loss import Cross_entropy_eval
from src.create_ckpt_file_lists import create_ckpt_file_list
from train import load_dataset

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num, get_rank_id

context.set_context(mode=context.GRAPH_MODE,
                    save_graphs=False,
                    device_target="Ascend",
                    device_id=get_device_id())


class CPMForInfer(nn.Cell):
    """
    Encapsulation class of CPM network infer.

    Args:
        network (nn.Cell): CPM model.
        batch_size (int): Batch size of input dataset.
        seq_length (int): Length of input tensor sequence.
        vocab_size (int): Size of the dictionary of embeddings.
        cfg: The config of networks.

    Returns:
        Tensor, losses.
    """
    def __init__(self, network, batch_size, seq_length, vocab_size, cfg):
        super(CPMForInfer, self).__init__(auto_prefix=False)
        self.network = network
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.loss_net = Cross_entropy_eval(batch_size=self.batch_size,
                                           seq_length=self.seq_length,
                                           vocab_size=self.vocab_size,
                                           config=cfg)

    def construct(self, input_ids, position_ids, attention_mask, loss_mask):
        logits = self.network(input_ids, position_ids, attention_mask)
        loss = self.loss_net(logits, loss_mask)
        return loss


class CPM_LAYER(nn.Cell):
    """
    CPM model training with loss function.
    """

    def __init__(self, config_eval):
        super(CPM_LAYER, self).__init__()
        self.cpm_model = CPMModel(batch_size=config_eval.batch_size,
                                  seq_length=config_eval.seq_length,
                                  vocab_size=config_eval.vocab_size,
                                  hidden_size=config_eval.hidden_size,
                                  num_hidden_layers=config_eval.num_hidden_layers,
                                  num_attention_heads=config_eval.num_attention_heads,
                                  config=config_eval)

    def construct(self, input_ids, position_ids=None, attention_mask=None):
        output = self.cpm_model(input_ids, position_ids, attention_mask)
        return output


def do_eval(args, config_eval, ckpt_file_list=None):
    """
    Building infer pipeline
    """
    with open(args.dataset_path, "r") as f:
        # cand_ids, data
        cand_ids, _ = json.load(f)
    print("++++ cand_ids: ", cand_ids)

    if args.distribute:
        dataset = load_dataset(args.dataset, config_eval.batch_size,
                               rank_size=get_device_num(),
                               rank_id=get_rank_id(),
                               drop_remainder=False,
                               is_training=False,
                               shuffle=False)
    else:
        dataset = load_dataset(args.dataset,
                               config_eval.batch_size,
                               drop_remainder=False,
                               is_training=False,
                               shuffle=False)

    cpm_model = CPM_LAYER(config_eval)

    if args.distribute:
        cpm_model = VirtualDatasetOneInputCell(cpm_model)
    params = cpm_model.trainable_params()
    print("+++++++current network parameter+++++")
    for pas in params:
        print(pas.name)
    print("++++++++++++")
    if not args.has_train_strategy:
        # load the checkpoint without train strategy.
        weights = load_checkpoint(args.ckpt_path_doc)
        can_be_loaded = {}
        print("+++++++loading weights+++++")
        for name, _ in weights.items():
            print('oldname:           ' + name)
            if 'cpm_model.' not in name:
                can_be_loaded['cpm_model.' + name] = weights[name]

                print('newname: cpm_model.' + name)
            else:
                can_be_loaded[name] = weights[name]
        print("+++++++loaded weights+++++")
        load_param_into_net(cpm_model, parameter_dict=can_be_loaded)

    infer_net = CPMForInfer(network=cpm_model,
                            batch_size=config_eval.batch_size,
                            seq_length=config_eval.seq_length,
                            vocab_size=config_eval.vocab_size,
                            cfg=config_eval)

    model = Model(infer_net)

    if args.has_train_strategy and not args.distribute:
        # load sliced checkpoint with train strategy, but will run standalone inference without model parallel.
        load_distributed_checkpoint(infer_net, ckpt_file_list, None)

    if args.has_train_strategy and args.distribute:
        # load sliced checkpoint with train strategy, will run distribute inference with model parallel.
        fake_input_ids = Tensor(np.ones((config_eval.batch_size, config_eval.seq_length)), mstype.int64)
        fake_position_ids = Tensor(np.random.randint(0, 10, [config_eval.batch_size, config_eval.seq_length]),
                                   mstype.int64)
        fake_attention_mask = Tensor(
            np.random.randn(config_eval.batch_size, config_eval.seq_length, config_eval.seq_length), mstype.float16)
        fake_loss_mask = Tensor(np.random.randn(config_eval.batch_size, config_eval.seq_length), mstype.float16)
        predict_layout = model.infer_predict_layout(fake_input_ids,
                                                    fake_position_ids,
                                                    fake_attention_mask,
                                                    fake_loss_mask)
        print("Loaded sliced checkpoint, will run distribute inference with model parallel.", flush=True)
        load_distributed_checkpoint(infer_net, ckpt_file_list, predict_layout)

    all_losses = []
    truth_labels = []

    steps_per_epoch = dataset.get_dataset_size()
    print("++++++Dataset size", steps_per_epoch, flush=True)

    for batch in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        print("++++ start")
        ms_truth = batch['truth']

        input_ids = Tensor(batch['input_ids'], mstype.int64)
        position_ids = Tensor(batch['position_ids'], mstype.int64)
        attention_mask = Tensor(batch['attention_mask'], mstype.float16)
        loss_mask = Tensor(batch['loss_mask'], mstype.float16)

        pred_id_tensor = model.predict(input_ids, position_ids, attention_mask, loss_mask)
        # numpy do it.
        pred_id_np = pred_id_tensor.asnumpy()
        pred_id_np = pred_id_np[:, cand_ids]
        pred_id = pred_id_np.argmax(axis=-1)
        print("++++ pred_id_np: ", pred_id_np)
        print("++++ ms_truth: ", ms_truth)

        all_losses.append(pred_id)
        truth_labels.append(ms_truth)

    all_losses = np.stack(all_losses).reshape(-1)
    truth_labels = np.stack(truth_labels).reshape(-1)
    print("++++ all_losses= \n", all_losses)
    print("++++ truthlabel= \n", truth_labels)
    result = sum([int(p == l) for p, l in zip(all_losses, truth_labels)]) / len(truth_labels)
    print("RESULT: ", result)
    return result


def set_parallel_env():
    r"""
    Parallel environment.
    """
    context.reset_auto_parallel_context()
    MultiAscend.init()

    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      device_num=get_device_num(),
                                      gradients_mean=True,
                                      full_batch=True)
    set_algo_parameters(elementwise_op_strategy_follow=True)

def modelarts_pre_process():
    '''modelarts pre process function.'''

@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    '''eval cpm network'''
    finetune_test_distrubute = config.finetune_test_distrubute
    finetune_test_standalone = config.finetune_test_standalone
    if config.distribute:
        set_parallel_env()

    ckpt_file_list_test = None
    if config.has_train_strategy:
        # Get the checkpoint with train strategy.
        train_strategy_list = create_ckpt_file_list(config, train_strategy="train_strategy.ckpt")
        context.set_auto_parallel_context(
            strategy_ckpt_load_file=train_strategy_list[0]
        )
        ckpt_file_list_test = create_ckpt_file_list(config)
        print("++++ Get sliced checkpoint file, lists: ", ckpt_file_list_test, flush=True)

    result_accuracy = 0.0
    if config.distribute:
        print("Start validation on 2 devices with model parallel.")
        result_accuracy = do_eval(config, finetune_test_distrubute, ckpt_file_list_test)
    else:
        print("Start validation on 1 device without model parallel.")
        result_accuracy = do_eval(config, finetune_test_standalone, ckpt_file_list_test)

    print("++++ Accuracy=", result_accuracy)

if __name__ == '__main__':
    run_eval()
