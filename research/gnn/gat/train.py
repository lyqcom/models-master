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
"""Test train gat"""
import os
import time
import numpy as np

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.dataset import load_and_process
from src.gat import GAT
from src.utils import LossAccuracyWrapper, TrainGAT

import mindspore.context as context
from mindspore.train.serialization import save_checkpoint, load_checkpoint
from mindspore import Tensor



def modelarts_pre_process():
    config.data_dir = os.path.join(config.data_dir, config.dataset)

@moxing_wrapper(pre_process=modelarts_pre_process)
def gnn_train():
    """Train GAT model."""

    if not os.path.exists("ckpts"):
        os.mkdir("ckpts")
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=config.device_target,
                        save_graphs=False)
    # train parameters
    hid_units = config.hid_units
    n_heads = config.n_heads
    early_stopping = config.early_stopping
    lr = config.lr
    l2_coeff = config.l2_coeff
    num_epochs = config.num_epochs
    feature, biases, y_train, train_mask, y_val, eval_mask, y_test, test_mask = \
        load_and_process(config.data_dir, config.train_nodes_num, \
            config.eval_nodes_num, config.test_nodes_num)
    feature_size = feature.shape[2]
    num_nodes = feature.shape[1]
    num_class = y_train.shape[2]

    gat_net = GAT(feature_size,
                  num_class,
                  num_nodes,
                  hid_units,
                  n_heads,
                  attn_drop=config.attn_dropout,
                  ftr_drop=config.feature_dropout)
    gat_net.add_flags_recursive(fp16=True)
    if config.dataset == "citeseer" and config.device_target == "GPU":
        gat_net.add_flags_recursive(fp32=True)

    feature = Tensor(feature)
    biases = Tensor(biases)

    eval_net = LossAccuracyWrapper(gat_net,
                                   num_class,
                                   y_val,
                                   eval_mask,
                                   l2_coeff)
    train_net = TrainGAT(gat_net,
                         num_class,
                         y_train,
                         train_mask,
                         lr,
                         l2_coeff)
    train_net.set_train(True)
    val_acc_max = 0.0
    val_loss_min = np.inf
    for _epoch in range(num_epochs):
        epoch_start = time.time()
        train_result = train_net(feature, biases)
        train_loss = train_result[0].asnumpy()
        train_acc = train_result[1].asnumpy()
        eval_result = eval_net(feature, biases)
        eval_loss = eval_result[0].asnumpy()
        eval_acc = eval_result[1].asnumpy()
        epoch_time = time.time() - epoch_start
        print("Epoch:{}, train loss={:.5f}, train acc={:.5f} | val loss={:.5f}, val acc={:.5f}, time={:.5f},".format(
            _epoch, train_loss, train_acc, eval_loss, eval_acc, epoch_time))
        if eval_acc >= val_acc_max or eval_loss < val_loss_min:
            if eval_acc >= val_acc_max and eval_loss < val_loss_min:
                val_acc_model = eval_acc
                val_loss_model = eval_loss
                if os.path.exists("ckpts/gat.ckpt"):
                    os.remove("ckpts/gat.ckpt")
                save_checkpoint(train_net.network, "ckpts/gat.ckpt")
            val_acc_max = np.max((val_acc_max, eval_acc))
            val_loss_min = np.min((val_loss_min, eval_loss))
            curr_step = 0
        else:
            curr_step += 1
            if curr_step == early_stopping:
                print("Early Stop Triggered!, Min loss: {}, Max accuracy: {}".format(val_loss_min, val_acc_max))
                print("Early stop model validation loss: {}, accuracy{}".format(val_loss_model, val_acc_model))
                break
    gat_net_test = GAT(feature_size,
                       num_class,
                       num_nodes,
                       hid_units,
                       n_heads,
                       attn_drop=0.0,
                       ftr_drop=0.0)
    load_checkpoint("ckpts/gat.ckpt", net=gat_net_test)
    gat_net_test.add_flags_recursive(fp16=True)
    if config.dataset == "citeseer" and config.device_target == "GPU":
        gat_net_test.add_flags_recursive(fp32=True)
    test_net = LossAccuracyWrapper(gat_net_test,
                                   num_class,
                                   y_test,
                                   test_mask,
                                   l2_coeff)
    test_result = test_net(feature, biases)
    print("Test loss={}, test acc={}".format(test_result[0], test_result[1]))


if __name__ == "__main__":
    gnn_train()
