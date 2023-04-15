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
"""
#################train textcnn example on movie review########################
python train.py
"""
import os
import math

import mindspore as ms
import mindspore.nn as nn
from mindspore.nn.metrics import Accuracy
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.model import Model
from mindspore.common import set_seed

from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_rank_id
from model_utils.config import config
from src.textcnn import TextCNN
from src.textcnn import SoftmaxCrossEntropyExpand, EvalCallback
from src.dataset import MovieReview, SST2, Subjectivity

set_seed(1)

def modelarts_pre_process():
    config.checkpoint_path = os.path.join(config.output_path, str(get_rank_id()), config.checkpoint_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def train_net():
    '''train net'''
    # set context
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target)
    ms.set_context(device_id=get_device_id())
    if config.dataset == 'MR':
        instance = MovieReview(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
    elif config.dataset == 'SUBJ':
        instance = Subjectivity(root_dir=config.data_path, maxlen=config.word_len, split=0.9)
        if config.device_target == "GPU":
            ms.set_context(enable_graph_kernel=True)
    elif config.dataset == 'SST2':
        instance = SST2(root_dir=config.data_path, maxlen=config.word_len, split=0.9)

    dataset = instance.create_train_dataset(batch_size=config.batch_size, epoch_size=config.epoch_size)
    eval_dataset = instance.create_test_dataset(batch_size=config.batch_size)
    batch_num = dataset.get_dataset_size()
    if config.sink_size == -1:
        config.sink_size = batch_num

    base_lr = float(config.base_lr)
    learning_rate = []
    warm_up = [base_lr / math.floor(config.epoch_size / 5) * (i + 1) for _ in range(batch_num) for i in
               range(math.floor(config.epoch_size / 5))]
    shrink = [base_lr / (16 * (i + 1)) for _ in range(batch_num) for i in range(math.floor(config.epoch_size * 3 / 5))]
    normal_run = [base_lr for _ in range(batch_num) for i in
                  range(config.epoch_size - math.floor(config.epoch_size / 5) - math.floor(config.epoch_size * 2 / 5))]
    learning_rate = learning_rate + warm_up + normal_run + shrink

    net = TextCNN(vocab_len=instance.get_dict_len(), word_len=config.word_len,
                  num_classes=config.num_classes, vec_length=config.vec_length)
    # Continue training if set pre_trained to be True
    if config.pre_trained:
        param_dict = ms.load_checkpoint(config.checkpoint_path)
        ms.load_param_into_net(net, param_dict)

    opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), \
                  learning_rate=learning_rate, weight_decay=float(config.weight_decay))
    loss = SoftmaxCrossEntropyExpand(sparse=True)

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc': Accuracy()})

    config_ck = CheckpointConfig(save_checkpoint_steps=config.sink_size,
                                 keep_checkpoint_max=config.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpt_save_dir = os.path.join(config.output_path, config.checkpoint_path)
    ckpoint_cb = ModelCheckpoint(prefix="train_textcnn", directory=ckpt_save_dir, config=config_ck)
    loss_cb = LossMonitor()
    eval_callback = EvalCallback(model, eval_dataset, save_path=ckpt_save_dir)
    if config.device_target == "CPU":
        model.train(config.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
    else:
        epoch_count = config.epoch_size * batch_num // config.sink_size
        model.train(epoch_count, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb, eval_callback],
                    sink_size=config.sink_size, dataset_sink_mode=True)
    print("train success")


if __name__ == '__main__':
    train_net()
