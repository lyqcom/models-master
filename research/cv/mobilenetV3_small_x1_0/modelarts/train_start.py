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
"""train_imagenet."""

import os
import ast
import argparse
import numpy as np
from mindspore import context
from mindspore import Tensor

from mindspore.nn import RMSProp
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.communication.management import init
from mindspore import export

from src.dataset import create_dataset
from src.lr_generator import get_lr
from src.config import config_ascend as config
from src.loss import CrossEntropyWithLabelSmooth
from src.monitor import Monitor
from src.mobilenetv3 import mobilenet_v3_small

set_seed(1)

parser = argparse.ArgumentParser(description='Image classification')
# modelarts parameter
parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
parser.add_argument('--train_url', type=str, default=None, help='Train output path')
# Ascend parameter
parser.add_argument('--dataset_path', type=str, default=None, help='Dataset path')

parser.add_argument('--run_modelarts', type=ast.literal_eval, default=True, help='Run mode')
parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
parser.add_argument('--num_classes', type=int, default=1000, help='The number of class')
parser.add_argument('--epoch_size', type=int, default=1, help='The epoch size')


args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False)

if __name__ == '__main__':
    # init distributed
    if args_opt.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_train_url = '/cache/ckpt'
        if device_num > 1:
            init()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode='data_parallel', gradients_mean=True)
            local_data_url = os.path.join(local_data_url, str(device_id))
        mox.file.copy_parallel(args_opt.data_url, local_data_url)
    else:
        if args_opt.run_distribute:
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_context(device_id=device_id)
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
        else:
            context.set_context(device_id=args_opt.device_id)
            device_num = 1
            device_id = 0
    # define net
    net = mobilenet_v3_small(num_classes=config.num_classes, multiplier=1.)
    # define loss
    if config.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(
            smooth_factor=config.label_smooth, num_classes=config.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # define dataset
    if args_opt.run_modelarts:
        dataset = create_dataset(dataset_path=local_data_url,
                                 do_train=True,
                                 batch_size=config.batch_size,
                                 device_num=device_num, rank=device_id)
    else:
        dataset = create_dataset(dataset_path=args_opt.dataset_path,
                                 do_train=True,
                                 batch_size=config.batch_size,
                                 device_num=device_num, rank=device_id)
    step_size = dataset.get_dataset_size()
    # resume
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
    # define optimizer
    loss_scale = FixedLossScaleManager(
        config.loss_scale, drop_overflow_update=False)
    lr = Tensor(get_lr(global_step=0,
                       lr_init=0,
                       lr_end=0,
                       lr_max=config.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=config.epoch_size,
                       steps_per_epoch=step_size))
    opt = RMSProp(net.trainable_params(), learning_rate=lr, decay=0.9, weight_decay=config.weight_decay,
                  momentum=config.momentum, epsilon=0.001, loss_scale=config.loss_scale)
    # define model
    model = Model(net, loss_fn=loss, optimizer=opt,
                  loss_scale_manager=loss_scale, amp_level='O3')

    cb = [Monitor(lr_init=lr.asnumpy())]

    if config.save_checkpoint and (device_num == 1 or device_id == 0):
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        if args_opt.run_modelarts:
            ckpt_cb = ModelCheckpoint(prefix="mobilenetv3_small", directory=local_train_url, config=config_ck)
        else:
            save_ckpt_path = os.path.join(config.save_checkpoint_path, 'model_' + str(device_id) + '/')
            ckpt_cb = ModelCheckpoint(prefix="mobilenetv3_small", directory=save_ckpt_path, config=config_ck)
        cb += [ckpt_cb]
    # begine train
    model.train(args_opt.epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)
    if args_opt.run_modelarts and config.save_checkpoint and (device_num == 1 or device_id == 0):
        mox.file.copy_parallel(local_train_url, args_opt.train_url)

    print("export to air begin")
    net = mobilenet_v3_small()
    print("output dir:" + local_train_url)
    for f in os.listdir(local_train_url):
        print(f)
        if f.endswith('.ckpt'):
            ckpt_path = os.path.join(local_train_url, f)
            param_dict = load_checkpoint(ckpt_path)
            load_param_into_net(net, param_dict)
            input_shp = [1, 3, 224, 224]
            input_array = Tensor(np.random.uniform(-1.0, 1.0, size=input_shp).astype(np.float32))
            export(net, input_array, file_name=local_train_url+'/mobilenetv3_small', file_format='AIR')
    if args_opt.run_modelarts and config.save_checkpoint and (device_num == 1 or device_id == 0):
        mox.file.copy_parallel(local_train_url, args_opt.train_url)
    for f in os.listdir('/cache'):
        print(f)
