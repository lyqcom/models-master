# Copyright 2022 Huawei Technologies Co., Ltd
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
"""train ghostnet."""

import os
import argparse
import ast

from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.serialization import load_checkpoint
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore.nn.optim.momentum import Momentum
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode

from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.dataset import create_dataset
from src.config import config
from src.ghostnet import ghostnet_1x, ghostnet_nose_1x
from src.ghostnet600 import ghostnet_600m
from src.dense import init_weight

parser = argparse.ArgumentParser(description='GhostNet')
parser.add_argument('--num_classes', type=int, default=5, help='Classes number')
parser.add_argument('--device_target', type=str, default='CPU', help='Device platform')
parser.add_argument('--save_checkpoint_path1', type=str, default='./ckpt/', help='Save path of ckpt file')
parser.add_argument('--data_url', type=str, default='./dataset/', help='Dataset path')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
parser.add_argument('--pre_trained', type=str, default='./pre_ckpt/ghostnet_1x_pets.ckpt',
                    help='Pretrained checkpoint path')
parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
parser.add_argument('--is_save_on_master', type=int, default=1, help='save ckpt on master or all rank')
args_opt = parser.parse_args()

set_seed(1)

if __name__ == '__main__':
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target, save_graphs=False)

    if args_opt.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        rank_size = int(os.environ.get("RANK_SIZE", 1))
        print(rank_size)
        device_num = rank_size
        context.set_context(device_id=device_id)
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        init()
        args_opt.rank = get_rank()

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args_opt.rank_save_ckpt_flag = 0
    if args_opt.is_save_on_master:
        if args_opt.rank == 0:
            args_opt.rank_save_ckpt_flag = 1
    else:
        args_opt.rank_save_ckpt_flag = 1

    # define net
    if 'ghostnet_1x_pets.ckpt' in args_opt.pre_trained:
        net = ghostnet_1x(num_classes=args_opt.num_classes)
    elif 'ghostnet_nose_1x_pets.ckpt' in args_opt.pre_trained:
        net = ghostnet_nose_1x(num_classes=args_opt.num_classes)
    elif 'ghostnet600M_pets.ckpt' in args_opt.pre_trained:
        net = ghostnet_600m(num_classes=args_opt.num_classes)
    net.to_float(mstype.float16)
    for _, cell in net.cells_and_names():
        if isinstance(cell, nn.Dense):
            cell.to_float(mstype.float32)

    local_data_path = args_opt.data_url
    print('Download data:')
    dataset = create_dataset(dataset_path=local_data_path, do_train=True, batch_size=config.batch_size,
                             num_parallel_workers=None)
    step_size = dataset.get_dataset_size()
    print('steps:', step_size)

    # init weight
    ckpt_param_dict = load_checkpoint(args_opt.pre_trained)
    init_weight(net=net, param_dict=ckpt_param_dict)

    # init lr
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end,
                lr_max=config.lr_max, warmup_epochs=config.warmup_epochs,
                total_epochs=config.epoch_size, steps_per_epoch=step_size)
    lr = Tensor(lr)

    if not config.use_label_smooth:
        config.label_smooth_factor = 0.0

    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=config.label_smooth_factor, num_classes=args_opt.num_classes)

    opt = Momentum(net.trainable_params(), lr, config.momentum, loss_scale=config.loss_scale,
                   weight_decay=config.weight_decay)

    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)

    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale,
                  metrics={'top_1_accuracy', 'top_5_accuracy'},
                  amp_level="O3", keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        if args_opt.rank_save_ckpt_flag:
            config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=config.keep_checkpoint_max)
            if 'ghostnet_1x_pets.ckpt' in args_opt.pre_trained:
                ckpt_cb = ModelCheckpoint(prefix="ghostnet_1x", directory=args_opt.save_checkpoint_path1,
                                          config=config_ck)
            elif 'ghostnet_nose_1x_pets.ckpt' in args_opt.pre_trained:
                ckpt_cb = ModelCheckpoint(prefix="ghostnet_nose_1x", directory=args_opt.save_checkpoint_path1,
                                          config=config_ck)
            elif 'ghostnet600M_pets.ckpt' in args_opt.pre_trained:
                ckpt_cb = ModelCheckpoint(prefix="ghostnet600M", directory=args_opt.save_checkpoint_path1,
                                          config=config_ck)
            cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size, dataset, callbacks=cb, sink_size=dataset.get_dataset_size())
