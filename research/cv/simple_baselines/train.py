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
"""Train simple baselines."""
from __future__ import division

import os
import ast
import argparse
import numpy as np

from mindspore import context, Tensor
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train import Model
from mindspore.train.callback import TimeMonitor, LossMonitor, ModelCheckpoint, CheckpointConfig
from mindspore.nn.optim import Adam
from mindspore.common import set_seed

from src.config import config
from src.pose_resnet import GetPoseResNet
from src.network_with_loss import JointsMSELoss, PoseResNetWithLoss
from src.dataset import keypoint_dataset

if config.MODELARTS.IS_MODEL_ARTS:
    import moxing as mox

set_seed(config.GENERAL.TRAIN_SEED)


def get_lr(begin_epoch, total_epochs, steps_per_epoch, lr_init=0.1, factor=0.1,
           epoch_number_to_drop=(90, 120)):
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    step_number_to_drop = [steps_per_epoch * x for x in epoch_number_to_drop]
    for i in range(int(total_steps)):
        if i in step_number_to_drop:
            lr_init = lr_init * factor
        lr_each_step.append(lr_init)
    current_step = steps_per_epoch * begin_epoch
    lr_each_step = np.array(lr_each_step, dtype=np.float32)
    learning_rate = lr_each_step[current_step:]
    return learning_rate


def parse_args():
    """command line arguments parsing"""
    parser = argparse.ArgumentParser(description="Simple Baselines training")
    parser.add_argument('--data_url', required=False, default=None, help='Location of data.')
    parser.add_argument('--train_url', required=False, default=None, help='Location of training outputs.')
    parser.add_argument('--device_id', required=False, default=None, type=int, help='Location of training outputs.')
    parser.add_argument("--device_target", type=str, choices=["Ascend", "GPU", "CPU"], default="Ascend",
                        help="device target")
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Location of training outputs.')
    parser.add_argument('--is_model_arts', type=ast.literal_eval, default=False, help='Location of training outputs.')
    args = parser.parse_args()
    return args


def main():
    print("loading parse...")
    args = parse_args()
    device_target = args.device_target
    config.GENERAL.RUN_DISTRIBUTE = args.run_distribute
    config.MODELARTS.IS_MODEL_ARTS = args.is_model_arts

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, save_graphs=False)
    if device_target == "Ascend":
        rank = 0
        device_num = 1
        context.set_context(device_id=rank)
        if config.GENERAL.RUN_DISTRIBUTE:
            init()
            rank = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    elif device_target == "GPU":
        rank = int(os.getenv('DEVICE_ID', "0"))
        device_num = int(os.getenv('RANK_SIZE', "0"))
        if device_num > 1:
            init()
            rank = get_rank()
            device_num = get_group_size()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        raise ValueError("Unsupported device, only GPU or Ascend is supported.")

    if config.MODELARTS.IS_MODEL_ARTS:
        mox.file.copy_parallel(src_url=args.data_url, dst_url=config.MODELARTS.CACHE_INPUT)

    dataset, _ = keypoint_dataset(config,
                                  rank=rank,
                                  group_size=device_num,
                                  train_mode=True)
    net = GetPoseResNet(config)
    loss = JointsMSELoss(config.LOSS.USE_TARGET_WEIGHT)
    net_with_loss = PoseResNetWithLoss(net, loss)
    dataset_size = dataset.get_dataset_size()
    lr = Tensor(get_lr(config.TRAIN.BEGIN_EPOCH,
                       config.TRAIN.END_EPOCH,
                       dataset_size,
                       lr_init=config.TRAIN.LR,
                       factor=config.TRAIN.LR_FACTOR,
                       epoch_number_to_drop=config.TRAIN.LR_STEP))
    opt = Adam(net.trainable_params(), learning_rate=lr)
    time_cb = TimeMonitor(data_size=dataset_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]

    if config.TRAIN.SAVE_CKPT:
        config_ck = CheckpointConfig(save_checkpoint_steps=dataset_size*config.TRAIN.SAVE_CKPT_EPOCH,
                                     keep_checkpoint_max=config.TRAIN.KEEP_CKPT_MAX)
        if config.GENERAL.RUN_DISTRIBUTE:
            prefix = 'multi_' + 'train_poseresnet_' + config.GENERAL.VERSION + '_' + str(rank)
        else:
            prefix = 'single_' + 'train_poseresnet_' + config.GENERAL.VERSION

        if config.MODELARTS.IS_MODEL_ARTS:
            directory = os.path.join(config.MODELARTS.CACHE_OUTPUT, 'device_' + str(rank))
        elif config.GENERAL.RUN_DISTRIBUTE:
            directory = os.path.join(config.TRAIN.CKPT_PATH, 'device_' + str(rank))
        else:
            directory = os.path.join(config.TRAIN.CKPT_PATH, 'device')

        ckpoint_cb = ModelCheckpoint(prefix=prefix, directory=directory, config=config_ck)
        cb.append(ckpoint_cb)

    model = Model(net_with_loss, loss_fn=None, optimizer=opt, amp_level="O2")
    epoch_size = config.TRAIN.END_EPOCH - config.TRAIN.BEGIN_EPOCH
    print("************ Start training now ************")
    print('start training, epoch size = %d' % epoch_size)
    model.train(epoch_size, dataset, callbacks=cb, dataset_sink_mode=True)

    if config.MODELARTS.IS_MODEL_ARTS:
        mox.file.copy_parallel(src_url=config.MODELARTS.CACHE_OUTPUT, dst_url=args.train_url)


if __name__ == '__main__':
    main()
