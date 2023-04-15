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

"""Train SSD and get checkpoint files."""

import os
import argparse
import ast
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore.communication.management import init, get_rank
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed, dtype
from src.ssd import SSD320, SSDWithLossCell, TrainingWrapper, ssd_mobilenet_v2
from src.config import config
from src.dataset import create_ssd_dataset, create_mindrecord
from src.lr_schedule import get_lr
from src.init_params import init_net_param, filter_checkpoint_parameter

set_seed(1)

def get_args():
    """get arguments"""
    parser = argparse.ArgumentParser(description="SSD training")
    parser.add_argument("--run_platform", type=str, default="Ascend", choices=("Ascend", "GPU"),
                        help="run platform.")
    parser.add_argument("--only_create_dataset", type=ast.literal_eval, default=False,
                        help="If set it true, only create Mindrecord, default is False.")
    parser.add_argument("--distribute", type=ast.literal_eval, default=False,
                        help="Run distribute, default is False.")
    parser.add_argument("--use_float16", type=ast.literal_eval, default=True,
                        help="use float16 or not, default is False", choices=(True, False))
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--device_num", type=int, default=1, help="Use device nums, default is 1.")
    parser.add_argument("--num_workers", type=int, default=64, help="num_workers, default is 64.")
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate, default is 0.05.")
    parser.add_argument("--mode", type=str, default="sink", help="Run sink mode or not, default is sink.")
    parser.add_argument("--dataset", type=str, default="coco", help="Dataset, default is coco.")
    parser.add_argument('--data_url', type=str, default=None, help='Dataset path')
    parser.add_argument('--train_url', type=str, default=None, help='Train output path')
    parser.add_argument('--modelarts_mode', type=ast.literal_eval, default=False,
                        help='train on modelarts or not, default is False')
    parser.add_argument('--mindrecord_mode', type=str, default="mindrecord", choices=("coco", "mindrecord"),
                        help='type of data, default is mindrecord')
    parser.add_argument("--epoch_size", type=int, default=500, help="Epoch size, default is 500.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--pre_trained", type=str, default=None, help="Pretrained Checkpoint file path.")
    parser.add_argument("--pre_trained_epoch_size", type=int, default=0, help="Pretrained epoch size.")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=10, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    parser.add_argument("--filter_weight", type=ast.literal_eval, default=False,
                        help="Filter head weight parameters, default is False.")
    parser.add_argument('--freeze_layer', type=str, default="none", choices=["none", "backbone"],
                        help="freeze the weights of network, support freeze the backbone's weights, "
                             "default is not freezing.")
    args_opt = parser.parse_args()
    return args_opt

def main():
    args_opt = get_args()
    if args_opt.modelarts_mode:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform, device_id=device_id)
        config.coco_root = os.path.join(config.coco_root, str(device_id))
        config.mindrecord_dir = os.path.join(config.mindrecord_dir, str(device_id))
        if args_opt.distribute:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            init()
            context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 89])
            rank = get_rank()
        else:
            rank = 0
        if args_opt.mindrecord_mode == "mindrecord":
            mox.file.copy_parallel(args_opt.data_url, config.mindrecord_dir)
        else:
            mox.file.copy_parallel(args_opt.data_url, config.coco_root)

    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.run_platform)
        if args_opt.distribute:
            if os.getenv("DEVICE_ID", "not_set").isdigit():
                context.set_context(device_id=int(os.getenv("DEVICE_ID")))
            device_num = args_opt.device_num
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                              device_num=device_num)
            init()
            context.set_auto_parallel_context(all_reduce_fusion_config=[29, 58, 89])
            rank = get_rank()
        else:
            rank = 0
            device_num = 1
            context.set_context(device_id=args_opt.device_id)
    mindrecord_file = create_mindrecord(args_opt.dataset, "ssd.mindrecord", True)
    if args_opt.only_create_dataset:
        if args_opt.modelarts_mode:
            mox.file.copy_parallel(config.mindrecord_dir, args_opt.train_url)
        return

    loss_scale = float(args_opt.loss_scale)

    # When create MindDataset, using the fitst mindrecord file, such as ssd.mindrecord0.
    dataset = create_ssd_dataset(mindrecord_file, repeat_num=1, batch_size=args_opt.batch_size,
                                 device_num=device_num, rank=rank, num_parallel_workers=args_opt.num_workers)

    dataset_size = dataset.get_dataset_size()
    print("Create dataset done!")

    backbone = ssd_mobilenet_v2()

    ssd = SSD320(backbone=backbone, config=config)
    if (hasattr(args_opt, 'use_float16') and args_opt.use_float16):
        ssd.to_float(dtype.float16)
    net = SSDWithLossCell(ssd, config)
    init_net_param(net)

    # checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * args_opt.save_checkpoint_epochs)
    save_ckpt_path = './ckpt_' + str(rank) + '/'
    ckpoint_cb = ModelCheckpoint(prefix="ssd", directory=save_ckpt_path, config=ckpt_config)

    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        if args_opt.filter_weight:
            filter_checkpoint_parameter(param_dict)
        load_param_into_net(net, param_dict)

    if args_opt.freeze_layer == "backbone":
        for param in backbone.feature_1.trainable_params():
            param.requires_grad = False

    lr = Tensor(get_lr(global_step=args_opt.pre_trained_epoch_size * dataset_size,
                       lr_init=config.lr_init, lr_end=config.lr_end_rate * args_opt.lr, lr_max=args_opt.lr,
                       warmup_epochs=config.warmup_epochs,
                       total_epochs=args_opt.epoch_size,
                       steps_per_epoch=dataset_size))

    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                      config.momentum, config.weight_decay, loss_scale)
    net = TrainingWrapper(net, opt, loss_scale)

    callback = [TimeMonitor(data_size=dataset_size), LossMonitor(), ckpoint_cb]
    model = Model(net)
    dataset_sink_mode = False
    if args_opt.mode == "sink":
        print("In sink mode, one epoch return a loss.")
        dataset_sink_mode = True
    print("Start train SSD, the first epoch will be slower because of the graph compilation.")
    model.train(args_opt.epoch_size, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)
    if args_opt.modelarts_mode:
        mox.file.copy_parallel(save_ckpt_path, args_opt.train_url)
if __name__ == '__main__':
    main()
