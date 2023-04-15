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
"""train ImageNet."""
import argparse
import ast
import datetime
import os
import time

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim import Momentum
from mindspore.train.callback import Callback
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model

from src.config import config
from src.crossentropy import CrossEntropy
from src.dataset import classification_dataset
from src.eval_callback import EvalCallBack
from src.image_classification import get_network
from src.lr_generator import get_lr
from src.utils.logging import get_logger
from src.utils.optimizers__init__ import get_param_groups
from src.utils.var_init import load_pretrain_model

set_seed(1)


class BuildTrainNetwork(nn.Cell):
    """build training network"""
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        loss = self.criterion(output, label)
        return loss


class ProgressMonitor(Callback):
    """monitor loss and time"""
    def __init__(self, args):
        super(ProgressMonitor, self).__init__()
        self.me_epoch_start_time = 0
        self.me_epoch_start_step_num = 0
        self.args = args
        self.ckpt_history = []

    def begin(self, run_context):
        self.args.logger.info('start network train...')

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context, *me_args):
        """describe network construct"""
        cb_params = run_context.original_args()
        me_step = cb_params.cur_step_num - 1

        real_epoch = me_step // self.args.steps_per_epoch
        time_used = time.time() - self.me_epoch_start_time
        fps_mean = (self.args.per_batch_size * (me_step-self.me_epoch_start_step_num))
        fps_mean = fps_mean * self.args.group_size
        fps_mean = fps_mean / time_used
        self.args.logger.info('epoch[{}], iter[{}], loss:{}, '
                              'mean_fps:{:.2f}'
                              'imgs/sec'.format(real_epoch,
                                                me_step,
                                                cb_params.net_outputs,
                                                fps_mean))

        if self.args.rank_save_ckpt_flag:
            import glob
            ckpts = glob.glob(os.path.join(self.args.outputs_dir, '*.ckpt'))
            for ckpt in ckpts:
                ckpt_fn = os.path.basename(ckpt)
                if not ckpt_fn.startswith('{}-'.format(self.args.rank)):
                    continue
                if ckpt in self.ckpt_history:
                    continue
                self.ckpt_history.append(ckpt)
                self.args.logger.info('epoch[{}], iter[{}], loss:{}, '
                                      'ckpt:{},'
                                      'ckpt_fn:{}'.format(real_epoch,
                                                          me_step,
                                                          cb_params.net_outputs,
                                                          ckpt,
                                                          ckpt_fn))

        self.me_epoch_start_step_num = me_step
        self.me_epoch_start_time = time.time()

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context, *me_args):
        pass

    def end(self, run_context):
        self.args.logger.info('end network train...')


def add_arguments(parser):
    """Add arguments to the parser"""
    parser.add_argument('--platform', type=str, default='Ascend',
                        choices=('Ascend', 'GPU'), help='run platform')

    # dataset related
    parser.add_argument('--data_dir', type=str, default='', help='train data dir')
    parser.add_argument('--per_batch_size', default=128, type=int, help='batch size for per gpu')
    # network related
    parser.add_argument('--pretrained',
                        default='',
                        type=str,
                        help='model_path, local pretrained model to load')

    # distributed related
    parser.add_argument('--is_distributed', type=int, default=1, help='if multi device')
    # Data sink
    parser.add_argument('--data_sink_mode',
                        type=int,
                        default=None,
                        help='(0 or 1) Turn data sink off or on.')
    # Learning rate
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    # roma obs
    parser.add_argument('--train_url', type=str, default="", help='train url')
    # new argument
    parser.add_argument("--eval_interval", type=int, default=1,
                        help="Evaluation interval when run_eval is True, default is 1.")
    parser.add_argument("--eval_start_epoch", type=int, default=120,
                        help="Evaluation start epoch when run_eval is True, default is 120.")
    parser.add_argument("--save_best_ckpt", type=ast.literal_eval, default=True,
                        help="Save best checkpoint when run_eval is True, default is True.")
    # dataset of eval dataset
    parser.add_argument('--eval_data_dir',
                        type=str,
                        default='',
                        help='eval data dir')
    parser.add_argument('--eval_per_batch_size',
                        default=32,
                        type=int,
                        help='batch size for per npu')
    parser.add_argument("--run_eval",
                        type=ast.literal_eval,
                        default=True,
                        help="Run evaluation when training, default is True.")
    # best ckpt
    parser.add_argument('--eval_log_path',
                        type=str,
                        default='eval_outputs/',
                        help='path to save log')
    parser.add_argument('--eval_is_distributed',
                        type=int,
                        default=0,
                        help='if multi device')
    parser.add_argument('--use_python_multiprocessing',
                        type=int,
                        default=None,
                        help='(0 or 1) Parallelize Python operations '
                             'with multiple worker processes.')


def parse_args(cloud_args=None):
    """parameters"""
    parser = argparse.ArgumentParser('mindspore classification training')
    add_arguments(parser)

    args, _ = parser.parse_known_args()
    args = merge_args(args, cloud_args)
    args.image_size = config['image_size']
    args.num_classes = config['num_classes']
    if args.lr is None:
        args.lr = config['lr']
    args.lr_scheduler = config['lr_scheduler']
    args.lr_epochs = config['lr_epochs']
    args.lr_gamma = config['lr_gamma']
    args.eta_min = config['eta_min']
    args.T_max = config['T_max']
    args.max_epoch = config['max_epoch']
    args.warmup_epochs = config['warmup_epochs']
    args.weight_decay = config['weight_decay']
    args.momentum = config['momentum']
    args.is_dynamic_loss_scale = config['is_dynamic_loss_scale']
    args.loss_scale = config['loss_scale']
    args.label_smooth = config['label_smooth']
    args.label_smooth_factor = config['label_smooth_factor']
    args.ckpt_interval = config['ckpt_interval']
    args.ckpt_save_max = config['ckpt_save_max']
    args.ckpt_path = config['ckpt_path']
    args.is_save_on_master = config['is_save_on_master']
    args.rank = config['rank']
    args.group_size = config['group_size']
    if args.data_sink_mode is None:
        args.data_sink_mode = config['data_sink_mode']
    else:
        args.data_sink_mode = bool(args.data_sink_mode)
    args.lr_epochs = list(map(int, args.lr_epochs.split(',')))
    args.image_size = list(map(int, args.image_size.split(',')))
    if args.use_python_multiprocessing is None:
        args.use_python_multiprocessing = config['use_python_multiprocessing']
    else:
        args.use_python_multiprocessing = bool(args.use_python_multiprocessing)

    # context.set_context(mode=context.PYNATIVE_MODE,
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=args.platform, save_graphs=False)
    # init distributed
    if args.is_distributed:
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
    else:
        args.rank = 0
        args.group_size = 1

    if args.is_dynamic_loss_scale == 1:
        args.loss_scale = 1  # for dynamic loss scale can not set loss scale in momentum opt

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    # logger
    args.outputs_dir = os.path.join(args.ckpt_path,
                                    datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    args.logger = get_logger(args.outputs_dir, args.rank)
    return args


def merge_args(args, cloud_args):
    """dictionary"""
    args_dict = vars(args)
    if isinstance(cloud_args, dict):
        for key in cloud_args.keys():
            val = cloud_args[key]
            if key in args_dict and val:
                arg_type = type(args_dict[key])
                if arg_type is not type(None):
                    val = arg_type(val)
                args_dict[key] = val
    return args


def apply_eval(eval_param):
    eval_model = eval_param["model"]
    eval_ds = eval_param["dataset"]
    metrics_name = eval_param["metrics_name"]
    res = eval_model.eval(eval_ds)
    return res[metrics_name]


def train(cloud_args=None):
    """training process"""
    args = parse_args(cloud_args)
    if args.platform == 'Ascend':
        if os.getenv('DEVICE_ID', "not_set").isdigit():
            context.set_context(device_id=int(os.getenv('DEVICE_ID')))

    # init distributed
    if args.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        context.set_auto_parallel_context(parallel_mode=parallel_mode, device_num=args.group_size,
                                          gradients_mean=True)
    # dataloader
    de_dataset = classification_dataset(args.data_dir, args.image_size,
                                        args.per_batch_size, 1,
                                        args.rank, args.group_size, num_parallel_workers=8)
    de_dataset.map_model = 4  # !!!important
    args.steps_per_epoch = de_dataset.get_dataset_size()

    # eval_dataset
    args.logger.save_args(args)
    # network
    args.logger.important_info('start create network')
    # get network and init
    if args.platform == 'Ascend':
        network = get_network(num_classes=args.num_classes, platform=args.platform)
    else:
        network = get_network(num_classes=args.num_classes, platform=args.platform, fp16=False)

    load_pretrain_model(args.pretrained, network, args)
    # lr scheduler
    lr = get_lr(args)
    # optimizer
    opt = Momentum(params=get_param_groups(network),
                   learning_rate=Tensor(lr),
                   momentum=args.momentum,
                   weight_decay=args.weight_decay,
                   loss_scale=args.loss_scale)
    # loss
    if not args.label_smooth:
        args.label_smooth_factor = 0.0
    loss = CrossEntropy(smooth_factor=args.label_smooth_factor, num_classes=args.num_classes)

    if args.platform == 'Ascend':
        if args.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536,
                                                         scale_factor=2,
                                                         scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)

    else:
        loss_scale_manager = None

    model = Model(network, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale_manager,
                  metrics={'acc'}, amp_level='O3')

    # checkpoint save
    progress_cb = ProgressMonitor(args)
    callbacks = [progress_cb, TimeMonitor()]

    if args.rank == 0 and not args.data_sink_mode:
        callbacks.append(LossMonitor(100))

    # code like eval.py
    # if run eval
    if args.run_eval:
        if args.eval_data_dir is None or (not os.path.isdir(args.eval_data_dir)):
            raise ValueError("{} is not a existing path.".format(args.eval_data_dir))
        eval_de_dataset = classification_dataset(args.eval_data_dir,
                                                 image_size=args.image_size,
                                                 per_batch_size=args.eval_per_batch_size,
                                                 max_epoch=1,
                                                 rank=args.rank,
                                                 group_size=args.group_size,
                                                 mode='eval')
        eval_param_dict = {"model": model, "dataset": eval_de_dataset, "metrics_name": "acc"}
        eval_callback = EvalCallBack(apply_eval,
                                     eval_param_dict,
                                     interval=args.eval_interval,
                                     eval_start_epoch=args.eval_start_epoch,
                                     save_best_ckpt=args.save_best_ckpt,
                                     ckpt_directory=args.ckpt_path,
                                     best_ckpt_name="best_acc.ckpt",
                                     metrics_name="acc"
                                     )
        callbacks.append(eval_callback)

    if args.rank_save_ckpt_flag:
        ckpt_config = CheckpointConfig(save_checkpoint_steps=args.ckpt_interval * args.steps_per_epoch,
                                       keep_checkpoint_max=args.ckpt_save_max)
        save_ckpt_path = os.path.join(args.outputs_dir, 'ckpt_' + str(args.rank) + '/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config,
                                  directory=save_ckpt_path,
                                  prefix='{}'.format(args.rank))
        callbacks.append(ckpt_cb)

    model.train(args.max_epoch, de_dataset, callbacks=callbacks, dataset_sink_mode=args.data_sink_mode)


if __name__ == "__main__":
    train()
