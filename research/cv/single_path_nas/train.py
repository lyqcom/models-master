# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# less required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
#################train spnasnet example########################
python train.py
"""
import argparse
import os

from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import LossMonitor
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import TimeMonitor
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.model import Model

from src import spnasnet
from src.CrossEntropySmooth import CrossEntropySmooth
from src.config import imagenet_cfg
from src.dataset import create_dataset_imagenet

set_seed(1)


def lr_steps_imagenet(_cfg, steps_per_epoch):
    """lr step for imagenet"""
    from src.lr_scheduler.warmup_step_lr import warmup_step_lr
    from src.lr_scheduler.warmup_cosine_annealing_lr import warmup_cosine_annealing_lr
    if _cfg.lr_scheduler == 'exponential':
        _lr = warmup_step_lr(_cfg.lr_init,
                             _cfg.lr_epochs,
                             steps_per_epoch,
                             _cfg.warmup_epochs,
                             _cfg.epoch_size,
                             gamma=_cfg.lr_gamma,
                             )
    elif _cfg.lr_scheduler == 'cosine_annealing':
        _lr = warmup_cosine_annealing_lr(_cfg.lr_init,
                                         steps_per_epoch,
                                         _cfg.warmup_epochs,
                                         _cfg.epoch_size,
                                         _cfg.T_max,
                                         _cfg.eta_min)
    else:
        raise NotImplementedError(_cfg.lr_scheduler)

    return _lr


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single-Path-NAS Training')
    parser.add_argument('--dataset_name', type=str, default='imagenet', choices=['imagenet'],
                        help='dataset name.')
    parser.add_argument('--filter_prefix', type=str, default='huawei',
                        help='filter_prefix name.')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend. (Default: None)')
    parser.add_argument('--device_target', type=str, choices=['Ascend', 'GPU'], required=True,
                        default="Ascend", help='Target device: Ascend or GPU')
    parser.add_argument('--data_path', type=str, default=None, required=True,
                        help='Path to the training dataset (e.g. "/datasets/imagenet/train/")')

    args_opt = parser.parse_args()

    if args_opt.dataset_name == "imagenet":
        cfg = imagenet_cfg
    else:
        raise ValueError("Unsupported dataset.")

    device_target = args_opt.device_target

    # We enabling the graph kernel only for the Ascend device.
    # enable_graph_kernel = (device_target == 'Ascend')
    enable_graph_kernel = False

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target,
                        enable_graph_kernel=enable_graph_kernel)

    device_num = int(os.environ.get("DEVICE_NUM", "1"))

    rank = 0
    if device_target == "Ascend":
        context.set_context(device_id=args_opt.device_id)

        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            rank = get_rank()
    elif device_target == "GPU":
        # Using the rank and devices number determined by the communication module.
        if device_num > 1:
            init('nccl')
            device_num = get_group_size()
            rank = get_rank()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
    else:
        raise ValueError("Unsupported platform.")

    dataset_drop_reminder = (device_target == 'GPU')

    if args_opt.dataset_name == "imagenet":
        if device_num > 1:
            dataset = create_dataset_imagenet(args_opt.data_path, 1, num_parallel_workers=8,
                                              device_num=device_num, rank_id=rank,
                                              drop_reminder=True)
        else:
            dataset = create_dataset_imagenet(args_opt.data_path, 1, num_parallel_workers=8,
                                              drop_reminder=True)
    else:
        raise ValueError("Unsupported dataset.")

    batch_num = dataset.get_dataset_size()

    net = spnasnet.get_spnasnet(num_classes=cfg.num_classes)
    net.update_parameters_name(args_opt.filter_prefix)

    loss_scale_manager = None
    if args_opt.dataset_name == 'imagenet':
        lr = lr_steps_imagenet(cfg, batch_num)

        def get_param_groups(network):
            """ get param groups """
            decay_params = []
            no_decay_params = []
            for x in network.trainable_params():
                parameter_name = x.name
                if parameter_name.endswith('.bias'):
                    # all bias not using weight decay
                    no_decay_params.append(x)
                elif parameter_name.endswith('.gamma'):
                    # bn weight bias not using weight decay, be carefully for now x not include BN
                    no_decay_params.append(x)
                elif parameter_name.endswith('.beta'):
                    # bn weight bias not using weight decay, be carefully for now x not include BN
                    no_decay_params.append(x)
                else:
                    decay_params.append(x)

            return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]


        if cfg.is_dynamic_loss_scale:
            cfg.loss_scale = 1

        opt = Momentum(params=net.get_parameters(),
                       learning_rate=Tensor(lr),
                       momentum=cfg.momentum,
                       weight_decay=cfg.weight_decay,
                       loss_scale=cfg.loss_scale)

        if not cfg.use_label_smooth:
            cfg.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)

        if cfg.is_dynamic_loss_scale == 1:
            loss_scale_manager = DynamicLossScaleManager(init_loss_scale=65536, scale_factor=2, scale_window=2000)
        else:
            loss_scale_manager = FixedLossScaleManager(cfg.loss_scale, drop_overflow_update=False)

    else:
        raise ValueError("Unsupported dataset.")

    model = Model(net, loss_fn=loss, optimizer=opt, metrics={'top_1_accuracy', 'top_5_accuracy', 'loss'},
                  amp_level="O3", keep_batchnorm_fp32=True, loss_scale_manager=loss_scale_manager)

    config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 1, keep_checkpoint_max=cfg.keep_checkpoint_max)
    time_cb = TimeMonitor(data_size=batch_num)
    ckpt_save_dir = "./ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="train_spnasnet_" + args_opt.dataset_name, directory=ckpt_save_dir,
                                 config=config_ck)
    loss_cb = LossMonitor()

    model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb], dataset_sink_mode=True)
    print("train success")
