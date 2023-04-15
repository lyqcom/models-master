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
"""train net."""
import argparse
import ast
import os

from mindspore import Model, Tensor, context, nn, set_seed
from mindspore.common import initializer as weight_init
from mindspore.communication.management import get_rank, init
from mindspore.context import ParallelMode
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.nn.optim.momentum import Momentum
from mindspore.parallel import set_algo_parameters
from mindspore.train.callback import (CheckpointConfig, LossMonitor,
                                      ModelCheckpoint, TimeMonitor)
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import config1 as config
from src.dataset import create_dataset_cifar10 as create_dataset
from src.lr_generator import get_lr
from src.sknet50 import sknet50 as sknet
from src.var_init import KaimingNormal

parser = argparse.ArgumentParser(description='Image classification')
parser.add_argument('--run_distribute', type=ast.literal_eval,
                    default=False, help='Run distribute')
parser.add_argument('--run_modelarts', type=ast.literal_eval,
                    default=False, help='Run Modelarts')
parser.add_argument('--device_id', type=int, default=0, help='Device id.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument('--data_url', type=str,
                    default="/path/to/cifar10", help='Dataset path')
parser.add_argument('--train_url', type=str,
                    default="/cache/ckpt", help='Train path')
parser.add_argument('--device_target', type=str, default='Ascend', choices=["Ascend", "GPU"],
                    help="Device target, support Ascend.")
parser.add_argument('--pre_trained', type=str, default=None,
                    help='Pretrained checkpoint path')
parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
args_opt = parser.parse_args()

set_seed(1)

if __name__ == '__main__':
    target = args_opt.device_target
    if args_opt.run_modelarts:
        import moxing as mox

        mox.file.copy_parallel(args_opt.data_url, "/cache/data")
        args_opt.data_url = "/cache/data"
        ckpt_save_dir = "/cache/ckpt"
    else:
        ckpt_save_dir = config.save_checkpoint_path
    os.environ["DEVICE_TARGET"] = target
    # init context
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=target, save_graphs=False)

    if args_opt.run_distribute:
        if args_opt.device_target == "Ascend":
            device_id = int(os.getenv('DEVICE_ID'))
            context.set_context(device_id=device_id)
        context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        context.set_auto_parallel_context(
            all_reduce_fusion_config=[85, 160])
        init()
        rank_id = get_rank()
    else:
        rank_id = 0
        device_id = args_opt.device_id
        context.set_context(device_id=device_id)
    # create dataset
    dataset = create_dataset(dataset_path=args_opt.data_url, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target, distribute=args_opt.run_distribute)
    step_size = dataset.get_dataset_size()
    print(step_size)
    # define net
    net = sknet(class_num=config.class_num)
    if args_opt.parameter_server:
        net.set_param_ps()

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.pre_trained)
        load_param_into_net(net, param_dict)
    else:
        for _, cell in net.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(KaimingNormal(mode='fan_out'),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))

    # init lr
    lr = get_lr(lr_init=config.lr_init, lr_end=config.lr_end, lr_max=config.lr_max,
                warmup_epochs=config.warmup_epochs, total_epochs=config.epoch_size, steps_per_epoch=step_size,
                lr_decay_mode=config.lr_decay_mode)
    lr = Tensor(lr)

    # define opt
    decayed_params = []
    no_decayed_params = []
    for param in net.trainable_params():
        if 'beta' not in param.name and 'gamma' not in param.name and 'bias' not in param.name:
            decayed_params.append(param)
        else:
            no_decayed_params.append(param)

    group_params = [{'params': decayed_params, 'weight_decay': config.weight_decay},
                    {'params': no_decayed_params},
                    {'order_params': net.trainable_params()}]
    # define loss, model
    opt = Momentum(group_params, lr, config.momentum,
                   loss_scale=config.loss_scale)
    loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(
        config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                  amp_level="O2", keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    ckpt_save_dir = ckpt_save_dir + f'_{rank_id}'
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(
            prefix="sknet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(config.epoch_size - config.pretrain_epoch_size, dataset,
                callbacks=cb, sink_size=dataset.get_dataset_size(), dataset_sink_mode=True)
    if args_opt.run_modelarts:
        mox.file.copy_parallel(ckpt_save_dir, os.path.join(args_opt.train_url, f"ckpt_{rank_id}"))
