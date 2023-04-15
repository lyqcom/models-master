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
"""train net."""
import os
import argparse
import ast
import glob
import datetime
import numpy as np
import moxing as mox
from mindspore import context
from mindspore import Tensor, export
from mindspore.nn.optim.momentum import Momentum
from mindspore.train.model import Model
from mindspore.context import ParallelMode
from mindspore.nn.loss import SoftmaxCrossEntropyWithLogits
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.common import set_seed
from mindspore.parallel import set_algo_parameters
import mindspore.nn as nn
import mindspore.common.initializer as weight_init
from src.lr_generator import get_lr
from src.CrossEntropySmooth import CrossEntropySmooth
from src.var_init import KaimingNormal


def obs_data2modelarts(FLAGS):
    """
    Copy train data from obs to modelarts by using moxing api.
    """
    start_ = datetime.datetime.now()
    print("===>>>Copy files from obs:{} to modelarts dir:{}".format(FLAGS.dataset_path, FLAGS.modelarts_data_dir))
    mox.file.copy_parallel(src_url=FLAGS.dataset_path, dst_url=FLAGS.modelarts_data_dir)
    end_ = datetime.datetime.now()
    print("===>>>Copy from obs to modelarts, time use:{}(s)".format((end_ - start_).seconds))
    files = os.listdir(FLAGS.modelarts_data_dir)
    print("===>>>Files:", files)
    if FLAGS.pre_trained:
        print("Copy ckpt from bos to modelarts")
        mox.file.copy_parallel(src_url=FLAGS.ckpt_path, dst_url=FLAGS.modelarts_ckpt_dir)
        files = os.listdir(FLAGS.modelarts_ckpt_dir)
        print("===>>>Files:", files)


def modelarts_result2obs(FLAGS):
    """
    Copy debug data from modelarts to obs.
    According to the switch flags, the debug data may contains auto tune repository,
    dump data for precision comparison, even the computation graph and profiling data.
    """

    mox.file.copy_parallel(src_url=FLAGS.modelarts_result_dir, dst_url=FLAGS.train_url)
    print("===>>>Copy Event or Checkpoint from modelarts dir:{} to obs:{}".format(FLAGS.modelarts_result_dir,
                                                                                  FLAGS.train_url))
    files = os.listdir()
    print("===>>>current Files:", files)
    mox.file.copy(src_url='sknet.air', dst_url=FLAGS.train_url + '/sknet.air')


def export_AIR(args_opt_, cfg):
    """start modelarts export"""
    ckpt_list = glob.glob(args_opt_.modelarts_result_dir + "/sknet*.ckpt")
    if not ckpt_list:
        print("ckpt file not generated.")

    ckpt_list.sort(key=os.path.getmtime)
    ckpt_model = ckpt_list[-1]
    print("checkpoint path", ckpt_model)

    net_ = sknet(cfg.class_num)
    param_dict_ = load_checkpoint(ckpt_model)
    load_param_into_net(net_, param_dict_)
    input_arr = Tensor(np.zeros([32, 3, 224, 224], np.float32))
    export(net_, input_arr, file_name="sknet", file_format='AIR')


if __name__ == '__main__':
    code_dir = os.path.dirname(__file__)
    work_dir = os.getcwd()
    print("===>>>code_dir:{}, work_dir:{}".format(code_dir, work_dir))

    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument("--train_url", type=str, default="./output")
    parser.add_argument("--modelarts_data_dir", type=str, default="/cache/dataset")
    parser.add_argument("--modelarts_result_dir", type=str, default="/cache/result")
    parser.add_argument("--modelarts_ckpt_dir", type=str, default="/cache/ckpt")
    parser.add_argument("--ckpt_path", type=str, default="./ckpt")
    parser.add_argument('--net', type=str, default="sknet50", help='sknet Model')
    parser.add_argument('--dataset', type=str, default="cifar10", help='Dataset, either cifar10 or imagenet2012')
    parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Run distribute')
    parser.add_argument('--device_id', type=int, default=0, help='Device id.')
    parser.add_argument('--device_num', type=int, default=1, help='Device num.')
    parser.add_argument('--epoch_size', type=int, default=90, help='Size of train epoch.')
    parser.add_argument('--dataset_path', type=str, default="/path/to/cifar10", help='Dataset path')
    parser.add_argument('--device_target', type=str, default='Ascend', choices="Ascend",
                        help="Device target, support Ascend.")
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
    parser.add_argument('--parameter_server', type=ast.literal_eval, default=False, help='Run parameter server train')
    args_opt = parser.parse_args()
    set_seed(1)
    target = args_opt.device_target

    if args_opt.net == "sknet50":
        from src.sknet50 import sknet50 as sknet

        if args_opt.dataset == "cifar10":
            from src.config import config1 as config
            from src.dataset import create_dataset_cifar10 as create_dataset

    obs_data2modelarts(args_opt)

    ckpt_save_dir = args_opt.modelarts_result_dir
    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    if args_opt.parameter_server:
        context.set_ps_context(enable_ps=True)
    if args_opt.run_distribute:
        device_id = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=device_id)
        context.set_auto_parallel_context(device_num=args_opt.device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
        set_algo_parameters(elementwise_op_strategy_follow=True)
        if args_opt.net == "sknet50":
            context.set_auto_parallel_context(all_reduce_fusion_config=[85, 160])
        init()
    else:
        device_id = args_opt.device_id
        context.set_context(device_id=device_id)
    # create dataset
    dataset = create_dataset(dataset_path=args_opt.modelarts_data_dir, do_train=True, repeat_num=1,
                             batch_size=config.batch_size, target=target, distribute=args_opt.run_distribute)
    step_size = dataset.get_dataset_size()
    print(step_size)
    # define net
    net = sknet(class_num=config.class_num)
    if args_opt.parameter_server:
        net.set_param_ps()

    # init weight
    if args_opt.pre_trained:
        param_dict = load_checkpoint(args_opt.modelarts_ckpt_dir + '/' + args_opt.pre_trained)
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
                warmup_epochs=config.warmup_epochs, total_epochs=args_opt.epoch_size, steps_per_epoch=step_size,
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
    if args_opt.dataset == "imagenet":
        opt = nn.SGD(group_params, lr, config.momentum, loss_scale=config.loss_scale)
        if not config.use_label_smooth:
            config.label_smooth_factor = 0.0
        loss = CrossEntropySmooth(sparse=True, reduction="mean",
                                  smooth_factor=config.label_smooth_factor, num_classes=config.class_num)
    else:
        opt = Momentum(group_params, lr, config.momentum, loss_scale=config.loss_scale)
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_scale = FixedLossScaleManager(config.loss_scale, drop_overflow_update=False)
    model = Model(net, loss_fn=loss, optimizer=opt, loss_scale_manager=loss_scale, metrics={'acc'},
                  amp_level="O2", keep_batchnorm_fp32=False)

    # define callbacks
    time_cb = TimeMonitor(data_size=step_size)
    loss_cb = LossMonitor()
    cb = [time_cb, loss_cb]
    if config.save_checkpoint:
        config_ck = CheckpointConfig(save_checkpoint_steps=config.save_checkpoint_epochs * step_size,
                                     keep_checkpoint_max=config.keep_checkpoint_max)
        ckpt_cb = ModelCheckpoint(prefix="sknet", directory=ckpt_save_dir, config=config_ck)
        cb += [ckpt_cb]

    # train model
    model.train(args_opt.epoch_size - config.pretrain_epoch_size, dataset, callbacks=cb,
                sink_size=dataset.get_dataset_size(), dataset_sink_mode=True)

    ## start export air
    if device_id == 0:
        print("start to export air model")
        start = datetime.datetime.now()
        export_AIR(args_opt, config)
        end = datetime.datetime.now()
        print("===>>end up exporting air model, time use:{}(s)".format((end - start).seconds))
    ## copy result from modelarts to obs
    modelarts_result2obs(args_opt)
