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
"""train"""

import argparse
import os
import numpy as np
import mindspore.nn as nn
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.communication.management import init, get_rank
from mindspore import load_checkpoint, load_param_into_net, export, save_checkpoint
import src.dataset as dt
from src.lr_generator import _generate_steps_lr
from src.config import relationnet_cfg as cfg
from src.relationnet import Encoder_Relation, weight_init, TrainOneStepCell
from src.net_train import train

parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-g", "--gpu", type=int, default=0)
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-dt", "--device_target", type=str, default='Ascend', choices=("Ascend", "GPU", "CPU"),
                    help="Device target, support Ascend, GPU and CPU.")
parser.add_argument("-di", "--device_id", type=int, default=0, help='device id of GPU or Ascend. (Default: 0)')
parser.add_argument("--ckpt_dir", default='/cache/ckpt', help='the path of output')
parser.add_argument("--data_path", default='/cache/data',
                    help="Path where the dataset is saved")
parser.add_argument("--data_url", default="")
parser.add_argument("--train_url", default='')
parser.add_argument('--episode', type=int, default=1000000, help='epochs  for training')
parser.add_argument('--class_num', type=int, default=5, help="")
parser.add_argument("--learning_rate", type=int, default=5, help="")
parser.add_argument("--feature_dim", type=int, default=64)
parser.add_argument("--relation_dim", type=int, default=32)
parser.add_argument("--cloud", default=True, help='if run on cloud')
parser.add_argument("--file_name", type=str, default="relationnet", help="output file name.")

parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="AIR", help="file format")
args = parser.parse_args()

# init operators
scatter = ops.ScatterNd()
concat0dim = ops.Concat(axis=0)


def main():
    local_data_url = args.data_path
    local_train_url = args.ckpt_dir
    # if run on the cloud
    if args.cloud:
        import moxing as mox
        device_target = args.device_target
        device_num = int(os.getenv("RANK_SIZE"))
        device_id = int(os.getenv("DEVICE_ID"))
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)##ascend
        context.set_context(save_graphs=False)
        if device_target == "Ascend":
            context.set_context(device_id=device_id)
            if device_num > 1:
                cfg.episode = int(cfg.episode / 2)
                cfg.learning_rate = cfg.learning_rate*2
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
                init()
                local_data_url = os.path.join(local_data_url, str(device_id))
                local_train_url = os.path.join(local_train_url, "_" + str(get_rank()))
        elif device_target == "GPU":
            if device_num > 1:
                init()
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
        else:
            raise ValueError("Unsupported platform.")
        import moxing as mox
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
    else:
        # run on the local server
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
        context.set_context(save_graphs=False)

    # Step 1 : create output dir
    if not os.path.exists(local_train_url): ##args.ckpt_dir
        os.makedirs(local_train_url)

    # Step 2 : init operators

    # Step 3 : init data folders
    print("init data folders")
    metatrain_character_folders, metatest_character_folders = dt.omniglot_character_folders(data_path=local_data_url)

    # Step 4 : init networks
    print("init neural networks")
    encoder_relation = Encoder_Relation(cfg.feature_dim, cfg.relation_dim)
    weight_init(encoder_relation)

    # Step 5 : load parameters
    load_ckpts = False

    print("init optim, loss")
    if load_ckpts:
        lr = Tensor(nn.piecewise_constant_lr(milestone=[50000 * i for i in range(1, 21)],
                                             learning_rates=[0.0001 * 0.5 ** i for i in range(0, 20)]),
                    dtype=mstype.float32)
    else:
        lr = _generate_steps_lr(lr_init=0.0005, lr_max=cfg.learning_rate, total_steps=1000000, warmup_steps=100)

    optim = nn.Adam(encoder_relation.trainable_params(), learning_rate=lr)

    print("init loss function and grads")
    criterion = nn.MSELoss()
    netloss = nn.WithLossCell(encoder_relation, criterion)
    net_g = TrainOneStepCell(netloss, optim)

    # train
    train(metatrain_character_folders=metatrain_character_folders, metatest_character_folders=metatest_character_folders
          , netloss=netloss, net_g=net_g, encoder_relation=encoder_relation, local_train_url=local_train_url, args=args)

    #****
    save_checkpoint(net_g, local_train_url + '/relationnet.ckpt')

    network = Encoder_Relation(cfg.feature_dim, cfg.relation_dim)
    # load network checkpoint
    param_dict = load_checkpoint(os.path.join(local_train_url, "relationnet.ckpt"))
    load_param_into_net(network, param_dict)

    # export network
    inputs = Tensor(np.ones([10, 1, cfg.image_height, cfg.image_width]), mstype.float32)
    export(network, inputs, file_name=local_train_url+'/relationnet', file_format=args.file_format)
    if args.cloud:
        import moxing as mox
        mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
    #****
if __name__ == '__main__':
    main()
