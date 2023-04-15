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
"""
Process the test set with the .ckpt model in turn.
"""
import argparse
import os

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore.nn.loss.loss import LossBase
from mindspore.ops import functional as F
from mindspore.ops import operations as P
from mindspore.train.model import Model
from mindspore.train.serialization import load_checkpoint
from mindspore.train.serialization import load_param_into_net

import src.spnasnet as spnasnet
from src.config import imagenet_cfg
from src.dataset import create_dataset_imagenet

set_seed(1)

parser = argparse.ArgumentParser(description='single-path-nas')
parser.add_argument('--dataset_name', type=str, default='imagenet', choices=['imagenet'],
                    help='dataset name.')
parser.add_argument('--val_data_path', type=str, default=None, required=True,
                    help='Path to the validation dataset (e.g. "/datasets/imagenet/val/")')
parser.add_argument('--device_target', type=str, choices=['Ascend', 'GPU', 'CPU'], required=True,
                    default="Ascend", help='Target device: Ascend, GPU or CPU')
parser.add_argument('--checkpoint_path', type=str, default='./ckpt_0', help='Checkpoint file path or dir path',
                    required=True)
parser.add_argument('--device_id', type=int, default=None, help='device id of Ascend. (Default: None)', required=True)
args_opt = parser.parse_args()


class CrossEntropySmooth(LossBase):
    """CrossEntropy"""

    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss_ = self.ce(logit, label)
        return loss_


if __name__ == '__main__':
    device_target = args_opt.device_target
    if args_opt.dataset_name == "imagenet":
        cfg = imagenet_cfg
        dataset = create_dataset_imagenet(args_opt.val_data_path, 1, False, drop_reminder=True)
    else:
        raise ValueError("dataset is not support.")

    if not cfg.use_label_smooth:
        cfg.label_smooth_factor = 0.0
    loss = CrossEntropySmooth(sparse=True, reduction="mean",
                              smooth_factor=cfg.label_smooth_factor, num_classes=cfg.num_classes)
    net = spnasnet.spnasnet(num_classes=cfg.num_classes)
    model = Model(net, loss_fn=loss, metrics={'top_1_accuracy', 'top_5_accuracy'})

    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
    context.set_context(device_id=args_opt.device_id)

    print(f'Checkpoint path: {args_opt.checkpoint_path}')

    if os.path.isfile(args_opt.checkpoint_path) and args_opt.checkpoint_path.endswith('.ckpt'):
        param_dict = load_checkpoint(args_opt.checkpoint_path)
        load_param_into_net(net, param_dict)
        net.set_train(False)
        acc = model.eval(dataset)
        print(f"model {args_opt.checkpoint_path}'s accuracy is {acc}", flush=True)
    elif os.path.isdir(args_opt.checkpoint_path):
        file_list = sorted(os.listdir(args_opt.checkpoint_path))
        for filename in file_list:
            de_path = os.path.join(args_opt.checkpoint_path, filename)
            if de_path.endswith('.ckpt'):
                param_dict = load_checkpoint(de_path)
                load_param_into_net(net, param_dict)
                net.set_train(False)

                acc = model.eval(dataset)
                print(f"model {de_path}'s accuracy is {acc}", flush=True)
    else:
        raise ValueError("args_opt.checkpoint_path must be a checkpoint file or dir contains checkpoint(s)")
