# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
import time
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.train.callback import Callback
from src.mobilenetV2 import MobileNetV2Backbone, MobileNetV2Head, mobilenet_v2
from src.model_utils.config import config


class CrossEntropyWithLabelSmooth(nn.LossBase):
    """
    CrossEntropyWith LabelSmooth.

    Args:
        smooth_factor (float): smooth factor. Default is 0.
        num_classes (int): number of classes. Default is 1000.

    Returns:
        None.

    Examples:
        >>> CrossEntropyWithLabelSmooth(smooth_factor=0., num_classes=1000)
    """

    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropyWithLabelSmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.on_value = ms.Tensor(1.0 - smooth_factor, ms.float32)
        self.off_value = ms.Tensor(1.0 * smooth_factor /
                                   (num_classes - 1), ms.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits()
        self.mean = ops.ReduceMean(False)
        self.cast = ops.Cast()

    def construct(self, logit, label):
        one_hot_label = self.onehot(self.cast(label, ms.int32), ops.shape(logit)[1],
                                    self.on_value, self.off_value)
        out_loss = self.ce(logit, one_hot_label)
        out_loss = self.mean(out_loss, 0)
        return out_loss


class Monitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy array): train lr

    Returns:
        None

    Examples:
        >>> Monitor(100,lr_init=ms.Tensor([0.05]*100).asnumpy())
    """

    def __init__(self, lr_init=None, model=None, eval_dataset=None):
        super(Monitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)
        self.model = model
        self.eval_dataset = eval_dataset
        self.best_acc = 0.

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num

        eval_acc = None
        if self.model is not None and self.eval_dataset is not None:
            eval_acc = self.model.eval(self.eval_dataset)["acc"]
        log = "epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses))
        if eval_acc is not None:
            if eval_acc > self.best_acc:
                self.best_acc = eval_acc
            log += ", eval_acc: {:.6f}, best_acc: {:.6f}".format(eval_acc, self.best_acc)
        print(log, flush=True)

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], ms.Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, ms.Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))


def load_ckpt(network, pretrain_ckpt_path, trainable=True):
    """load checkpoint into network."""
    param_dict = ms.load_checkpoint(pretrain_ckpt_path)
    ms.load_param_into_net(network, param_dict)
    if not trainable:
        for param in network.get_parameters():
            param.requires_grad = False


def define_net(cfg, is_training=True):
    backbone_net = MobileNetV2Backbone()
    activation = cfg.activation if not is_training else "None"
    head_net = MobileNetV2Head(input_channel=backbone_net.out_channels,
                               num_classes=cfg.num_classes,
                               activation=activation)
    net = mobilenet_v2(backbone_net, head_net)
    return backbone_net, head_net, net


def build_params_groups(net):
    """build params groups"""
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
    return group_params
