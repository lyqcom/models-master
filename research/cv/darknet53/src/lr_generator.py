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
"""learning rate generator"""
import math
import numpy as np


def get_lr(config, steps_per_epoch):
    """
    generate learning rate array

    Args:
       config: config file

    Returns:
       np.array, learning rate array
    """

    if config.lr_scheduler == "cosine_annealing":
        lr_each_step = warmup_cosine_annealing_lr(config.lr,
                                                  steps_per_epoch,
                                                  config.warmup_epochs,
                                                  config.epoch_num,
                                                  config.T_max,
                                                  config.eta_min)
    else:
        raise NotImplementedError

    lr_each_step = np.array(lr_each_step).astype(np.float32)[config.pretrained_epoch_num * steps_per_epoch:]

    assert len(lr_each_step) == (config.epoch_num - config.pretrained_epoch_num) * steps_per_epoch
    return lr_each_step


def linear_warmup_lr(current_step, warmup_steps, base_lr, init_lr):
    """Applies liner decay to generate learning rate array."""

    lr_inc = (float(base_lr) - float(init_lr)) / float(warmup_steps)
    lr = float(init_lr) + lr_inc * current_step
    return lr


def warmup_cosine_annealing_lr(lr, steps_per_epoch, warmup_epochs, max_epoch, T_max, eta_min=0):
    """Applies cosine decay to generate learning rate array."""

    base_lr = lr
    warmup_init_lr = 0
    total_steps = int(max_epoch * steps_per_epoch)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    lr_each_step = []
    for i in range(total_steps):
        last_epoch = i // steps_per_epoch
        if i < warmup_steps:
            lr = linear_warmup_lr(i + 1, warmup_steps, base_lr, warmup_init_lr)
        else:
            lr = eta_min + (base_lr - eta_min) * (1. + math.cos(math.pi * last_epoch / T_max)) / 2
        lr_each_step.append(lr)

    return np.array(lr_each_step).astype(np.float32)
