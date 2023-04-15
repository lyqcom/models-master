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
import math
import random
from PIL import ImageFilter
import mindspore as ms
from mindspore import ops


class GaussianBlur():
    def __init__(self, sigma_low=0.1, sigma_upper=2.0):
        self.sigma = [sigma_low, sigma_upper]

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(args, optimizer, cur_iter, total_iter):
    lr = args.lr
    eta_min = lr * 1e-3
    lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * cur_iter / total_iter)) / 2

    ops.assign(optimizer.learning_rate, ms.Tensor(lr, ms.float32))
