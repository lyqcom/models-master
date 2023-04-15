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
Image classifiation.
"""
import math
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common import initializer as init
from mindspore.common.initializer import initializer, HeNormal

from src.config import config
npy = config['vgg_npy']

def _make_layer(base, args, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = 3
    layer = []
    for v in base:
        if v == 'M':
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
            layers.append(layer)
            layer = []
        else:
            weight = 'ones'
            if args.initialize_mode == "XavierUniform":
                weight_shape = (v, in_channels, 3, 3)
                weight = initializer('XavierUniform', shape=weight_shape, dtype=mstype.float32).init_data()

            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=args.padding,
                               pad_mode=args.pad_mode,
                               has_bias=args.has_bias,
                               weight_init=weight)
            if batch_norm:
                layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layer += [conv2d, nn.ReLU()]
            in_channels = v
    layer1 = nn.SequentialCell(layers[0])
    layer2 = nn.SequentialCell(layers[1])
    layer3 = nn.SequentialCell(layers[2])
    layer4 = nn.SequentialCell(layers[3])
    layer5 = nn.SequentialCell(layers[4])

    return layer1, layer2, layer3, layer4, layer5


class Vgg(nn.Cell):
    """
    VGG network definition.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        num_classes (int): Class numbers. Default: 1000.
        batch_norm (bool): Whether to do the batchnorm. Default: False.
        batch_size (int): Batch size. Default: 1.

    Returns:
        Tensor, infer output tensor.

    Examples:
        >>> Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        >>>     num_classes=1000, batch_norm=False, batch_size=1)
    """

    def __init__(self, base, num_classes=1000, batch_norm=False, batch_size=1, args=None, phase="train"):
        super(Vgg, self).__init__()
        _ = batch_size
        self.layer1, self.layer2, self.layer3, self.layer4, self.layer5 = _make_layer(base, args, batch_norm=batch_norm)
        if args.initialize_mode == "KaimingNormal":
            #default_recurisive_init(self)
            self.custom_init_weight()

    def construct(self, x):
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        return l2, l3, l4, l5

    def custom_init_weight(self):
        """
        Init the weight of Conv2d and Dense in the net.
        """
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(init.initializer(
                    HeNormal(negative_slope=math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                    cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(init.initializer(
                    init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))


cfg = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(num_classes=1000, args=None, phase="train"):
    """
    Get Vgg16 neural network with batch normalization.

    Args:
        num_classes (int): Class numbers. Default: 1000.
        args(namespace): param for net init.
        phase(str): train or test mode.

    Returns:
        Cell, cell instance of Vgg16 neural network with batch normalization.

    Examples:
        >>> vgg16(num_classes=1000, args=args)
    """

    if args is None:
        from src.config import cifar_cfg
        args = cifar_cfg
    net = Vgg(cfg['16'], num_classes=num_classes, args=args, batch_norm=args.batch_norm, phase=phase)
    return net
