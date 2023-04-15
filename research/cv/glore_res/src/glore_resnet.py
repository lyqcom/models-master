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
"""glore_resnet"""
from collections import OrderedDict
import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight, has_bias=False)


def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight, has_bias=False)


def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=7, stride=stride, padding=3, pad_mode='pad', weight_init=weight, has_bias=False)


def _bn(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.92,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
def _bn_50(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.08,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

def _bn_last(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.92,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)
def _bn_last_50(channel):
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.92,
                          gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)

def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class BN_AC_Conv(nn.Cell):
    """
     Basic convolution block.
    """
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel=1,
                 pad=0,
                 pad_mode='same',
                 stride=1,
                 groups=1,
                 has_bias=False):
        super(BN_AC_Conv, self).__init__()
        self.bn = _bn(in_channel)
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              pad_mode=pad_mode,
                              padding=pad,
                              kernel_size=kernel,
                              stride=stride,
                              has_bias=has_bias,
                              group=groups)

    def construct(self, x):
        out = self.bn(x)
        out = self.relu(out)
        out = self.conv(out)
        return out


class GCN(nn.Cell):
    """
     Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_mode, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_mode, num_mode, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, has_bias=bias)
        self.transpose = ops.Transpose()
        self.add = P.Add()

    def construct(self, x):
        """construct"""
        identity = x
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        out = self.transpose(x, (0, 2, 1))
        out = self.conv1(out)
        out = self.transpose(out, (0, 2, 1))
        out = self.add(out, identity)
        out = self.relu2(out)
        out = self.conv2(out)
        return out

class GloreUnit_50(nn.Cell):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    Args:
        num_in: Input channel
        num_mid:
    """

    def __init__(self, num_in, num_mid,
                 normalize=False):
        super(GloreUnit_50, self).__init__()
        self.normalize = normalize
        self.num_s = int(2 * num_mid)  # 512   num_in = 1024
        self.num_n = int(1 * num_mid)  # 256
        # reduce dim
        self.conv_state = nn.SequentialCell([_bn_50(num_in),
                                             nn.ReLU(),
                                             _conv1x1(num_in, self.num_s, stride=1)])
        # projection map
        self.conv_proj = nn.SequentialCell([_bn_50(num_in),
                                            nn.ReLU(),
                                            _conv1x1(num_in, self.num_n, stride=1)])

        self.gcn = GCN(num_state=self.num_s, num_mode=self.num_n)

        self.conv_extend = nn.SequentialCell([_bn_last_50(self.num_s),
                                              nn.ReLU(),
                                              _conv1x1(self.num_s, num_in, stride=1)])

        self.reshape = P.Reshape()
        self.matmul = P.BatchMatMul()
        self.transpose = P.Transpose()
        self.add = P.Add()
        self.cast = P.Cast()

    def construct(self, x):
        """construct Graph-based Global Reasoning Unit"""
        n = x.shape[0]
        identity = x
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_conv_state = self.conv_state(x)
        x_state_reshaped = self.reshape(x_conv_state, (n, self.num_s, -1))

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_conv_proj = self.conv_proj(x)
        x_proj_reshaped = self.reshape(x_conv_proj, (n, self.num_n, -1))

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_proj_reshaped = self.transpose(x_proj_reshaped, (0, 2, 1))

        x_state_reshaped_fp16 = self.cast(x_state_reshaped, mstype.float16)
        x_proj_reshaped_fp16 = self.cast(x_proj_reshaped, mstype.float16)
        x_n_state_fp16 = self.matmul(x_state_reshaped_fp16, x_proj_reshaped_fp16)
        x_n_state = self.cast(x_n_state_fp16, mstype.float32)

        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.shape[2])

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_n_rel_fp16 = self.cast(x_n_rel, mstype.float16)
        x_rproj_reshaped_fp16 = self.cast(x_rproj_reshaped, mstype.float16)
        x_state_reshaped_fp16 = self.matmul(x_n_rel_fp16, x_rproj_reshaped_fp16)
        x_state_reshaped = self.cast(x_state_reshaped_fp16, mstype.float32)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = self.reshape(x_state_reshaped, (n, self.num_s, identity.shape[2], identity.shape[3]))

        # (n, num_state, h, w) -> (n, num_in, h, w)
        x_conv_extend = self.conv_extend(x_state)
        out = self.add(x_conv_extend, identity)
        return out

class GloreUnit_200(nn.Cell):
    """
    Graph-based Global Reasoning Unit
    Parameter:
        'normalize' is not necessary if the input size is fixed
    Args:
        num_in: Input channel
        num_mid:
    """

    def __init__(self, num_in, num_mid, cast_fp16=True,
                 normalize=False):
        super(GloreUnit_200, self).__init__()
        self.normalize = normalize
        self.cast_fp16 = cast_fp16
        self.num_s = int(2 * num_mid)  # 512   num_in = 1024
        self.num_n = int(1 * num_mid)  # 256
        # reduce dim
        self.conv_state = nn.SequentialCell([_bn(num_in),
                                             nn.ReLU(),
                                             _conv1x1(num_in, self.num_s, stride=1)])
        # projection map
        self.conv_proj = nn.SequentialCell([_bn(num_in),
                                            nn.ReLU(),
                                            _conv1x1(num_in, self.num_n, stride=1)])

        self.gcn = GCN(num_state=self.num_s, num_mode=self.num_n)

        self.conv_extend = nn.SequentialCell([_bn_last(self.num_s),
                                              nn.ReLU(),
                                              _conv1x1(self.num_s, num_in, stride=1)])

        self.reshape = ops.Reshape()
        self.matmul = ops.BatchMatMul()
        self.transpose = ops.Transpose()
        self.add = P.Add()
        self.cast = P.Cast()

    def construct(self, x):
        """construct"""
        n = x.shape[0]
        identity = x
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_conv_state = self.conv_state(x)
        x_state_reshaped = self.reshape(x_conv_state, (n, self.num_s, -1))

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_conv_proj = self.conv_proj(x)
        x_proj_reshaped = self.reshape(x_conv_proj, (n, self.num_n, -1))

        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_proj_reshaped = self.transpose(x_proj_reshaped, (0, 2, 1))

        if self.cast_fp16:
            # 提高速度
            x_state_reshaped_fp16 = self.cast(x_state_reshaped, mstype.float16)
            x_proj_reshaped_fp16 = self.cast(x_proj_reshaped, mstype.float16)
            x_n_state_fp16 = self.matmul(x_state_reshaped_fp16, x_proj_reshaped_fp16)
            x_n_state = self.cast(x_n_state_fp16, mstype.float32)
        else:
            x_n_state = self.matmul(x_state_reshaped, x_proj_reshaped)

        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.shape[2])

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)

        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        if self.cast_fp16:
            x_n_rel_fp16 = self.cast(x_n_rel, mstype.float16)
            x_rproj_reshaped_fp16 = self.cast(x_rproj_reshaped, mstype.float16)
            x_state_reshaped_fp16 = self.matmul(x_n_rel_fp16, x_rproj_reshaped_fp16)
            x_state_reshaped = self.cast(x_state_reshaped_fp16, mstype.float32)
        else:
            x_state_reshaped = self.matmul(x_n_rel, x_rproj_reshaped)

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = self.reshape(x_state_reshaped, (n, self.num_s, identity.shape[2], identity.shape[3]))

        # (n, num_state, h, w) -> (n, num_in, h, w)
        x_conv_extend = self.conv_extend(x_state)
        out = self.add(x_conv_extend, identity)
        return out


class Residual_Unit_200(nn.Cell):
    """
    Residual unit used in Resnet
    """
    def __init__(self,
                 in_channel,
                 mid_channel,
                 out_channel,
                 groups=1,
                 stride=1,
                 first_block=False):
        super(Residual_Unit_200, self).__init__()
        self.first_block = first_block
        self.BN_AC_Conv1 = BN_AC_Conv(in_channel, mid_channel, kernel=1, pad=0)
        self.BN_AC_Conv2 = BN_AC_Conv(mid_channel, mid_channel, kernel=3, pad_mode='pad', pad=1, stride=stride,
                                      groups=groups)
        self.BN_AC_Conv3 = BN_AC_Conv(mid_channel, out_channel, kernel=1, pad=0)
        if self.first_block:
            self.BN_AC_Conv_w = BN_AC_Conv(in_channel, out_channel, kernel=1, pad=0, stride=stride)
        self.add = P.Add()

    def construct(self, x):
        identity = x
        out = self.BN_AC_Conv1(x)
        out = self.BN_AC_Conv2(out)
        out = self.BN_AC_Conv3(out)
        if self.first_block:
            identity = self.BN_AC_Conv_w(identity)

        out = self.add(out, identity)
        return out

class ResidualBlock(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlock(3, 256, stride=2)
    """
    expansion = 4

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1):
        super(ResidualBlock, self).__init__()
        self.stride = stride
        channel = out_channel // self.expansion
        self.conv1 = _conv1x1(in_channel, channel, stride=1)
        self.bn1 = _bn_50(channel)

        self.conv2 = _conv3x3(channel, channel, stride=stride)
        self.bn2 = _bn_50(channel)

        self.conv3 = _conv1x1(channel, out_channel, stride=1)
        self.bn3 = _bn_last_50(out_channel)
        self.relu = nn.ReLU()
        self.down_sample = False
        if stride != 1 or in_channel != out_channel:
            self.down_sample = True
        self.down_sample_layer = None

        if self.down_sample:
            self.down_sample_layer = nn.SequentialCell(
                [
                    nn.AvgPool2d(kernel_size=stride, stride=stride),
                    _conv1x1(in_channel, out_channel, stride=1),
                    _bn_50(out_channel)
                ])
        self.add = P.Add()

    def construct(self, x):
        """construct ResidualBlock"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            identity = self.down_sample_layer(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out

class ResNet_101(nn.Cell):
    """
    Resnet architecture
    """
    def __init__(self,
                 layer_nums,
                 num_classes,
                 use_glore=False,
                 cast_fp16=True):
        super(ResNet_101, self).__init__()
        self.layer1 = nn.SequentialCell(OrderedDict([
            ('conv', _conv7x7(3, 64, stride=2)),
            ('bn', _bn(64),),
            ('relu', nn.ReLU(),),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same"))
        ]))

        num_in = [64, 256, 512, 1024]
        num_mid = [64, 128, 256, 512]
        num_out = [256, 512, 1024, 2048]
        self.layer2 = nn.SequentialCell(OrderedDict([
            ("Residual_Unit{}".format(i), Residual_Unit_200(in_channel=(num_in[0] if i == 1 else num_out[0]),
                                                            mid_channel=num_mid[0],
                                                            out_channel=num_out[0],
                                                            stride=1,
                                                            first_block=(i == 1))) for i in range(1, layer_nums[0] + 1)
        ]))

        blocks_layer3 = []
        for i in range(1, layer_nums[1] + 1):
            blocks_layer3.append(
                ("Residual_Unit{}".format(i), Residual_Unit_200(in_channel=(num_in[1] if i == 1 else num_out[1]),
                                                                mid_channel=num_mid[1],
                                                                out_channel=num_out[1],
                                                                stride=(2 if i == 1 else 1),
                                                                first_block=(i == 1))))
            if use_glore and i in [1, 3]:
                blocks_layer3.append(("Residual_Unit{}_GloreUnit".format(i), GloreUnit_200(num_out[1], num_mid[1],
                                                                                           cast_fp16)))
        self.layer3 = nn.SequentialCell(OrderedDict(blocks_layer3))

        blocks_layer4 = []
        for i in range(1, layer_nums[2] + 1):
            blocks_layer4.append(
                ("Residual_Unit{}".format(i), Residual_Unit_200(in_channel=(num_in[2] if i == 1 else num_out[2]),
                                                                mid_channel=num_mid[2],
                                                                out_channel=num_out[2],
                                                                stride=(2 if i == 1 else 1),
                                                                first_block=(i == 1))))
            if use_glore and i in [3, 11, 19]:
                blocks_layer4.append(("Residual_Unit{}_GloreUnit".format(i), GloreUnit_200(num_out[2], num_mid[2],
                                                                                           cast_fp16)))
        self.layer4 = nn.SequentialCell(OrderedDict(blocks_layer4))

        self.layer5 = nn.SequentialCell(OrderedDict([
            ("Residual_Unit{}".format(i), Residual_Unit_200(in_channel=(num_in[3] if i == 1 else num_out[3]),
                                                            mid_channel=num_mid[3],
                                                            out_channel=num_out[3],
                                                            stride=(2 if i == 1 else 1),
                                                            first_block=(i == 1))) for i in range(1, layer_nums[3] + 1)
        ]))

        self.tail = nn.SequentialCell(OrderedDict([
            ('bn', _bn(num_out[3])),
            ('relu', nn.ReLU())
        ]))

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.classifier = _fc(num_out[3], num_classes)
        self.print = ops.Print()

    def construct(self, x):
        """construct"""
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        out = self.tail(c5)
        out = self.mean(out, (2, 3))
        out = self.flatten(out)
        out = self.classifier(out)
        return out

class ResNet_200(nn.Cell):
    """
    Resnet architecture
    """
    def __init__(self,
                 layer_nums,
                 num_classes,
                 use_glore=False,
                 cast_fp16=True):
        super(ResNet_200, self).__init__()
        self.layer1 = nn.SequentialCell(OrderedDict([
            ('conv', _conv7x7(3, 64, stride=2)),
            ('bn', _bn(64),),
            ('relu', nn.ReLU(),),
            ('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same"))
        ]))

        num_in = [64, 256, 512, 1024]
        num_mid = [64, 128, 256, 512]
        num_out = [256, 512, 1024, 2048]
        self.layer2 = nn.SequentialCell(OrderedDict([
            ("Residual_Unit{}".format(i), Residual_Unit_200(in_channel=(num_in[0] if i == 1 else num_out[0]),
                                                            mid_channel=num_mid[0],
                                                            out_channel=num_out[0],
                                                            stride=1,
                                                            first_block=(i == 1))) for i in range(1, layer_nums[0] + 1)
        ]))

        blocks_layer3 = []
        for i in range(1, layer_nums[1] + 1):
            blocks_layer3.append(
                ("Residual_Unit{}".format(i), Residual_Unit_200(in_channel=(num_in[1] if i == 1 else num_out[1]),
                                                                mid_channel=num_mid[1],
                                                                out_channel=num_out[1],
                                                                stride=(2 if i == 1 else 1),
                                                                first_block=(i == 1))))
            if use_glore and i in [12, 18]:
                blocks_layer3.append(("Residual_Unit{}_GloreUnit".format(i), GloreUnit_200(num_out[1], num_mid[1],
                                                                                           cast_fp16)))
        self.layer3 = nn.SequentialCell(OrderedDict(blocks_layer3))

        blocks_layer4 = []
        for i in range(1, layer_nums[2] + 1):
            blocks_layer4.append(
                ("Residual_Unit{}".format(i), Residual_Unit_200(in_channel=(num_in[2] if i == 1 else num_out[2]),
                                                                mid_channel=num_mid[2],
                                                                out_channel=num_out[2],
                                                                stride=(2 if i == 1 else 1),
                                                                first_block=(i == 1))))
            if use_glore and i in [16, 24, 32]:
                blocks_layer4.append(("Residual_Unit{}_GloreUnit".format(i), GloreUnit_200(num_out[2], num_mid[2],
                                                                                           cast_fp16)))
        self.layer4 = nn.SequentialCell(OrderedDict(blocks_layer4))

        self.layer5 = nn.SequentialCell(OrderedDict([
            ("Residual_Unit{}".format(i), Residual_Unit_200(in_channel=(num_in[3] if i == 1 else num_out[3]),
                                                            mid_channel=num_mid[3],
                                                            out_channel=num_out[3],
                                                            stride=(2 if i == 1 else 1),
                                                            first_block=(i == 1))) for i in range(1, layer_nums[3] + 1)
        ]))

        self.tail = nn.SequentialCell(OrderedDict([
            ('bn', _bn(num_out[3])),
            ('relu', nn.ReLU())
        ]))

        self.mean = ops.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.classifier = _fc(num_out[3], num_classes)
        self.print = ops.Print()

    def construct(self, x):
        """construct"""
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        c5 = self.layer5(c4)

        out = self.tail(c5)
        out = self.mean(out, (2, 3))
        out = self.flatten(out)
        out = self.classifier(out)
        return out

class ResNet_50(nn.Cell):
    """
    ResNet architecture.

    Args:
        block (Cell): Block for network.
        layer_nums (list): Numbers of block in different layers.
        in_channels (list): Input channel in each layer.
        out_channels (list): Output channel in each layer.
        strides (list):  Stride size in each layer.
        num_classes (int): The number of classes that the training images are belonging to.
        usr_glore (bool): Whether glore-unit should be appended in this model.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResNet(ResidualBlock,
        >>>        [3, 4, 6, 3],
        >>>        [64, 256, 512, 1024],
        >>>        [256, 512, 1024, 2048],
        >>>        [1, 2, 2, 2],
        >>>        10)
    """

    def __init__(self,
                 block,
                 layer_nums,
                 in_channels,
                 out_channels,
                 strides,
                 num_classes,
                 use_glore=False):
        super(ResNet_50, self).__init__()

        if not len(layer_nums) == len(in_channels) == len(out_channels) == 4:
            raise ValueError("the length of layer_num, in_channels, out_channels list must be 4!")

        self.conv1 = nn.SequentialCell(OrderedDict([
            ('conv_1', _conv3x3(3, 32, stride=2)),
            ('bn1', _bn_50(32)),
            ('relu1', nn.ReLU()),
            ('conv_2', _conv3x3(32, 32, stride=1)),
            ('bn2', _bn_50(32)),
            ('relu2', nn.ReLU()),
            ('conv_3', _conv3x3(32, 64, stride=1)),
            ]))
        self.bn1 = _bn_50(64)
        self.relu = P.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block,
                                       layer_nums[0],
                                       in_channel=in_channels[0],
                                       out_channel=out_channels[0],
                                       stride=strides[0])
        self.layer2 = self._make_layer(block,
                                       layer_nums[1],
                                       in_channel=in_channels[1],
                                       out_channel=out_channels[1],
                                       stride=strides[1])
        self.layer3 = self._make_layer(block,
                                       layer_nums[2],
                                       in_channel=in_channels[2],
                                       out_channel=out_channels[2],
                                       stride=strides[2],
                                       use_glore=use_glore)
        self.layer4 = self._make_layer(block,
                                       layer_nums[3],
                                       in_channel=in_channels[3],
                                       out_channel=out_channels[3],
                                       stride=strides[3])
        self.mean = P.ReduceMean(keep_dims=True)
        self.flatten = nn.Flatten()
        self.end_point = _fc(out_channels[3], num_classes)

    def _make_layer(self, block, layer_num, in_channel, out_channel, stride,
                    use_glore=False, glore_pos=None):
        """
        Make stage network of ResNet.

        Args:
            block (Cell): Resnet block.
            layer_num (int): Layer number.
            in_channel (int): Input channel.
            out_channel (int): Output channel.
            stride (int): Stride size for the first convolutional layer.
            usr_glore (bool): Whether glore-unit should be appended in this model.
            glore_pos (list of int): Which layers to append glore-unit to.
        Returns:
            SequentialCell, the output layer.

        Examples:
            >>> _make_layer(ResidualBlock, 3, 128, 256, 2)
        """
        if use_glore and glore_pos is None:
            glore_pos = [1, 3, 5]

        layers = []
        for i in range(1, layer_num + 1):
            resnet_block = block(in_channel=(in_channel if i == 1 else out_channel),
                                 out_channel=out_channel,
                                 stride=(stride if i == 1 else 1))
            layers.append(resnet_block)
            if use_glore and i in glore_pos:
                glore_unit = GloreUnit_50(out_channel, int(out_channel / 4))
                layers.append(glore_unit)
        return nn.SequentialCell(layers)

    def construct(self, x):
        """construct ResNet"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        c1 = self.maxpool(x)

        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        out = self.mean(c5, (2, 3))
        out = self.flatten(out)
        out = self.end_point(out)

        return out

def glore_resnet200(cast_fp16=True, class_num=1000, use_glore=True):
    return ResNet_200(layer_nums=[3, 24, 36, 3],
                      num_classes=class_num,
                      use_glore=use_glore,
                      cast_fp16=cast_fp16)

def glore_resnet50(class_num=1000, use_glore=True):
    """
    Get ResNet50 with GloreUnit neural network.

    Args:
        class_num (int): The number of classes that the training images are belonging to.
        use_glore (bool): Whether glore-unit should be appended in this model.
    Returns:
        Cell, cell instance of ResNet50 neural network.

    Examples:
        >>> net = glore_resnet50(10)
    """
    return ResNet_50(ResidualBlock,
                     [3, 4, 6, 3],
                     [64, 256, 512, 1024],
                     [256, 512, 1024, 2048],
                     [1, 2, 2, 2],
                     class_num,
                     use_glore=use_glore)

def glore_resnet101(cast_fp16=True, class_num=1000, use_glore=True):
    return ResNet_101(layer_nums=[3, 4, 23, 3],
                      num_classes=class_num,
                      use_glore=use_glore,
                      cast_fp16=cast_fp16)
