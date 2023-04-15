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
"""
network operations
"""
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


class GroupConv2d(nn.Cell):
    """
    group convolution operation.

    Args:
        in_channels (int): Input channels of feature map.
        out_channels (int): Output channels of feature map.
        kernel_size (int): Size of convolution kernel.
        stride (int): Stride size for the group convolution layer.

    Returns:
        tensor, output tensor.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 pad_mode="pad", padding=0, group=1, has_bias=False):
        super(GroupConv2d, self).__init__()
        assert in_channels % group == 0 and out_channels % group == 0
        self.group = group
        self.convs = nn.CellList()
        self.op_split = P.Split(axis=1, output_num=self.group)
        self.op_concat = P.Concat(axis=1)
        self.cast = P.Cast()
        for _ in range(group):
            self.convs.append(nn.Conv2d(in_channels // group, out_channels // group,
                                        kernel_size=kernel_size, stride=stride, has_bias=has_bias,
                                        padding=padding, pad_mode=pad_mode, group=1))

    def construct(self, x):
        features = self.op_split(x)
        outputs = ()
        for i in range(self.group):
            outputs = outputs + (self.convs[i](self.cast(features[i], mstype.float32)),)
        out = self.op_concat(outputs)
        return out
