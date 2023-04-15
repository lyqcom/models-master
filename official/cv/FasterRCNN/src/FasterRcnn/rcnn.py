# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""FasterRcnn Rcnn network."""

import numpy as np
import mindspore as ms
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter


class DenseNoTranpose(nn.Cell):
    """Dense method"""

    def __init__(self, input_channels, output_channels, weight_init):
        super(DenseNoTranpose, self).__init__()
        self.weight = Parameter(ms.common.initializer.initializer(weight_init, \
                                                                  [input_channels, output_channels], ms.float32))
        self.bias = Parameter(ms.common.initializer.initializer("zeros", \
                                                                [output_channels], ms.float32))

        self.matmul = ops.MatMul(transpose_b=False)
        self.bias_add = ops.BiasAdd()
        self.cast = ops.Cast()
        self.device_type = "Ascend" if ms.get_context("device_target") == "Ascend" else "Others"

    def construct(self, x):
        if self.device_type == "Ascend":
            x = self.cast(x, ms.float16)
            weight = self.cast(self.weight, ms.float16)
            output = self.bias_add(self.matmul(x, weight), self.bias)
        else:
            output = self.bias_add(self.matmul(x, self.weight), self.bias)
        return output


class Rcnn(nn.Cell):
    """
    Rcnn subnet.

    Args:
        config (dict) - Config.
        representation_size (int) - Channels of shared dense.
        batch_size (int) - Batchsize.
        num_classes (int) - Class number.
        target_means (list) - Means for encode function. Default: (.0, .0, .0, .0]).
        target_stds (list) - Stds for encode function. Default: (0.1, 0.1, 0.2, 0.2).

    Returns:
        Tuple, tuple of output tensor.

    Examples:
        Rcnn(config=config, representation_size = 1024, batch_size=2, num_classes = 81, \
             target_means=(0., 0., 0., 0.), target_stds=(0.1, 0.1, 0.2, 0.2))
    """

    def __init__(self,
                 config,
                 representation_size,
                 batch_size,
                 num_classes,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2)
                 ):
        super(Rcnn, self).__init__()
        cfg = config
        self.dtype = np.float32
        self.ms_type = ms.float32
        self.rcnn_loss_cls_weight = Tensor(np.array(cfg.rcnn_loss_cls_weight).astype(self.dtype))
        self.rcnn_loss_reg_weight = Tensor(np.array(cfg.rcnn_loss_reg_weight).astype(self.dtype))
        self.rcnn_fc_out_channels = cfg.rcnn_fc_out_channels
        self.target_means = target_means
        self.target_stds = target_stds
        self.without_bg_loss = config.without_bg_loss
        self.num_classes = num_classes
        self.num_classes_fronted = num_classes
        if self.without_bg_loss:
            self.num_classes_fronted = num_classes - 1
        self.in_channels = cfg.rcnn_in_channels
        self.train_batch_size = batch_size
        self.test_batch_size = cfg.test_batch_size

        shape_0 = (self.rcnn_fc_out_channels, representation_size)
        weights_0 = ms.common.initializer.initializer("XavierUniform", shape=shape_0[::-1], \
                                                      dtype=self.ms_type).init_data()
        shape_1 = (self.rcnn_fc_out_channels, self.rcnn_fc_out_channels)
        weights_1 = ms.common.initializer.initializer("XavierUniform", shape=shape_1[::-1], \
                                                      dtype=self.ms_type).init_data()
        self.shared_fc_0 = DenseNoTranpose(representation_size, self.rcnn_fc_out_channels, weights_0)
        self.shared_fc_1 = DenseNoTranpose(self.rcnn_fc_out_channels, self.rcnn_fc_out_channels, weights_1)

        cls_weight = ms.common.initializer.initializer('Normal', shape=[num_classes, self.rcnn_fc_out_channels][::-1],
                                                       dtype=self.ms_type).init_data()
        reg_weight = ms.common.initializer.initializer('Normal', shape=[self.num_classes_fronted * 4,
                                                                        self.rcnn_fc_out_channels][::-1],
                                                       dtype=self.ms_type).init_data()
        self.cls_scores = DenseNoTranpose(self.rcnn_fc_out_channels, num_classes, cls_weight)
        self.reg_scores = DenseNoTranpose(self.rcnn_fc_out_channels, self.num_classes_fronted * 4, reg_weight)

        self.flatten = ops.Flatten()
        self.relu = ops.ReLU()
        self.logicaland = ops.LogicalAnd()
        self.loss_cls = ops.SoftmaxCrossEntropyWithLogits()
        self.loss_bbox = ops.SmoothL1Loss(beta=1.0)
        self.reshape = ops.Reshape()
        self.onehot = ops.OneHot()
        self.greater = ops.Greater()
        self.cast = ops.Cast()
        self.sum_loss = ops.ReduceSum()
        self.tile = ops.Tile()
        self.expandims = ops.ExpandDims()

        self.gather = ops.GatherNd()
        self.argmax = ops.ArgMaxWithValue(axis=1)

        self.on_value = Tensor(1.0, ms.float32)
        self.off_value = Tensor(0.0, ms.float32)
        self.value = Tensor(1.0, self.ms_type)

        self.num_bboxes = (cfg.num_expected_pos_stage2 + cfg.num_expected_neg_stage2) * batch_size

        rmv_first = np.ones((self.num_bboxes, self.num_classes_fronted))
        self.rmv_first_tensor = Tensor(rmv_first.astype(self.dtype))

        self.num_bboxes_test = cfg.rpn_max_num * cfg.test_batch_size

        range_max = np.arange(self.num_bboxes_test).astype(np.int32)
        self.range_max = Tensor(range_max)
        self.delta = 0.0001  # Avoid to produce 0

    def construct(self, featuremap, bbox_targets, labels, mask):
        x = self.flatten(featuremap)

        x = self.relu(self.shared_fc_0(x))
        x = self.relu(self.shared_fc_1(x))

        x_cls = self.cls_scores(x)
        x_reg = self.reg_scores(x)

        if self.training:
            bbox_weights = self.cast(self.logicaland(self.greater(labels, 0), mask), ms.int32) * labels
            labels = self.onehot(labels, self.num_classes, self.on_value, self.off_value)
            bbox_targets = self.tile(self.expandims(bbox_targets, 1), (1, self.num_classes_fronted, 1))

            loss, loss_cls, loss_reg, loss_print = self.loss(x_cls, x_reg, bbox_targets, bbox_weights, labels, mask)
            out = (loss, loss_cls, loss_reg, loss_print)
        else:
            out = (x_cls, (x_cls / self.value), x_reg, x_cls)

        return out

    def loss(self, cls_score, bbox_pred, bbox_targets, bbox_weights, labels, weights):
        """Loss method."""
        loss_print = ()
        loss_cls, _ = self.loss_cls(cls_score, labels)

        weights = self.cast(weights, self.ms_type)
        loss_cls = loss_cls * weights
        loss_cls = self.sum_loss(loss_cls, (0,)) / self.sum_loss(weights, (0,))

        bbox_weights = self.cast(self.onehot(bbox_weights, self.num_classes, self.on_value, self.off_value),
                                 self.ms_type)
        if self.without_bg_loss:
            bbox_weights = bbox_weights[:, 1:] * self.rmv_first_tensor
        else:
            bbox_weights = bbox_weights * self.rmv_first_tensor
        pos_bbox_pred = self.reshape(bbox_pred, (self.num_bboxes, -1, 4))
        loss_reg = self.loss_bbox(pos_bbox_pred, bbox_targets)
        loss_reg = self.sum_loss(loss_reg, (2,))
        loss_reg = loss_reg * bbox_weights
        if self.without_bg_loss:
            loss_reg = loss_reg / (self.sum_loss(weights, (0,)) + self.delta)
        else:
            loss_reg = loss_reg / (self.sum_loss(weights, (0,)))
        loss_reg = self.sum_loss(loss_reg, (0, 1))

        loss = self.rcnn_loss_cls_weight * loss_cls + self.rcnn_loss_reg_weight * loss_reg
        loss_print += (loss_cls, loss_reg)

        return loss, loss_cls, loss_reg, loss_print
