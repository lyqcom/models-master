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
"""CPTN network definition."""

import numpy as np
from mindspore import context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from src.CTPN.rpn import RPN
from src.CTPN.anchor_generator import AnchorGenerator
from src.CTPN.proposal_generator import Proposal
from src.CTPN.vgg16 import VGG16FeatureExtraction
from src.weight_init import lstm_default_state

if context.get_context("device_target") == "Ascend":
    mtype = mstype.float16
    nptype = np.float16
else:
    mtype = mstype.float32
    nptype = np.float32

class BiLSTM(nn.Cell):
    """
     Define a BiLSTM network which contains two LSTM layers

     Args:
        config(EasyDict): config for ctpn network
        batch_size(int): batch size of input data, only support 1
    """
    def __init__(self, config, batch_size):
        super(BiLSTM, self).__init__()
        self.batch_size = batch_size
        self.batch_size = self.batch_size * config.rnn_batch_size
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.num_step = config.num_step
        self.reshape = P.Reshape()
        self.cast = P.Cast()
        k = (1 / self.hidden_size) ** 0.5
        self.rnn1 = P.DynamicRNN(forget_bias=0.0)
        self.rnn_bw = P.DynamicRNN(forget_bias=0.0)
        self.w1 = Parameter(np.random.uniform(-k, k, \
            (self.input_size + self.hidden_size, 4 * self.hidden_size)).astype(np.float32), name="w1")
        self.w1_bw = Parameter(np.random.uniform(-k, k, \
            (self.input_size + self.hidden_size, 4 * self.hidden_size)).astype(np.float32), name="w1_bw")

        self.b1 = Parameter(np.random.uniform(-k, k, (4 * self.hidden_size)).astype(np.float32), name="b1")
        self.b1_bw = Parameter(np.random.uniform(-k, k, (4 * self.hidden_size)).astype(np.float32), name="b1_bw")

        self.h1 = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))
        self.h1_bw = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))

        self.c1 = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))
        self.c1_bw = Tensor(np.zeros(shape=(1, self.batch_size, self.hidden_size)).astype(np.float32))
        self.reverse_seq = P.ReverseV2(axis=[0])
        self.concat = P.Concat()
        self.transpose = P.Transpose()
        self.concat1 = P.Concat(axis=2)
        self.dropout = nn.Dropout(p=0.3)
        self.use_dropout = config.use_dropout
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
    def construct(self, x):
        if self.use_dropout:
            x = self.dropout(x)
        x = self.cast(x, mtype)
        bw_x = self.reverse_seq(x)
        y1, _, _, _, _, _, _, _ = self.rnn1(x, self.w1, self.b1, None, self.h1, self.c1)
        y1_bw, _, _, _, _, _, _, _ = self.rnn_bw(bw_x, self.w1_bw, self.b1_bw, None, self.h1_bw, self.c1_bw)
        y1_bw = self.reverse_seq(y1_bw)
        output = self.concat1((y1, y1_bw))
        return output

class CTPN(nn.Cell):
    """
     Define CTPN network

     Args:
        config(EasyDict): config for ctpn network
        batch_size(int): batch size of input data, only support 1
        is_training(bool): whether training, default is True
     """
    def __init__(self, config, batch_size, is_training=True):
        super(CTPN, self).__init__()
        self.config = config
        self.batch_size = batch_size
        self.num_step = config.num_step
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.vgg16_feature_extractor = VGG16FeatureExtraction().to_float(mtype)
        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=0, pad_mode='same').to_float(mtype)
        self.rnn = BiLSTM(self.config, batch_size=self.batch_size).to_float(mtype)
        self.rnn2 = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            bidirectional=True).to_float(mtype)
        self.h, self.c = lstm_default_state(self.batch_size * config.rnn_batch_size,
                                            self.hidden_size, bidirectional=True)
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.cast = P.Cast()
        self.is_training = is_training
        self.device_target = context.get_context("device_target")

        # rpn block
        self.rpn_with_loss = RPN(config,
                                 self.batch_size,
                                 config.rpn_in_channels,
                                 config.rpn_feat_channels,
                                 config.num_anchors,
                                 config.rpn_cls_out_channels)
        self.anchor_generator = AnchorGenerator(config)
        self.featmap_size = config.feature_shapes
        self.anchor_list = self.get_anchors(self.featmap_size)
        self.proposal_generator_test = Proposal(config,
                                                self.batch_size,
                                                config.activate_num_classes,
                                                config.use_sigmoid_cls)
        self.proposal_generator_test.set_train_local(config, False)

    def construct(self, img_data, gt_bboxes, gt_labels, gt_valids, img_metas=None):
        x = self.vgg16_feature_extractor(img_data)
        x = self.conv(x)
        x = self.cast(x, mtype)
        x = self.transpose(x, (0, 2, 1, 3))
        x = self.reshape(x, (-1, self.input_size, self.num_step))
        x = self.transpose(x, (2, 0, 1))
        if self.device_target == "Ascend":
            x = self.rnn(x)
        else:
            x, _ = self.rnn2(x, (self.h, self.c))
        rpn_loss, cls_score, bbox_pred, rpn_cls_loss, rpn_reg_loss = self.rpn_with_loss(x,
                                                                                        img_metas,
                                                                                        self.anchor_list,
                                                                                        gt_bboxes,
                                                                                        gt_labels,
                                                                                        gt_valids)
        if self.training:
            return rpn_loss, cls_score, bbox_pred, rpn_cls_loss, rpn_reg_loss
        proposal, proposal_mask = self.proposal_generator_test(cls_score, bbox_pred, self.anchor_list)
        return proposal, proposal_mask

    def get_anchors(self, featmap_size):
        anchors = self.anchor_generator.grid_anchors(featmap_size)
        return Tensor(anchors, mtype)


class CTPN_Infer(nn.Cell):
    """
     Define CTPN_Infer network

     Args:
        config(EasyDict): config for ctpn network
        batch_size(int): batch size of input data, only support 1
     """
    def __init__(self, config, batch_size):
        super(CTPN_Infer, self).__init__()
        self.network = CTPN(config, batch_size=batch_size, is_training=False)
        self.network.set_train(False)

    def construct(self, img_data):
        output = self.network(img_data, None, None, None, None)
        return output
