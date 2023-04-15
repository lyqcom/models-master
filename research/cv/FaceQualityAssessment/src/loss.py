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
"""Face Quality Assessment loss."""
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore.nn.loss.loss import LossBase
from mindspore import Tensor

eps = 1e-24


class log_softmax(nn.Cell):
    ''' replacement of P.LogSoftmax() that supports x.shape=3. '''
    def __init__(self):
        super(log_softmax, self).__init__()
        self.lsm = P.LogSoftmax()
        self.concat = P.Concat(1)
        self.reshape = P.Reshape()

    def construct(self, x):
        dim1 = x.shape[1]
        result = []
        for i in range(dim1):
            lsm = self.lsm(x[:, i, :])
            lsm = self.reshape(lsm, (F.shape(lsm)[0], 1, F.shape(lsm)[1]))
            result = lsm if i == 0 else self.concat((result, lsm))
        return result


class CEWithIgnoreIndex3D(LossBase):
    '''CEWithIgnoreIndex3D'''
    def __init__(self):
        super(CEWithIgnoreIndex3D, self).__init__()
        self.exp = P.Exp()
        self.sum = P.ReduceSum()
        self.reshape = P.Reshape()
        self.log = P.Log()
        self.cast = P.Cast()
        self.eps_const = Tensor(eps, dtype=mstype.float32)
        self.ones = P.OnesLike()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.relu = P.ReLU()
        self.maximum = P.Maximum()
        self.resum = P.ReduceSum(keep_dims=False)
        self.logsoftmax = log_softmax()

    def construct(self, logit, label):
        '''construct'''
        mask = self.reshape(label, (F.shape(label)[0], F.shape(label)[1], 1))
        mask = self.cast(mask, mstype.float32)
        mask = mask + F.scalar_to_tensor(0.00001)
        mask = self.relu(mask) / (mask)
        logit = logit * mask

        softmax_result = self.logsoftmax(logit)
        one_hot_label = self.onehot(
            self.cast(label, mstype.int32), F.shape(logit)[2], self.on_value, self.off_value)
        loss = (softmax_result * self.cast(one_hot_label, mstype.float32) * self.cast(F.scalar_to_tensor(-1),
                                                                                      mstype.float32))

        loss = self.sum(loss, -1)
        loss = self.sum(loss, -1)
        loss = self.sum(loss, 0)

        return loss


class CriterionsFaceQA(nn.Cell):
    '''CriterionsFaceQA'''
    def __init__(self):
        super(CriterionsFaceQA, self).__init__()
        self.gatherv2 = P.Gather()
        self.squeeze = P.Squeeze(axis=1)
        self.shape = P.Shape()
        self.reshape = P.Reshape()

        self.euler_label_list = Tensor([0, 1, 2], dtype=mstype.int32)
        self.mse_loss = nn.MSELoss(reduction='sum')

        self.kp_label_list = Tensor([3, 4, 5, 6, 7], dtype=mstype.int32)
        self.kps_loss = CEWithIgnoreIndex3D()

    def construct(self, x1, x2, label):
        '''construct'''
        # euler
        euler_label = self.gatherv2(label, self.euler_label_list, 1)
        loss_euler = self.mse_loss(x1, euler_label)

        # key points
        b, _, _, _ = self.shape(x2)
        x2 = self.reshape(x2, (b, 5, 48 * 48))

        kps_label = self.gatherv2(label, self.kp_label_list, 1)
        loss_kps = self.kps_loss(x2, kps_label)

        loss_tot = (loss_kps + loss_euler) / b
        return loss_tot
