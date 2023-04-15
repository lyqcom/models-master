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
"""postprocess."""
import argparse
import os

import numpy as np
from mindspore import Tensor

from src.config import ModelConfig
from src.metrics import AUCMetric

parser = argparse.ArgumentParser(description='CTR Prediction')
parser.add_argument('--result_path', type=str, default="./result_Files", help='Dataset path')
parser.add_argument('--label_path', type=str, default="./CriteoBinary/batch_labels", help='Checkpoint path')
args = parser.parse_args()


def get_acc():
    ''' get accuracy '''
    config = ModelConfig()
    batch_size = config.batch_size
    auc_metric = AUCMetric()
    files = os.listdir(args.label_path)


    for f in files:
        rst_file = os.path.join(args.result_path, f.split('.')[0] + '_0.bin')
        label_file = os.path.join(args.label_path, f)

        logit = Tensor(np.fromfile(rst_file, np.float32).reshape(batch_size, 1))
        label = Tensor(np.fromfile(label_file, np.float32).reshape(batch_size, 1))

        res = []
        res.append(logit)
        res.append(logit)
        res.append(label)

        auc_metric.update(*res)
    auc = auc_metric.eval()
    print("auc : {}".format(auc))


if __name__ == '__main__':
    get_acc()
