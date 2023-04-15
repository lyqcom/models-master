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
"""post process for 310 inference"""
import os
import json
import numpy as np
from src.model_utils.config import config
from mindspore.nn import Top1CategoricalAccuracy, Top5CategoricalAccuracy

if __name__ == '__main__':
    top1_acc = Top1CategoricalAccuracy()
    top5_acc = Top5CategoricalAccuracy()
    rst_path = config.result_path
    if config.dataset.lower() == "cifar10":
        labels = np.load(config.label_path, allow_pickle=True)
        for idx, label in enumerate(labels):
            f_name = os.path.join(rst_path, "cifar10_bs" + str(config.batch_size) + "_" + str(idx) + "_0.bin")
            pred = np.fromfile(f_name, np.float32)
            pred = pred.reshape(config.batch_size, int(pred.shape[0] / config.batch_size))
            top1_acc.update(pred, labels[idx])
            top5_acc.update(pred, labels[idx])
        print("Top1 acc: ", top1_acc.eval())
        print("Top5 acc: ", top5_acc.eval())
    else:
        file_list = os.listdir(rst_path)
        with open(config.label_path, "r") as label:
            labels = json.load(label)
        for f in file_list:
            label = f.split("_0.bin")[0] + ".JPEG"
            pred = np.fromfile(os.path.join(rst_path, f), np.float32)
            pred = pred.reshape(config.batch_size, int(pred.shape[0] / config.batch_size))
            top1_acc.update(pred, [labels[label],])
            top5_acc.update(pred, [labels[label],])
        print("Top1 acc: ", top1_acc.eval())
        print("Top5 acc: ", top5_acc.eval())
