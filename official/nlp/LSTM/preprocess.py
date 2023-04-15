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
##############preprocess#################
"""
import os
import numpy as np

from src.dataset import lstm_create_dataset
from src.model_utils.config import config


if __name__ == '__main__':
    print("============== Starting Data Pre-processing ==============")
    dataset = lstm_create_dataset(config.preprocess_path, config.batch_size, training=False)
    img_path = os.path.join(config.result_path, "00_data")
    os.makedirs(img_path)
    label_list = []
    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "LSTM_data_bs" + str(config.batch_size) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        data['feature'].tofile(file_path)
        label_list.append(data['label'])

    np.save(config.result_path + "label_ids.npy", label_list)
    print("="*20, "export bin files finished", "="*20)
