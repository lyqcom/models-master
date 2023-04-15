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
"""generate dataset for ascend 310"""
import os
import numpy as np
from sklearn import preprocessing

from src import dataloader, config
from src.argparser import arg_parser

args = arg_parser()

cfg = config.stgcn_chebconv_45min_cfg
cfg.batch_size = 1

if __name__ == "__main__":

    zscore = preprocessing.StandardScaler()

    dataset = dataloader.create_dataset(args.data_url+args.data_path, cfg.batch_size,
                                        cfg.n_his, cfg.n_pred, zscore, True, mode=2)

    img_path = os.path.join(args.result_path, "00_data")
    os.mkdir(img_path)

    label_list = []
    # dataset is an instance of Dataset object
    iterator = dataset.create_dict_iterator(output_numpy=True)
    for i, data in enumerate(iterator):
        file_name = "STGCN_data_bs" + str(cfg.batch_size) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        data['inputs'].tofile(file_path)
        label_list.append(data['labels'])

    np.save(args.result_path + "label_ids.npy", label_list)
    print("="*20, "export bin files finished", "="*20)
