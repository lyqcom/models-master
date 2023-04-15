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
"""compute acc for ascend 310"""
import os
import numpy as np
from sklearn import preprocessing

from src.argparser import arg_parser
from src.config import stgcn_chebconv_45min_cfg
from src import dataloader

args = arg_parser()

cfg = stgcn_chebconv_45min_cfg
cfg.batch_size = 1

if __name__ == "__main__":

    zscore = preprocessing.StandardScaler()

    rst_path = args.result_dir
    labels = np.load(args.label_dir)

    dataset = dataloader.create_dataset(args.data_url+args.data_path, cfg.batch_size,
                                        cfg.n_his, cfg.n_pred, zscore, True, mode=2)

    mae, sum_y, mape, mse = [], [], [], []

    for i in range(len(os.listdir(rst_path))):
        file_name = os.path.join(rst_path, "STGCN_data_bs" + str(cfg.batch_size) + '_' + str(i) + '_0.bin')
        output = np.fromfile(file_name, np.float16)
        output = zscore.inverse_transform(output)
        label = zscore.inverse_transform(labels[i])

        d = np.abs(label - output)
        mae += d.tolist()
        sum_y += label.tolist()
        mape += (d / label).tolist()
        mse += (d ** 2).tolist()

    MAE = np.array(mae).mean()
    MAPE = np.array(mape).mean()
    RMSE = np.sqrt(np.array(mse).mean())

    print(f'MAE {MAE:.2f} | MAPE {MAPE*100:.2f} | RMSE {RMSE:.2f}')
