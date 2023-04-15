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
######################## train mcnn example ########################
train mcnn and get network model files(.ckpt) :
python eval.py
"""

import os
import argparse
import ast
from src.dataset import create_dataset
from src.mcnn import MCNN
from src.data_loader_3channel import ImageDataLoader_3channel
from src.config import crowd_cfg as cfg
from mindspore import context
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint
import numpy as np

local_path = cfg['local_path']
local_gt_path = cfg['local_gt_path']
local_ckpt_url = cfg['local_ckpt_url']
ckptpath = cfg['ckptpath']

parser = argparse.ArgumentParser(description='MindSpore MCNN Example')
parser.add_argument('--run_offline', type=ast.literal_eval,
                    default=True, help='run in offline is False or True')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--ckpt_path', type=str, default="./train_output", help='Location of ckpt.')
parser.add_argument('--data_url', default=None, help='Location of data.')
parser.add_argument('--train_url', default=None, help='Location of training outputs.')
parser.add_argument('--val_path',
                    default='../MCNN/data/original/shanghaitech/part_A_final/test_data/images',
                    help='Location of data.')
parser.add_argument('--val_gt_path',
                    default='../MCNN/data/original/shanghaitech/part_A_final/test_data/ground_truth_csv',
                    help='Location of data.')
args = parser.parse_args()
set_seed(64678)


if __name__ == "__main__":

    device_target = args.device_target

    if device_target == "Ascend":
        device_id = int(os.getenv("DEVICE_ID"))
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        context.set_context(save_graphs=False)
        context.set_context(device_id=device_id)
        if args.run_offline:
            local_path = args.val_path
            local_gt_path = args.val_gt_path
            local_ckpt_url = args.ckpt_path
        else:
            import moxing as mox

            mox.file.copy_parallel(src_url=args.val_path, dst_url=local_path)
            mox.file.copy_parallel(src_url=args.val_gt_path, dst_url=local_gt_path)
            mox.file.copy_parallel(src_url=ckptpath, dst_url=local_ckpt_url)

        data_loader_val = ImageDataLoader_3channel(local_path, local_gt_path, shuffle=False, gt_downsample=True,
                                                   pre_load=True)
        ds_val = create_dataset(data_loader_val, target=args.device_target, train=False)

    elif device_target == "GPU":
        local_path = args.val_path
        local_gt_path = args.val_gt_path
        local_ckpt_url = args.ckpt_path
        context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
        context.set_context(save_graphs=False)
        context.set_context(device_id=0)
        data_loader_val = ImageDataLoader_3channel(local_path, local_gt_path, shuffle=False, gt_downsample=True,
                                                   pre_load=True)
        ds_val = create_dataset(data_loader_val, target="GPU", train=False)

    else:
        raise ValueError("Unsupported platform.")
    ds_val = ds_val.batch(1)
    network = MCNN()

    model_name = os.path.join(local_ckpt_url, 'best.ckpt')
    print(model_name)
    mae = 0.0
    mse = 0.0
    load_checkpoint(model_name, net=network)
    network.set_train(False)
    for sample in ds_val.create_dict_iterator():
        im_data = sample['data']
        gt_data = sample['gt_density']
        density_map = network(im_data)
        gt_count = np.sum(gt_data.asnumpy())
        et_count = np.sum(density_map.asnumpy())
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
    mae = mae / ds_val.get_dataset_size()
    mse = np.sqrt(mse / ds_val.get_dataset_size())
    print('MAE:', mae, '  MSE:', mse)
