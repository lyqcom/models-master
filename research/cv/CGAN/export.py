# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""export"""
import argparse
import numpy as np
from mindspore import Tensor
from mindspore import context
from mindspore.common import dtype as mstype
from mindspore.train.serialization import load_checkpoint, load_param_into_net, export
from src.model import Generator


def preLauch():
    """parse the console argument"""
    parser = argparse.ArgumentParser(description='MindSpore cgan training')
    parser.add_argument('--device_target', type=str, default='GPU',
                        help='device target, Ascend or GPU (Default: GPU)')
    parser.add_argument('--device_id', type=int, default=0,
                        help='device id of Ascend or GPU (Default: 0)')
    parser.add_argument('--ckpt_dir', type=str,
                        default='ckpt', help='checkpoint dir of CGAN')
    parser.add_argument('--file_format', type=str,
                        default='MINDIR', help='file_format of model')
    args = parser.parse_args()
    context.set_context(device_id=args.device_id, mode=context.GRAPH_MODE, device_target=args.device_target)
    return args

def main():
    # before training, we should set some arguments
    args = preLauch()

    # training argument
    input_dim = 100
    n_col = 20
    n_row = 10
    n_image = n_row * n_col

    # create G Cell
    netG = Generator(input_dim)

    latent_code_eval = Tensor(np.random.randn(n_image, input_dim).astype(np.float32))
    label_eval = Tensor((np.arange(n_image) % 10).astype(np.int32), mstype.int32)

    param_G = load_checkpoint(args.ckpt_dir)
    load_param_into_net(netG, param_G)
    netG.set_train(False)
    export(netG, latent_code_eval, label_eval, file_name="CGAN", file_format=args.file_format)
    print("CGAN exported")

if __name__ == '__main__':
    main()
