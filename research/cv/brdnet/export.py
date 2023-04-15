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
##############export checkpoint file into air, onnx, mindir models#################
python export.py
"""
import argparse
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, load_checkpoint, load_param_into_net, export

from src.models import BRDNet
from src.models_onnx import BRDNet_onnx

## Params
parser = argparse.ArgumentParser()

parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--channel', default=3, type=int
                    , help='image channel, 3 for color, 1 for gray')
parser.add_argument("--image_height", type=int, default=500, help="Image height.")
parser.add_argument("--image_width", type=int, default=500, help="Image width.")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="brdnet", help="output file name.")
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument('--device_target', type=str, default='Ascend'
                    , help='device where the code will be implemented. (Default: Ascend)')
parser.add_argument("--device_id", type=int, default=0, help="Device id")

args_opt = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                    device_id=args_opt.device_id)

if __name__ == '__main__':
    if args_opt.file_format == "ONNX":
        net = BRDNet_onnx(args_opt.channel)
    else:
        net = BRDNet(args_opt.channel)

    param_dict = load_checkpoint(args_opt.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([args_opt.batch_size, args_opt.channel, \
                                args_opt.image_height, args_opt.image_width]), ms.float32)
    export(net, input_arr, file_name=args_opt.file_name, file_format=args_opt.file_format)
