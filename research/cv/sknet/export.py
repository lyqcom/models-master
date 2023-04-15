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
##############export checkpoint file into air and onnx models#################
python export.py
"""
import argparse

import numpy as np
from mindspore import (Tensor, context, export, load_checkpoint,
                       load_param_into_net)

from src.config import config1 as config
from src.sknet50 import sknet50 as sknet

parser = argparse.ArgumentParser(description='sknet export')
parser.add_argument("--device_id", type=int, default=1, help="Device id")
parser.add_argument("--batch_size", type=int, default=32, help="batch size")
parser.add_argument("--ckpt_file", type=str, default="/path/to/sknet-90_195.ckpt", help="Checkpoint file path.")
parser.add_argument("--file_name", type=str, default="sknet_export", help="output file name.")
parser.add_argument('--width', type=int, default=224, help='input width')
parser.add_argument('--height', type=int, default=224, help='input height')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
parser.add_argument("--device_target", type=str, default="Ascend",
                    choices=["Ascend", "GPU"], help="device target(default: Ascend)")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
context.set_context(device_id=args.device_id)

if __name__ == '__main__':
    net = sknet(config.class_num)
    param_dict = load_checkpoint(args.ckpt_file)
    load_param_into_net(net, param_dict)

    input_arr = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export(net, input_arr, file_name=args.file_name, file_format=args.file_format)
