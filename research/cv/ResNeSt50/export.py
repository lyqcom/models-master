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
##############export checkpoint file into air, onnx or mindir model#################
python export.py
"""
import argparse
import numpy as np

from mindspore import Tensor, export, context

from src.models.resnest import get_network

parser = argparse.ArgumentParser(description='ResNeSt200 export')
parser.add_argument("--device_id", type=int, default=0, help="Device id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--ckpt_file", type=str, required=True, help="Checkpoint file path.")
parser.add_argument("--net_name", type=str, default="resnest50", help="network name.")
parser.add_argument('--width', type=int, default=256, help='input width')
parser.add_argument('--height', type=int, default=256, help='input height')
parser.add_argument("--file_format", type=str, choices=["AIR", "ONNX", "MINDIR"], default="MINDIR", help="file format")
args = parser.parse_args()

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
context.set_context(device_id=args.device_id)

if __name__ == '__main__':

    assert args.ckpt_file is not None, "checkpoint_path is None."

    net = get_network(args.net_name, True, args.ckpt_file)

    input_arr = Tensor(np.zeros([args.batch_size, 3, args.height, args.width], np.float32))
    export(net, input_arr, file_name=args.net_name, file_format=args.file_format)
