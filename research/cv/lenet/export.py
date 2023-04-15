# Copyright 2020 Huawei Technologies Co., Ltd
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
"""export checkpoint file into air, onnx, mindir models"""

from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id
from src.model_utils.moxing_adapter import moxing_wrapper
from src.lenet import LeNet5

import numpy as np
import mindspore
from mindspore import Tensor, context, load_checkpoint, load_param_into_net, export


context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target)
if config.device_target == "Ascend":
    context.set_context(device_id=get_device_id())

def modelarts_process():
    pass

@moxing_wrapper(pre_process=modelarts_process)
def export_lenet():
    print('export with config: ', config)
    # define fusion network
    network = LeNet5(config.num_classes)
    # load network checkpoint
    param_dict = load_checkpoint(config.ckpt_file)
    load_param_into_net(network, param_dict)

    # export network
    inputs = Tensor(np.ones([config.batch_size, 1, config.image_height, config.image_width]), mindspore.float32)
    export(network, inputs, file_name=config.file_name, file_format=config.file_format)

if __name__ == '__main__':
    export_lenet()
