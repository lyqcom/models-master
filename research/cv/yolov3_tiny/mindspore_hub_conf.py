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
"""hub config."""
from src.yolo import YOLOv3


def create_network(name, *args, **kwargs):
    """Create network"""
    if name == "yolov3_tiny":
        yolov3_net = YOLOv3(is_training=True)
        return yolov3_net
    raise NotImplementedError(f"{name} is not implemented in the repo")
