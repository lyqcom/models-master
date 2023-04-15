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

# This file was copied from anomalib [openvinotoolkit] [anomalib]
"""Framework for Easily Invertible Architectures.

Module to construct invertible networks with pytorch, based on a graph
structure of operations.

Link to the original repo: https://github.com/VLL-HD/FrEIA
"""

from .framework import SequenceINN
from .modules import AllInOneBlock

__all__ = ["SequenceINN", "AllInOneBlock"]
