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
network config setting, will be used in main.py
"""

from easydict import EasyDict as edict

imagenet_cfg = edict({
    'name': 'imagenet',
    'pre_trained': False,
    'use_dataset_sink': True,
    'num_classes': 1000,
    'lr_init': 0.02,  # 1P: 0.02, 8P: 0.18
    'batch_size': 128,
    'epoch_size': 160,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'image_height': 224,
    'image_width': 224,
    'data_path': '/data/ILSVRC2012_train/',
    'val_data_path': '/data/ILSVRC2012_val/',
    'device_target': 'Ascend',
    'device_id': 0,
    'keep_checkpoint_max': 25,
    'checkpoint_path': None,
    'onnx_filename': 'resnet50-bam',
    'air_filename': 'resnet50-bam',

    # optimizer and lr related
    'lr_scheduler': 'cosine_annealing',
    'lr_epochs': [30, 60, 90, 120],
    'lr_gamma': 0.3,
    'eta_min': 0.0,
    'T_max': 150,
    'warmup_epochs': 0,

    # loss related
    'is_dynamic_loss_scale': 0,
    'loss_scale': 1024,
    'label_smooth_factor': 0.1,
    'use_label_smooth': True,
})
