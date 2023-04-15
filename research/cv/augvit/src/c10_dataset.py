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
"""
create train or eval dataset.
"""
import os
import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.transforms as data_trans
import mindspore.dataset.vision as vision

def create_dataset(dataset_path, do_train, config, platform, repeat_num=1, batch_size=1):
    """
    create a train or eval dataset

    Args:
        dataset_path(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1
        batch_size(int): the batch size of dataset. Default: 32

    Returns:
        dataset
    """
    if platform == "Ascend":
        rank_size = int(os.getenv("RANK_SIZE"))
        rank_id = int(os.getenv("RANK_ID"))
        if rank_size == 1:
            ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True)
        else:
            ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                   num_shards=rank_size, shard_id=rank_id)
    elif platform == "GPU":
        if do_train:
            from mindspore.communication.management import get_rank, get_group_size
            ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                   num_shards=get_group_size(), shard_id=get_rank())
        else:
            ds = de.Cifar10Dataset(dataset_path, num_parallel_workers=8, shuffle=False)
    else:
        raise ValueError("Unsupported platform.")

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # define map operations
    random_crop_op = vision.RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = vision.RandomHorizontalFlip()
    resize_op = vision.Resize((resize_height, resize_width))  # interpolation default BILINEAR
    rescale_op = vision.Rescale(1.0 / 255.0, 0.0)
    normalize_op = vision.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    changeswap_op = vision.HWC2CHW()
    type_cast_op = data_trans.TypeCast(mstype.int32)

    c_trans = []
    if do_train:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op, changeswap_op]

    ds = ds.map(input_columns="image", operations=c_trans)
    ds = ds.map(input_columns="label", operations=type_cast_op)

    # apply shuffle operations
    ds = ds.shuffle(buffer_size=buffer_size)

    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    ds = ds.repeat(repeat_num)
    return ds
