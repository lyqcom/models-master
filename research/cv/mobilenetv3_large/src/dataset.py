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
"""Create train or eval dataset."""
import mindspore.common.dtype as mstype
import mindspore.dataset as de
import mindspore.dataset.vision as C
import mindspore.dataset.transforms as C2
from mindspore.communication.management import get_rank, get_group_size

def create_dataset(dataset_path, do_train, config, repeat_num=1, batch_size=32, run_distribute=True):
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

    if not do_train:
        data_set = de.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)
    else:
        if run_distribute:
            rank_id = get_rank()
            rank_size = get_group_size()
            data_set = de.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True,
                                             num_shards=rank_size, shard_id=rank_id)
        else:
            data_set = de.ImageFolderDataset(dataset_path, num_parallel_workers=8, shuffle=True)

    resize_height = config.image_height
    resize_width = config.image_width
    buffer_size = 1000

    # define map operations
    decode_op = C.Decode()
    resize_crop_op = C.RandomCropDecodeResize(resize_height, scale=(0.08, 1.0), ratio=(0.75, 1.333))
    horizontal_flip_op = C.RandomHorizontalFlip(prob=0.5)

    resize_op = C.Resize(256)
    center_crop = C.CenterCrop(resize_width)
    rescale_op = C.RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)
    normalize_op = C.Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                               std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
    change_swap_op = C.HWC2CHW()

    if do_train:
        trans = [resize_crop_op, horizontal_flip_op, rescale_op, normalize_op, change_swap_op]
    else:
        trans = [decode_op, resize_op, center_crop, normalize_op, change_swap_op]

    type_cast_op = C2.TypeCast(mstype.int32)

    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)

    # apply shuffle operations
    data_set = data_set.shuffle(buffer_size=buffer_size)

    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set
