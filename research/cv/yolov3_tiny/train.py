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
"""Yolo train."""
import datetime
import os
import time

import mindspore as ms
from mindspore import Tensor
from mindspore import context
from mindspore.communication.management import get_group_size
from mindspore.communication.management import get_rank
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.nn.optim.momentum import Momentum
from mindspore.profiler.profiling import Profiler
from mindspore.train.callback import CheckpointConfig
from mindspore.train.callback import ModelCheckpoint
from mindspore.train.callback import RunContext
from mindspore.train.callback import _InternalCallbackParam

from model_utils.config import config
from model_utils.device_adapter import get_device_id
from model_utils.device_adapter import get_device_num
from model_utils.moxing_adapter import moxing_wrapper
from src.initializer import default_recurisive_init
from src.initializer import load_yolo_params
from src.logger import get_logger
from src.lr_scheduler import get_lr
from src.util import AverageMeter
from src.util import get_param_groups
from src.yolo import TrainingWrapper
from src.yolo import YOLOWithLossCell
from src.yolo import YOLOv3
from src.yolo_dataset import create_yolo_dataset

ms.set_seed(1)


def set_default():
    """default setting"""
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.t_max:
        config.t_max = config.max_epoch

    config.lr_epochs = list(map(int, config.lr_epochs.split(',')))
    if config.finetune:
        config.data_root = os.path.join(config.data_dir, 'train/images')
        config.ann_file = os.path.join(config.data_dir, 'annotations_json/train.json')
    else:
        config.data_root = os.path.join(config.data_dir, 'train2017')
        config.ann_file = os.path.join(config.data_dir, 'annotations/instances_train2017.json')

    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, device_id=get_device_id())

    if config.need_profiler:
        profiler = Profiler(output_path=config.outputs_dir, is_detail=True, is_show_op_path=True)
    else:
        profiler = None

    # init distributed
    if config.is_distributed:
        if config.device_target == "Ascend":
            init()
        else:
            init("nccl")
        config.rank = get_rank()
        config.group_size = get_group_size()

    # select for master rank save ckpt or all rank save, compatible for model parallel
    config.rank_save_ckpt_flag = 0
    if config.is_save_on_master:
        if config.rank == 0:
            config.rank_save_ckpt_flag = 1
    else:
        config.rank_save_ckpt_flag = 1

    # logger
    config.outputs_dir = os.path.join(config.ckpt_path,
                                      datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))
    config.logger = get_logger(config.outputs_dir, config.rank)
    config.logger.save_args(config)

    return profiler


def convert_training_shape(args_training_shape):
    """Convert training shape"""
    training_shape = [int(args_training_shape), int(args_training_shape)]
    return training_shape


def modelarts_pre_process():
    """modelarts pre process function."""
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    config.save_checkpoint_dir = os.path.join(config.output_path, config.save_checkpoint_dir)


def setup_training():
    """Setup Mindspore context"""
    context.reset_auto_parallel_context()

    if config.is_distributed:
        parallel_mode = ParallelMode.DATA_PARALLEL
        degree = get_group_size()
    else:
        degree = 1
        parallel_mode = ParallelMode.STAND_ALONE
    context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=degree)

    if config.resume_epoch:
        config.remain_epochs = config.max_epoch - config.resume_epoch
    else:
        config.remain_epochs = config.max_epoch
        config.resume_epoch = 0

    if config.training_shape:
        config.multi_scale = [convert_training_shape(config.training_shape)]
    if config.resize_rate:
        config.resize_rate = config.resize_rate


def prepare_network():
    """Prepare Network"""
    network = YOLOv3(is_training=True)
    # default is kaiming-normal
    default_recurisive_init(network)
    if config.finetune:
        param_dict = ms.load_checkpoint(config.finetune_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('yolo_network.'):
                param_dict_new[key[13:]] = values
            else:
                param_dict_new[key] = values
        for key in list(param_dict_new.keys()):
            if 'feature_map.head1.weight' in key:
                del param_dict_new[key]
            if 'feature_map.head1.bias' in key:
                del param_dict_new[key]
            if 'feature_map.head2.weight' in key:
                del param_dict_new[key]
            if 'feature_map.head2.bias' in key:
                del param_dict_new[key]
        ms.load_param_into_net(network, param_dict_new)
    else:
        load_yolo_params(config, network)

    network = YOLOWithLossCell(network)
    return network


def prepare_dataset():
    """Prepare dataset"""
    return create_yolo_dataset(
        image_dir=config.data_root,
        anno_path=config.ann_file,
        is_training=True,
        batch_size=config.per_batch_size,
        max_epoch=config.max_epoch,
        device_num=config.group_size,
        rank=config.rank,
        config=config,
        num_parallel_workers=config.num_parallel_workers
    )


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    """Train start"""
    profiler = set_default()
    setup_training()
    loss_meter = AverageMeter('loss')
    network = prepare_network()

    ds, data_size = prepare_dataset()
    config.logger.info('Finish loading dataset')

    config.steps_per_epoch = int(data_size / config.per_batch_size / config.group_size)

    if config.ckpt_interval <= 0:
        config.ckpt_interval = config.steps_per_epoch

    lr = get_lr(config)

    opt = Momentum(
        params=get_param_groups(network),
        learning_rate=Tensor(lr),
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        loss_scale=config.loss_scale
    )

    network = TrainingWrapper(network, opt, config.loss_scale)
    network.set_train()

    if config.rank_save_ckpt_flag:
        # checkpoint save
        ckpt_max_num = 5 * config.steps_per_epoch // config.ckpt_interval
        ckpt_config = CheckpointConfig(save_checkpoint_steps=config.ckpt_interval, keep_checkpoint_max=ckpt_max_num)
        save_ckpt_path = os.path.join(config.outputs_dir, f'ckpt_{config.rank}/')
        ckpt_cb = ModelCheckpoint(config=ckpt_config, directory=save_ckpt_path, prefix='yolov3_tiny')
        cb_params = _InternalCallbackParam()
        cb_params.train_network = network
        cb_params.epoch_num = ckpt_max_num
        cb_params.cur_epoch_num = 1
        run_context = RunContext(cb_params)
        ckpt_cb.begin(run_context)

    t_end = time.time()

    data_loader = ds.create_dict_iterator(output_numpy=True, num_epochs=config.remain_epochs)

    i_start = config.resume_epoch * config.steps_per_epoch
    if config.rank_save_ckpt_flag:
        cb_params.cur_epoch_num = config.resume_epoch + 1
        # pylint: disable=protected-access
        ckpt_cb._last_triggered_step = i_start
    old_progress = i_start - 1
    for i, data in enumerate(data_loader, i_start):
        images = data["image"]
        images = Tensor.from_numpy(images)
        batch_y_true_0 = Tensor.from_numpy(data['bbox1'])
        batch_y_true_1 = Tensor.from_numpy(data['bbox2'])
        batch_gt_box0 = Tensor.from_numpy(data['gt_box1'])
        batch_gt_box1 = Tensor.from_numpy(data['gt_box2'])

        loss = network(images, batch_y_true_0, batch_y_true_1, batch_gt_box0, batch_gt_box1)
        loss_meter.update(loss.asnumpy())

        if config.rank_save_ckpt_flag:
            # ckpt progress
            cb_params.cur_step_num = i + 1  # current step number
            cb_params.batch_num = i + 2
            ckpt_cb.step_end(run_context)

        if i % config.log_interval == 0:
            time_used = time.time() - t_end
            epoch = int(i / config.steps_per_epoch)
            fps = config.per_batch_size * (i - old_progress) * config.group_size / time_used
            if config.rank == 0:
                config.logger.info(
                    'epoch[{}], iter[{}], {}, fps:{:.2f} imgs/sec, lr:{}'.format(
                        epoch, i, loss_meter, fps, lr[i - i_start]
                    )
                )
            t_end = time.time()
            loss_meter.reset()
            old_progress = i

        if (i + 1) % config.steps_per_epoch == 0 and config.rank_save_ckpt_flag:
            cb_params.cur_epoch_num += 1

        if config.need_profiler and profiler is not None:
            if i == 10:
                profiler.analyse()
                break

    config.logger.info('==========end training===============')


if __name__ == "__main__":
    run_train()
