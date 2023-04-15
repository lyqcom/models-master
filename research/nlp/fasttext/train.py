# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
"""FastText for train"""
import os
import time
from mindspore import context
from mindspore.nn.optim import Adam
from mindspore.common import set_seed
from mindspore.train.model import Model
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.context import ParallelMode
from mindspore.train.callback import Callback, TimeMonitor
from mindspore.communication import management as MultiDevice
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.load_dataset import load_dataset
from src.lr_schedule import polynomial_decay_scheduler
from src.fasttext_train import FastTextTrainOneStepCell, FastTextNetWithLoss

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num


def get_ms_timestamp():
    t = time.time()
    return int(round(t * 1000))

set_seed(1)
time_stamp_init = False
time_stamp_first = 0
context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target=config.device_target)


class LossCallBack(Callback):
    """
    Monitor the loss in training.

    If the loss is NAN or INF terminating training.

    Note:
        If per_print_times is 0 do not print loss.

    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, per_print_times=1, rank_ids=0):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.rank_id = rank_ids
        global time_stamp_init, time_stamp_first
        if not time_stamp_init:
            time_stamp_first = get_ms_timestamp()
            time_stamp_init = True

    def step_end(self, run_context):
        """Monitor the loss in training."""
        global time_stamp_first
        time_stamp_current = get_ms_timestamp()
        cb_params = run_context.original_args()
        print("time: {}, epoch: {}, step: {}, outputs are {}".format(time_stamp_current - time_stamp_first,
                                                                     cb_params.cur_epoch_num,
                                                                     cb_params.cur_step_num,
                                                                     str(cb_params.net_outputs)))
        with open("./loss_{}.log".format(self.rank_id), "a+") as f:
            f.write("time: {}, epoch: {}, step: {}, loss: {}".format(
                time_stamp_current - time_stamp_first,
                cb_params.cur_epoch_num,
                cb_params.cur_step_num,
                str(cb_params.net_outputs.asnumpy())))
            f.write('\n')


def _build_training_pipeline(pre_dataset, run_distribute=False):
    """
    Build training pipeline

    Args:
        pre_dataset: preprocessed dataset
    """
    net_with_loss = FastTextNetWithLoss(config.vocab_size, config.embedding_dims, config.num_class)
    net_with_loss.init_parameters_data()
    if config.pretrain_ckpt_dir:
        parameter_dict = load_checkpoint(config.pretrain_ckpt_dir)
        load_param_into_net(net_with_loss, parameter_dict)
    if pre_dataset is None:
        raise ValueError("pre-process dataset must be provided")

    #get learning rate
    update_steps = config.epoch * pre_dataset.get_dataset_size()
    decay_steps = pre_dataset.get_dataset_size()
    rank_size = os.getenv("RANK_SIZE")
    if isinstance(rank_size, int):
        raise ValueError("RANK_SIZE must be integer")
    if rank_size is not None and int(rank_size) > 1:
        base_lr = config.lr
    else:
        base_lr = config.lr / 10
    print("+++++++++++Total update steps ", update_steps)
    lr = Tensor(polynomial_decay_scheduler(lr=base_lr,
                                           min_lr=config.min_lr,
                                           decay_steps=decay_steps,
                                           total_update_num=update_steps,
                                           warmup_steps=config.warmup_steps,
                                           power=config.poly_lr_scheduler_power), dtype=mstype.float32)
    optimizer = Adam(net_with_loss.trainable_params(), lr, beta1=0.9, beta2=0.999)

    net_with_grads = FastTextTrainOneStepCell(net_with_loss, optimizer=optimizer)
    net_with_grads.set_train(True)
    model = Model(net_with_grads)
    loss_monitor = LossCallBack(rank_ids=config.rank_id)
    dataset_size = pre_dataset.get_dataset_size()
    time_monitor = TimeMonitor(data_size=dataset_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=decay_steps * config.epoch,
                                   keep_checkpoint_max=config.keep_ckpt_max)
    callbacks = [time_monitor, loss_monitor]
    if not run_distribute:
        ckpt_callback = ModelCheckpoint(prefix='fasttext',
                                        directory=os.path.join(config.save_ckpt_dir,
                                                               'ckpt_{}'.format(os.getenv("DEVICE_ID"))),
                                        config=ckpt_config)
        callbacks.append(ckpt_callback)
    if run_distribute and MultiDevice.get_rank() % 8 == 0:
        ckpt_callback = ModelCheckpoint(prefix='fasttext',
                                        directory=os.path.join(config.save_ckpt_dir,
                                                               'ckpt_{}'.format(os.getenv("DEVICE_ID"))),
                                        config=ckpt_config)
        callbacks.append(ckpt_callback)
    print("Prepare to Training....")
    epoch_size = pre_dataset.get_repeat_count()
    print("Epoch size ", epoch_size)
    if run_distribute:
        print(f" | Rank {MultiDevice.get_rank()} Call model train.")
    model.train(epoch=config.epoch, train_dataset=pre_dataset, callbacks=callbacks, dataset_sink_mode=False)


def train_single(input_file_path):
    """
    Train model on single device
    Args:
        input_file_path: preprocessed dataset path
    """
    print("Staring training on single device.")
    preprocessed_data = load_dataset(dataset_path=input_file_path,
                                     batch_size=config.batch_size,
                                     epoch_count=config.epoch_count,
                                     bucket=config.buckets)
    _build_training_pipeline(preprocessed_data)


def set_parallel_env():
    context.reset_auto_parallel_context()
    MultiDevice.init()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      device_num=MultiDevice.get_group_size(),
                                      gradients_mean=True)


def train_paralle(input_file_path):
    """
    Train model on multi device
    Args:
        input_file_path: preprocessed dataset path
    """
    set_parallel_env()
    print("Starting traning on multiple devices. |~ _ ~| |~ _ ~| |~ _ ~| |~ _ ~|")
    batch_size = config.batch_size
    if config.device_target == 'GPU':
        batch_size = config.distribute_batch_size_gpu

    preprocessed_data = load_dataset(dataset_path=input_file_path,
                                     batch_size=batch_size,
                                     epoch_count=config.epoch_count,
                                     rank_size=MultiDevice.get_group_size(),
                                     rank_id=MultiDevice.get_rank(),
                                     bucket=config.buckets,
                                     shuffle=False)
    _build_training_pipeline(preprocessed_data, True)


def modelarts_pre_process():
    '''modelarts pre process function.'''
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

    config.save_ckpt_dir = os.path.join(config.output_path, config.save_ckpt_dir)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_train():
    '''run train.'''
    config.rank_id = int(os.environ.get("RANK_ID", "0"))
    if config.run_distribute:
        train_paralle(config.dataset_path)
    else:
        train_single(config.dataset_path)

if __name__ == "__main__":
    run_train()
