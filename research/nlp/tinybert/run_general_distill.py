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

"""general distill script"""

import os
import time
import datetime
import mindspore.communication.management as D
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore.train.model import Model
from mindspore.train.callback import TimeMonitor
from mindspore.context import ParallelMode
from mindspore.nn.optim import AdamWeightDecay
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore import log as logger
from mindspore.common import set_seed
from src.dataset import create_tinybert_dataset, DataType
from src.utils import LossCallBack, ModelSaveCkpt, BertLearningRate
from src.model_utils.config import config as args_opt, common_cfg, bert_teacher_net_cfg, bert_student_net_cfg
from src.tinybert_for_gd_td import BertTrainWithLossScaleCell, BertNetworkWithLoss_gd, BertTrainCell
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num


def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, args_opt.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("Unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("Unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("Cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if args_opt.modelarts_dataset_unzip_name:
        zip_file_1 = os.path.join(args_opt.data_path, args_opt.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(args_opt.data_path)

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
    _file_dir = os.path.dirname(os.path.abspath(__file__))
    args_opt.device_id = get_device_id()
    args_opt.device_num = get_device_num()
    args_opt.data_dir = os.path.join(args_opt.data_path, args_opt.data_dir)
    args_opt.schema_dir = os.path.join(args_opt.data_path, args_opt.schema_dir)
    args_opt.save_ckpt_path = os.path.join(args_opt.output_path, args_opt.save_ckpt_path)
    args_opt.load_teacher_ckpt_path = os.path.join(_file_dir, args_opt.load_teacher_ckpt_path)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_general_distill():
    """
    run general distill
    """
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target,
                        reserve_class_name_in_scope=False)
    if args_opt.device_target == "Ascend":
        context.set_context(device_id=args_opt.device_id)

    save_ckpt_dir = os.path.join(args_opt.save_ckpt_path,
                                 datetime.datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S'))

    if args_opt.distribute == "true":
        if args_opt.device_target == 'Ascend':
            D.init()
            device_num = args_opt.device_num
            rank = args_opt.device_id % device_num
        else:
            D.init()
            device_num = D.get_group_size()
            rank = D.get_rank()
        save_ckpt_dir = save_ckpt_dir + '_ckpt_' + str(rank)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=device_num)
    else:
        rank = 0
        device_num = 1

    if not os.path.exists(save_ckpt_dir):
        os.makedirs(save_ckpt_dir)

    enable_loss_scale = True
    if args_opt.device_target == "GPU":
        context.set_context(enable_graph_kernel=True)
        if bert_student_net_cfg.compute_type != mstype.float32:
            logger.warning('Compute about the student only support float32 temporarily, run with float32.')
            bert_student_net_cfg.compute_type = mstype.float32
        # Backward of the network are calculated using fp32,
        # and the loss scale is not necessary
        enable_loss_scale = False

    if args_opt.device_target == "CPU":
        logger.warning('CPU only support float32 temporarily, run with float32.')
        bert_teacher_net_cfg.dtype = mstype.float32
        bert_teacher_net_cfg.compute_type = mstype.float32
        bert_student_net_cfg.dtype = mstype.float32
        bert_student_net_cfg.compute_type = mstype.float32
        enable_loss_scale = False

    netwithloss = BertNetworkWithLoss_gd(teacher_config=bert_teacher_net_cfg,
                                         teacher_ckpt=args_opt.load_teacher_ckpt_path,
                                         student_config=bert_student_net_cfg,
                                         is_training=True, use_one_hot_embeddings=False)

    if args_opt.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif args_opt.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        raise Exception("dataset format is not supported yet")
    dataset = create_tinybert_dataset('gd', common_cfg.batch_size, device_num, rank,
                                      args_opt.do_shuffle, args_opt.data_dir, args_opt.schema_dir,
                                      data_type=dataset_type)
    dataset_size = dataset.get_dataset_size()
    print('dataset size: ', dataset_size)
    print("dataset repeatcount: ", dataset.get_repeat_count())
    if args_opt.enable_data_sink == "true":
        repeat_count = args_opt.epoch_size * dataset_size // args_opt.data_sink_steps
        time_monitor_steps = args_opt.data_sink_steps
    else:
        repeat_count = args_opt.epoch_size
        time_monitor_steps = dataset_size

    lr_schedule = BertLearningRate(learning_rate=common_cfg.AdamWeightDecay.learning_rate,
                                   end_learning_rate=common_cfg.AdamWeightDecay.end_learning_rate,
                                   warmup_steps=int(dataset_size * args_opt.epoch_size / 10),
                                   decay_steps=int(dataset_size * args_opt.epoch_size),
                                   power=common_cfg.AdamWeightDecay.power)
    params = netwithloss.trainable_params()
    decay_params = list(filter(common_cfg.AdamWeightDecay.decay_filter, params))
    other_params = list(filter(lambda x: not common_cfg.AdamWeightDecay.decay_filter(x), params))
    group_params = [{'params': decay_params, 'weight_decay': common_cfg.AdamWeightDecay.weight_decay},
                    {'params': other_params, 'weight_decay': 0.0},
                    {'order_params': params}]

    optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=common_cfg.AdamWeightDecay.eps)

    callback = [TimeMonitor(time_monitor_steps), LossCallBack(), ModelSaveCkpt(netwithloss.bert,
                                                                               args_opt.save_ckpt_step,
                                                                               args_opt.max_ckpt_num,
                                                                               save_ckpt_dir)]
    if enable_loss_scale:
        update_cell = DynamicLossScaleUpdateCell(loss_scale_value=common_cfg.loss_scale_value,
                                                 scale_factor=common_cfg.scale_factor,
                                                 scale_window=common_cfg.scale_window)
        netwithgrads = BertTrainWithLossScaleCell(netwithloss, optimizer=optimizer, scale_update_cell=update_cell)
    else:
        netwithgrads = BertTrainCell(netwithloss, optimizer=optimizer)
    model = Model(netwithgrads)
    model.train(repeat_count, dataset, callbacks=callback,
                dataset_sink_mode=(args_opt.enable_data_sink == "true"),
                sink_size=args_opt.data_sink_steps)


if __name__ == '__main__':
    set_seed(0)
    run_general_distill()
