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
"""Train api."""
import os
import ast
import argparse
import numpy as np

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.nn import Momentum
from mindspore.nn.optim import Lamb
from mindspore.train.model import Model
from mindspore.train.loss_scale_manager import DynamicLossScaleManager
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor
from mindspore.train.callback import LossMonitor
from mindspore import context, Parameter
from mindspore.context import ParallelMode
from mindspore.communication import management as MultiAscend
from mindspore.train.serialization import load_checkpoint
from mindspore.common import set_seed
from mindspore.train.serialization import export

from config.config import Seq2seqConfig
from src.seq2seq_model.seq2seq import Seq2seqModel
from src.dataset import load_dataset
from src.seq2seq_model.seq2seq_for_train import Seq2seqNetworkWithLoss, Seq2seqTrainOneStepWithLossScaleCell
from src.utils import LossCallBack
from src.utils import one_weight, weight_variable
from src.utils.lr_scheduler import square_root_schedule, polynomial_decay_scheduler, Warmup_MultiStepLR_scheduler
from src.utils.optimizer import Adam
from src.utils import zero_weight
from src.utils.load_weights import load_infer_weights

os.system('pip install subword_nmt')
os.system('pip install sacremoses')

parser = argparse.ArgumentParser(description='Seq2seq train entry point.')

parser.add_argument("--is_modelarts", type=ast.literal_eval, default=True, help="model config json file path.")
parser.add_argument("--data_url", type=str, default=None, help="pre-train dataset address.")
parser.add_argument('--train_url', type=str, default=None, help='Location of training outputs.')
parser.add_argument("--config", type=str, required=True, help="model config json file path.")
parser.add_argument("--pre_train_dataset", type=str, required=True, help="pre-train dataset address.")
args = parser.parse_args()
if args.is_modelarts:
    import moxing as mox
context.set_context(
    mode=context.GRAPH_MODE,
    save_graphs=False,
    device_target="Ascend",
    reserve_class_name_in_scope=True)


def get_config(config):
    config = Seq2seqConfig.from_json_file(config)
    config.compute_type = mstype.float16
    config.dtype = mstype.float32
    return config


def _train(model, config: Seq2seqConfig,
           pre_training_dataset=None, fine_tune_dataset=None, test_dataset=None,
           callbacks: list = None):
    """
    Train model.

    Args:
        model (Model): MindSpore model instance.
        config (seq2seqConfig): Config of mass model.
        pre_training_dataset (Dataset): Pre-training dataset.
        fine_tune_dataset (Dataset): Fine-tune dataset.
        test_dataset (Dataset): Test dataset.
        callbacks (list): A list of callbacks.
    """
    callbacks = callbacks if callbacks else []

    if pre_training_dataset is not None:
        print(" | Start pre-training job.")
        epoch_size = pre_training_dataset.get_repeat_count()
        print("epoch size ", epoch_size)
        if os.getenv("RANK_SIZE") is not None and int(os.getenv("RANK_SIZE")) > 1:
            print(f" | Rank {MultiAscend.get_rank()} Call model train.")
        model.train(config.epochs, pre_training_dataset,
                    callbacks=callbacks, dataset_sink_mode=config.dataset_sink_mode)

    if fine_tune_dataset is not None:
        print(" | Start fine-tuning job.")
        epoch_size = fine_tune_dataset.get_repeat_count()

        model.train(config.epochs, fine_tune_dataset,
                    callbacks=callbacks, dataset_sink_mode=config.dataset_sink_mode)


def _load_checkpoint_to_net(config, network):
    """load parameters to network from checkpoint."""
    if config.existed_ckpt:
        if config.existed_ckpt.endswith(".npz"):
            weights = np.load(config.existed_ckpt)
        else:
            weights = load_checkpoint(config.existed_ckpt)
        for param in network.trainable_params():
            weights_name = param.name
            if weights_name not in weights:
                raise ValueError(f"Param {weights_name} is not found in ckpt file.")

            if isinstance(weights[weights_name], Parameter):
                param.set_data(weights[weights_name].data)
            elif isinstance(weights[weights_name], Tensor):
                param.set_data(Tensor(weights[weights_name].asnumpy(), config.dtype))
            elif isinstance(weights[weights_name], np.ndarray):
                param.set_data(Tensor(weights[weights_name], config.dtype))
            else:
                param.set_data(weights[weights_name])
    else:
        for param in network.trainable_params():
            name = param.name
            value = param.data
            if isinstance(value, Tensor):
                if name.endswith(".gamma"):
                    param.set_data(one_weight(value.asnumpy().shape))
                elif name.endswith(".beta") or name.endswith(".bias"):
                    # param.set_data(zero_weight(value.asnumpy().shape))
                    if param.data.dtype == "Float32":
                        param.set_data((weight_variable(value.asnumpy().shape).astype(np.float32)))
                    elif param.data.dtype == "Float16":
                        param.set_data((weight_variable(value.asnumpy().shape).astype(np.float16)))
                else:
                    if param.data.dtype == "Float32":
                        param.set_data(Tensor(weight_variable(value.asnumpy().shape).astype(np.float32)))
                    elif param.data.dtype == "Float16":
                        param.set_data(Tensor(weight_variable(value.asnumpy().shape).astype(np.float16)))


def _get_lr(config, update_steps):
    """generate learning rate."""
    if config.lr_scheduler == "isr":
        lr = Tensor(square_root_schedule(lr=config.lr,
                                         update_num=update_steps,
                                         decay_start_step=config.decay_start_step,
                                         warmup_steps=config.warmup_steps,
                                         min_lr=config.min_lr), dtype=mstype.float32)
    elif config.lr_scheduler == "poly":
        lr = Tensor(polynomial_decay_scheduler(lr=config.lr,
                                               min_lr=config.min_lr,
                                               decay_steps=config.decay_steps,
                                               total_update_num=update_steps,
                                               warmup_steps=config.warmup_steps,
                                               power=config.lr_scheduler_power), dtype=mstype.float32)
    elif config.lr_scheduler == "WarmupMultiStepLR":
        lr = Tensor(Warmup_MultiStepLR_scheduler(base_lr=config.lr,
                                                 total_update_num=update_steps,
                                                 warmup_steps=config.warmup_steps,
                                                 remain_steps=config.warmup_lr_remain_steps,
                                                 decay_interval=config.warmup_lr_decay_interval,
                                                 decay_steps=config.decay_steps,
                                                 decay_factor=config.lr_scheduler_power), dtype=mstype.float32)
    else:
        lr = config.lr
    return lr


def _get_optimizer(config, network, lr):
    """get gnmt optimizer, support Adam, Lamb, Momentum."""
    if config.optimizer.lower() == "adam":
        optimizer = Adam(network.trainable_params(), lr, beta1=0.9, beta2=0.98)
    elif config.optimizer.lower() == "lamb":
        optimizer = Lamb(network.trainable_params(), learning_rate=lr,
                         eps=1e-6)
    elif config.optimizer.lower() == "momentum":
        optimizer = Momentum(network.trainable_params(), lr, momentum=0.9)
    else:
        raise ValueError(f"optimizer only support `adam` and `momentum` now.")

    return optimizer


def _build_training_pipeline(config: Seq2seqConfig,
                             pre_training_dataset=None,
                             fine_tune_dataset=None,
                             test_dataset=None):
    """
    Build training pipeline.

    Args:
        config (seq2seqConfig): Config of seq2seq model.
        pre_training_dataset (Dataset): Pre-training dataset.
        fine_tune_dataset (Dataset): Fine-tune dataset.
        test_dataset (Dataset): Test dataset.
    """

    net_with_loss = Seq2seqNetworkWithLoss(config, is_training=True, use_one_hot_embeddings=True)
    net_with_loss.init_parameters_data()
    _load_checkpoint_to_net(config, net_with_loss)

    dataset = pre_training_dataset if pre_training_dataset is not None \
        else fine_tune_dataset

    if dataset is None:
        raise ValueError("pre-training dataset or fine-tuning dataset must be provided one.")

    update_steps = config.epochs * dataset.get_dataset_size()

    lr = _get_lr(config, update_steps)
    optimizer = _get_optimizer(config, net_with_loss, lr)

    # Dynamic loss scale.
    scale_manager = DynamicLossScaleManager(init_loss_scale=config.init_loss_scale,
                                            scale_factor=config.loss_scale_factor,
                                            scale_window=config.scale_window)
    net_with_grads = Seq2seqTrainOneStepWithLossScaleCell(
        network=net_with_loss, optimizer=optimizer,
        scale_update_cell=scale_manager.get_update_cell()
    )
    net_with_grads.set_train(True)
    model = Model(net_with_grads)
    loss_monitor = LossCallBack(config)
    dataset_size = dataset.get_dataset_size()
    time_cb = TimeMonitor(data_size=dataset_size)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset.get_dataset_size(),
                                   keep_checkpoint_max=config.keep_ckpt_max)

    rank_size = os.getenv('RANK_SIZE')
    callbacks = [time_cb, loss_monitor]
    callbacks.append(LossMonitor())

    if rank_size is not None and int(rank_size) > 1 and MultiAscend.get_rank() % 8 == 0:
        ckpt_callback = ModelCheckpoint(
            prefix=config.ckpt_prefix,
            directory=os.path.join(config.ckpt_path, 'ckpt_{}'.format(os.getenv('DEVICE_ID'))),
            config=ckpt_config)
        callbacks.append(ckpt_callback)

    if rank_size is None or int(rank_size) == 1:
        ckpt_callback = ModelCheckpoint(
            prefix=config.ckpt_prefix,
            directory=os.path.join(config.ckpt_path, 'ckpt_{}'.format(os.getenv('DEVICE_ID'))),
            config=ckpt_config)
        callbacks.append(ckpt_callback)

    print(f" | ALL SET, PREPARE TO TRAIN.")
    _train(model=model, config=config,
           pre_training_dataset=pre_training_dataset,
           fine_tune_dataset=fine_tune_dataset,
           test_dataset=test_dataset,
           callbacks=callbacks)

    # frozen into air
    config_export = config
    last_ckpt = config.ckpt_prefix + '-' + str(config.epochs) + '_' + str(dataset.get_dataset_size()) + '.ckpt'
    config_export.existed_ckpt = os.path.join(config.ckpt_path, \
                                'ckpt_{}'.format(os.getenv('DEVICE_ID')), \
                                last_ckpt)
    tfm_model = Seq2seqModel(
        config=config_export,
        is_training=False,
        use_one_hot_embeddings=False)
    params = tfm_model.trainable_params()
    weights = load_infer_weights(config_export)
    for param in params:
        value = param.data
        weights_name = param.name
        if weights_name not in weights:
            raise ValueError(f"{weights_name} is not found in weights.")
        if isinstance(value, Tensor):
            if weights_name in weights:
                assert weights_name in weights
                if isinstance(weights[weights_name], Parameter):
                    if param.data.dtype == "Float32":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float32))
                    elif param.data.dtype == "Float16":
                        param.set_data(Tensor(weights[weights_name].data.asnumpy(), mstype.float16))

                elif isinstance(weights[weights_name], Tensor):
                    param.set_data(Tensor(weights[weights_name].asnumpy(), config.dtype))
                elif isinstance(weights[weights_name], np.ndarray):
                    param.set_data(Tensor(weights[weights_name], config.dtype))
                else:
                    param.set_data(weights[weights_name])
            else:
                print("weight not found in checkpoint: " + weights_name)
                param.set_data(zero_weight(value.asnumpy().shape))

    source_ids = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))
    source_mask = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))
    target_ids = Tensor(np.ones((config.batch_size, config.seq_length)).astype(np.int32))
    if args.is_modelarts:
        export(tfm_model, source_ids, source_mask, target_ids, \
                file_name="/cache/train_output/"+last_ckpt, \
                file_format="AIR")
    else:
        export(tfm_model, source_ids, source_mask, target_ids, \
                file_name=last_ckpt, \
                file_format="AIR")
    print("export success!")


def _setup_parallel_env():
    context.reset_auto_parallel_context()
    MultiAscend.init()
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.DATA_PARALLEL,
        device_num=MultiAscend.get_group_size(),
        gradients_mean=True
    )


def train_parallel(config: Seq2seqConfig):
    """
    Train model with multi ascend chips.

    Args:
        config (seq2seqConfig): Config for Seq2seq model.
    """
    _setup_parallel_env()
    print(f" | Starting training on {os.getenv('RANK_SIZE', None)} devices.")

    pre_train_dataset = load_dataset(
        data_files=config.pre_train_dataset,
        batch_size=config.batch_size,
        sink_mode=config.dataset_sink_mode,
        rank_size=MultiAscend.get_group_size(),
        rank_id=MultiAscend.get_rank()
    ) if config.pre_train_dataset else None
    fine_tune_dataset = load_dataset(
        data_files=config.fine_tune_dataset,
        batch_size=config.batch_size,
        sink_mode=config.dataset_sink_mode,
        rank_size=MultiAscend.get_group_size(),
        rank_id=MultiAscend.get_rank()
    ) if config.fine_tune_dataset else None
    test_dataset = load_dataset(
        data_files=config.test_dataset,
        batch_size=config.batch_size,
        sink_mode=config.dataset_sink_mode,
        rank_size=MultiAscend.get_group_size(),
        rank_id=MultiAscend.get_rank()
    ) if config.test_dataset else None

    _build_training_pipeline(config=config,
                             pre_training_dataset=pre_train_dataset,
                             fine_tune_dataset=fine_tune_dataset,
                             test_dataset=test_dataset)


def train_single(config: Seq2seqConfig):
    """
    Train model on single device.

    Args:
        config (seq2seqConfig): Config for seq2seq model.
    """
    print(" | Starting training on single device.")

    pre_train_dataset = load_dataset(data_files=config.pre_train_dataset,
                                     batch_size=config.batch_size,
                                     sink_mode=config.dataset_sink_mode) if config.pre_train_dataset else None
    fine_tune_dataset = load_dataset(data_files=config.fine_tune_dataset,
                                     batch_size=config.batch_size,
                                     sink_mode=config.dataset_sink_mode) if config.fine_tune_dataset else None
    test_dataset = load_dataset(data_files=config.test_dataset,
                                batch_size=config.batch_size,
                                sink_mode=config.dataset_sink_mode) if config.test_dataset else None

    _build_training_pipeline(config=config,
                             pre_training_dataset=pre_train_dataset,
                             fine_tune_dataset=fine_tune_dataset,
                             test_dataset=test_dataset)


def _check_args(config):
    if not os.path.exists(config):
        raise FileNotFoundError("`config` is not existed.")
    if not isinstance(config, str):
        raise ValueError("`config` must be type of str.")


if __name__ == '__main__':
    _rank_size = os.getenv('RANK_SIZE')

    _check_args(args.config)
    _config = get_config(args.config)
    if args.is_modelarts:
        mox.file.copy_parallel(src_url=args.data_url, dst_url='/cache/dataset_menu/')
        _config.pre_train_dataset = '/cache/dataset_menu/train.tok.clean.bpe.32000.en.mindrecord'
        _config.ckpt_path = '/cache/train_output/'
    else:
        _config.pre_train_dataset = args.pre_train_dataset

    set_seed(_config.random_seed)

    if _rank_size is not None and int(_rank_size) > 1:
        train_parallel(_config)
    else:
        train_single(_config)

    if args.is_modelarts:
        mox.file.copy_parallel(src_url='/cache/train_output/', dst_url=args.train_url)
