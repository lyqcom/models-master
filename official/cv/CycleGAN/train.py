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

"""General-purpose training script for image-to-image translation.
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').
Example:
    Train a resnet model:
        python train.py --dataroot ./data/horse2zebra --model ResNet
"""

import mindspore as ms
import mindspore.nn as nn
from mindspore.communication.management import init, get_rank, get_group_size
from src.utils.args import get_args
from src.utils.reporter import Reporter
from src.utils.tools import get_lr, ImagePool, load_ckpt
from src.dataset.cyclegan_dataset import create_dataset
from src.models.losses import DiscriminatorLoss, GeneratorLoss
from src.models.cycle_gan import get_generator, get_discriminator, Generator, TrainOneStepG, TrainOneStepD

ms.set_seed(1)

def train():
    """Train function."""
    args = get_args("train")
    if args.device_num > 1:
        ms.set_context(mode=ms.GRAPH_MODE, device_target=args.platform, save_graphs=args.save_graphs)
        init()
        ms.reset_auto_parallel_context()
        ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        args.rank = get_rank()
        args.group_size = get_group_size()
    else:
        ms.set_context(mode=ms.GRAPH_MODE, device_target=args.platform,
                       save_graphs=args.save_graphs, device_id=args.device_id)
        args.rank = 0
        args.device_num = 1

    if args.platform == "GPU":
        ms.set_context(enable_graph_kernel=True)
    if args.need_profiler:
        from mindspore.profiler.profiling import Profiler
        profiler = Profiler(output_path=args.outputs_dir, is_detail=True, is_show_op_path=True)
    ds = create_dataset(args)
    G_A = get_generator(args)
    G_B = get_generator(args)
    D_A = get_discriminator(args)
    D_B = get_discriminator(args)
    if args.load_ckpt:
        load_ckpt(args, G_A, G_B, D_A, D_B)
    imgae_pool_A = ImagePool(args.pool_size)
    imgae_pool_B = ImagePool(args.pool_size)
    generator = Generator(G_A, G_B, args.lambda_idt > 0)

    loss_D = DiscriminatorLoss(args, D_A, D_B)
    loss_G = GeneratorLoss(args, generator, D_A, D_B)
    optimizer_G = nn.Adam(generator.trainable_params(), get_lr(args), beta1=args.beta1)
    optimizer_D = nn.Adam(loss_D.trainable_params(), get_lr(args), beta1=args.beta1)

    net_G = TrainOneStepG(loss_G, generator, optimizer_G)
    net_D = TrainOneStepD(loss_D, optimizer_D)

    data_loader = ds.create_dict_iterator()
    if args.rank == 0:
        reporter = Reporter(args)
        reporter.info('==========start training===============')
    for _ in range(args.max_epoch):
        if args.rank == 0:
            reporter.epoch_start()
        for data in data_loader:
            img_A = data["image_A"]
            img_B = data["image_B"]
            res_G = net_G(img_A, img_B)
            fake_A = res_G[0]
            fake_B = res_G[1]
            res_D = net_D(img_A, img_B, imgae_pool_A.query(fake_A), imgae_pool_B.query(fake_B))
            if args.rank == 0:
                reporter.step_end(res_G, res_D)
                reporter.visualizer(img_A, img_B, fake_A, fake_B)
        if args.rank == 0:
            reporter.epoch_end(net_G)
        if args.need_profiler:
            profiler.analyse()
            break
    if args.rank == 0:
        reporter.info('==========end training===============')


if __name__ == "__main__":
    train()
