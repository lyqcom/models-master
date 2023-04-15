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
"""Object Recognition eval."""
import os
import re
import warnings
import time
import numpy as np
from PIL import Image
from tqdm import tqdm

import mindspore.dataset.vision as V
import mindspore.dataset.transforms as T
from mindspore import context, Tensor, Parameter
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.reid import SphereNet

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

warnings.filterwarnings('ignore')


def inclass_likehood(ims_info, types='cos'):
    '''Inclass likehood.'''
    obj_feas = {}
    likehoods = []
    for name, _, fea in ims_info:
        if re.split('_\\d\\d\\d\\d', name)[0] not in obj_feas:
            obj_feas[re.split('_\\d\\d\\d\\d', name)[0]] = []
        obj_feas[re.split('_\\d\\d\\d\\d', name)[0]].append(fea) # pylint: "_\d\d\d\d" -> "_\\d\\d\\d\\d"
    for _, feas in tqdm(obj_feas.items()):
        feas = np.array(feas)
        if types == 'cos':
            likehood_mat = np.dot(feas, np.transpose(feas)).tolist()
            for row in likehood_mat:
                likehoods += row
        else:
            for fea in feas.tolist():
                likehoods += np.sum(-(fea - feas) ** 2, axis=1).tolist()

    likehoods = np.array(likehoods)
    return likehoods


def btclass_likehood(ims_info, types='cos'):
    '''Btclass likehood.'''
    likehoods = []
    count = 0
    for name1, _, fea1 in tqdm(ims_info):
        count += 1
        # pylint: "_\d\d\d\d" -> "_\\d\\d\\d\\d"
        frame_id1, _ = re.split('_\\d\\d\\d\\d', name1)[0], name1.split('_')[-1]
        fea1 = np.array(fea1)
        for name2, _, fea2 in ims_info:
            # pylint: "_\d\d\d\d" -> "_\\d\\d\\d\\d"
            frame_id2, _ = re.split('_\\d\\d\\d\\d', name2)[0], name2.split('_')[-1]
            if frame_id1 == frame_id2:
                continue
            fea2 = np.array(fea2)
            if types == 'cos':
                likehoods.append(np.sum(fea1 * fea2))
            else:
                likehoods.append(np.sum(-(fea1 - fea2) ** 2))

    likehoods = np.array(likehoods)
    return likehoods


def tar_at_far(inlikehoods, btlikehoods):
    test_point = [0.5, 0.3, 0.1, 0.01, 0.001, 0.0001, 0.00001]
    tar_far = []
    for point in test_point:
        thre = btlikehoods[int(btlikehoods.size * point)]
        n_ta = np.sum(inlikehoods > thre)
        tar_far.append((point, float(n_ta) / inlikehoods.size, thre))

    return tar_far


def load_images(paths, batch_size=128):
    '''Load images.'''
    ll = []
    resize = V.Resize((96, 64))
    transform = T.Compose([
        V.ToTensor(),
        V.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], is_hwc=False)])
    for i, _ in enumerate(paths):
        im = Image.open(paths[i])
        im = resize(im)
        img = np.array(im)
        ts = transform(img)
        ll.append(ts)
        if len(ll) == batch_size:
            yield np.stack(ll, axis=0)
            ll.clear()
    if ll:
        yield np.stack(ll, axis=0)


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


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    '''run eval.'''
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)

    if config.device_target == 'Ascend':
        devid = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=devid)
    print(config)

    model_path = config.pretrained
    result_file = model_path.replace('.ckpt', '.txt')
    if os.path.exists(result_file):
        os.remove(result_file)

    with open(result_file, 'a+') as result_fw:
        result_fw.write(model_path + '\n')

        network = SphereNet(num_layers=12, feature_dim=128, shape=(96, 64))
        if os.path.isfile(model_path):
            param_dict = load_checkpoint(model_path)
            param_dict_new = {}
            for key, values in param_dict.items():
                np_value = values.asnumpy()
                np_value[np.isnan(np_value)] = 0
                values = Parameter(np_value)
                if key.startswith('moments.'):
                    continue
                elif key.startswith('model.'):
                    param_dict_new[key[6:]] = values
                else:
                    param_dict_new[key] = values
            load_param_into_net(network, param_dict_new)
            print('-----------------------load model success-----------------------')
        else:
            print('-----------------------load model failed -----------------------')

        if config.device_target == 'CPU':
            network.add_flags_recursive(fp32=True)
        else:
            network.add_flags_recursive(fp16=True)
        network.set_train(False)

        root_path = config.eval_dir
        root_file_list = os.listdir(root_path)
        ims_info = []
        for sub_path in root_file_list:
            for im_path in os.listdir(os.path.join(root_path, sub_path)):
                ims_info.append((im_path.split('.')[0], os.path.join(root_path, sub_path, im_path)))

        paths = [path for name, path in ims_info]
        names = [name for name, path in ims_info]
        print("exact feature...")

        l_t = []
        for batch in load_images(paths):
            batch = batch.astype(np.float32)
            batch = Tensor(batch)
            fea = network(batch)
            l_t.append(fea.asnumpy().astype(np.float16))
        feas = np.concatenate(l_t, axis=0)
        ims_info = list(zip(names, paths, feas.tolist()))

        print("exact inclass likehood...")
        inlikehoods = inclass_likehood(ims_info)
        inlikehoods[::-1].sort()

        print("exact btclass likehood...")
        btlikehoods = btclass_likehood(ims_info)
        btlikehoods[::-1].sort()
        tar_far = tar_at_far(inlikehoods, btlikehoods)

        for far, tar, thre in tar_far:
            print('---{}: {}@{}'.format(far, tar, thre))

        for far, tar, thre in tar_far:
            result_fw.write('{}: {}@{} \n'.format(far, tar, thre))


if __name__ == "__main__":
    run_eval()
