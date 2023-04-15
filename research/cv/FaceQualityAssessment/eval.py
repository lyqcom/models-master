# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Face Quality Assessment eval."""
import os
import time
import warnings
import numpy as np
import cv2

import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.ops import operations as P
from mindspore import context

from src.face_qa import FaceQABackbone

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num

warnings.filterwarnings('ignore')


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=1)

def get_md_output(out):
    '''get md output'''
    out_eul = out[0].asnumpy().astype(np.float32)[0]
    heatmap = out[1].asnumpy().astype(np.float32)[0]
    eulers = out_eul * 90

    kps_score_sum = 0
    kp_scores = list()
    kp_coord_ori = list()

    for i, _ in enumerate(heatmap):
        map_1 = heatmap[i].reshape(1, 48*48)
        map_1 = softmax(map_1)

        kp_coor = map_1.argmax()
        max_response = map_1.max()
        kp_scores.append(max_response)
        kps_score_sum += min(max_response, 0.25)
        kp_coor = int((kp_coor % 48) * 2.0), int((kp_coor / 48) * 2.0)
        kp_coord_ori.append(kp_coor)

    return kp_scores, kps_score_sum, kp_coord_ori, eulers, 1


def read_gt(txt_path, x_length, y_length):
    '''read gt'''
    txt_line = open(txt_path).readline()
    eulers_txt = txt_line.strip().split(" ")[:3]
    kp_list = [[-1, -1], [-1, -1], [-1, -1], [-1, -1], [-1, -1]]
    box_cur = txt_line.strip().split(" ")[3:]
    bndbox = []
    for index in range(len(box_cur) // 2):
        bndbox.append([box_cur[index * 2], box_cur[index * 2 + 1]])
    kp_id = -1
    for box in bndbox:
        kp_id = kp_id + 1
        x_coord = float(box[0])
        y_coord = float(box[1])
        if x_coord < 0 or y_coord < 0:
            continue

        kp_list[kp_id][0] = int(float(x_coord) / x_length * 96)

        kp_list[kp_id][1] = int(float(y_coord) / y_length * 96)

    return eulers_txt, kp_list


def read_img(img_path):
    img_ori = cv2.imread(img_path)
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    img = img.transpose(2, 0, 1)
    img = np.array([img]).astype(np.float32)/255.
    img = Tensor(img)
    return img, img_ori


blur_soft = nn.Softmax(0)
kps_soft = nn.Softmax(-1)
reshape = P.Reshape()
argmax = P.ArgMaxWithValue()


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


def eval_func(network):
    '''run eval'''
    path = config.eval_dir
    kp_error_all = [[], [], [], [], []]
    eulers_error_all = [[], [], []]
    kp_ipn = []

    file_list = os.listdir(path)
    for file_name in file_list:
        if file_name.endswith('jpg'):
            img_path = os.path.join(path, file_name)
            img, img_ori = read_img(img_path)

            txt_path = img_path.replace('jpg', 'txt')

            if os.path.exists(txt_path):
                euler_kps_do = True
                x_length = img_ori.shape[1]
                y_length = img_ori.shape[0]
                eulers_gt, kp_list = read_gt(txt_path, x_length, y_length)
            else:
                euler_kps_do = False
                continue

            out = network(img)

            _, _, kp_coord_ori, eulers_ori, _ = get_md_output(out)

            if euler_kps_do:
                eulgt = list(eulers_gt)
                for euler_id, _ in enumerate(eulers_ori):
                    eulori = eulers_ori[euler_id]
                    eulers_error_all[euler_id].append(abs(eulori-float(eulgt[euler_id])))

                eye01 = kp_list[0]
                eye02 = kp_list[1]
                eye_dis = 1
                cur_flag = True
                if eye01[0] < 0 or eye01[1] < 0 or eye02[0] < 0 or eye02[1] < 0:
                    cur_flag = False
                else:
                    eye_dis = np.sqrt(np.square(abs(eye01[0]-eye02[0]))+np.square(abs(eye01[1]-eye02[1])))
                cur_error_list = []
                for i in range(5):
                    kp_coord_gt = kp_list[i]
                    kp_coord_model = kp_coord_ori[i]
                    if kp_coord_gt[0] != -1:
                        dis = np.sqrt(np.square(
                            kp_coord_gt[0] - kp_coord_model[0]) + np.square(kp_coord_gt[1] - kp_coord_model[1]))
                        kp_error_all[i].append(dis)
                        cur_error_list.append(dis)
                if cur_flag:
                    kp_ipn.append(sum(cur_error_list)/len(cur_error_list)/eye_dis)

    kp_ave_error = []
    for kps, _ in enumerate(kp_error_all):
        kp_ave_error.append("%.3f" % (sum(kp_error_all[kps])/len(kp_error_all[kps])))

    euler_ave_error = []
    elur_mae = []
    for eulers, _ in enumerate(eulers_error_all):
        euler_ave_error.append("%.3f" % (sum(eulers_error_all[eulers])/len(eulers_error_all[eulers])))
        elur_mae.append((sum(eulers_error_all[eulers])/len(eulers_error_all[eulers])))

    ipn = sum(kp_ipn)/len(kp_ipn)*100
    mae = sum(elur_mae)/len(elur_mae)
    return kp_ave_error, euler_ave_error, ipn, mae


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    '''run eval'''
    print('----eval----begin----')
    ckpt_files = os.listdir(config.ckpt_dir)
    network = FaceQABackbone()

    best_ipn = 100
    for ckpt_file in ckpt_files:
        if not ckpt_file.endswith(".ckpt"):
            continue
        ckpt_path = os.path.join(config.ckpt_dir, ckpt_file)
        param_dict = load_checkpoint(ckpt_path)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        err1, err2, ipn, mae = eval_func(network)
        if ipn < best_ipn:
            best_ipn = ipn
            print("current best result as follows, ckpt path {}", ckpt_path, flush=True)
            print(r'5 keypoints average err:', err1, flush=True)
            print(r'3 eulers average err:', err2, flush=True)
            print('IPN of 5 keypoints:', ipn, flush=True)
            print('MAE of elur:', mae, flush=True)

    print('----eval----end----')


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, device_target=config.device_target, save_graphs=False)
    if config.device_target == 'Ascend':
        devid = int(os.getenv('DEVICE_ID'))
        context.set_context(device_id=devid)
    run_eval()
