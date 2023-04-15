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
"""main"""
import argparse
import os
import time
import numpy as np
import cv2
from api.infer import SdkApi
from config import config as cfg


def parser_args():
    """parser_args"""
    parser = argparse.ArgumentParser(description="vehiclenet inference")

    parser.add_argument("--img_path",
                        type=str,
                        required=False,
                        default="../data/input/",
                        help="image directory.")
    parser.add_argument(
        "--pipeline_path",
        type=str,
        required=False,
        default="./config/vehiclenet.pipeline",
        help="image file path. The default is '/infer/sdk/config/vehiclenet.pipeline'. ")
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
        default="dvpp",
        help=
        "rgb: high-precision, dvpp: high performance. The default is 'dvpp'.")
    parser.add_argument(
        "--infer_mode",
        type=str,
        required=False,
        default="infer",
        help=
        "infer:only infer, eval: accuracy evaluation. The default is 'infer'.")
    parser.add_argument(
        "--infer_result_dir",
        type=str,
        required=False,
        default="./infer_result",
        help=
        "cache dir of inference result. The default is './infer_result'."
    )
    arg = parser.parse_args()
    return arg


def process_img(img_file):
    img = np.fromfile(img_file, dtype=np.float32).reshape(1, 384, 384, 3)
    img = img.reshape(3, 384, 384)
    return img


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):
    """resize a rectangular image to a padded rectangular"""
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def image_inference(pipeline_path, stream_name, img_dir, result_dir):
    """image_inference"""
    sdk_api = SdkApi(pipeline_path)
    if not sdk_api.init():
        exit(-1)
    print(stream_name)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    img_data_plugin_id = 0

    print("\nBegin to inference for {}.\n".format(img_dir))
    file_list = os.listdir(img_dir)
    total_len = len(file_list)
    for img_id, file_name in enumerate(file_list):
        file_path = os.path.join(img_dir, file_name)
        save_path = os.path.join(result_dir, "{}_0.bin".format(os.path.splitext(file_name)[0]))
        print(save_path)
        img_np = process_img(file_path)
        img_shape = img_np.shape
        sdk_api.send_img_input(stream_name,
                               img_data_plugin_id, "appsrc0",
                               img_np.tobytes(), img_shape)
        start_time = time.time()
        result = sdk_api.get_result(stream_name)
        end_time = time.time() - start_time
        with open(save_path, "wb") as fp:
            fp.write(result)
            print(
                f"End-2end inference, file_name: {file_path},"
                f"{img_id + 1}/{total_len}, elapsed_time: {end_time}.\n"
            )


if __name__ == "__main__":
    args = parser_args()
    image_inference(args.pipeline_path, cfg.STREAM_NAME.encode("utf-8"), args.img_path,
                    args.infer_result_dir)
