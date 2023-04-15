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
"""
Use this file for 310 inference accuracy evaluation
"""

import argparse
import os.path
from src.config import CONFIG
from src.dataset import TestingDataSet
from src.utils import compute_IoU_recall_top_n_forreg
import numpy as np
from mindspore.common import set_seed



def get_args():
    parser = argparse.ArgumentParser(description='Train CTRL')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend'],
                        help='device target, only support Ascend.')
    parser.add_argument('--device_id', type=int, default=0, help='device id of Ascend.')
    parser.add_argument('--eval_data_dir', type=str, default=None,
                        help='the directory of train data.')
    return parser.parse_args()


args = get_args()
cfg = CONFIG(data_dir=args.eval_data_dir)
set_seed(cfg.seed)

if __name__ == '__main__':

    print("Get Dataset...")
    dataset = TestingDataSet(cfg.test_feature_dir, cfg.test_csv_path, cfg.test_batch_size)
    print('Done.')

    print("Start eval...")
    IoU_thresh = [0.1, 0.3, 0.5]
    all_correct_num_10 = [0.0] * 5
    all_correct_num_5 = [0.0] * 5
    all_correct_num_1 = [0.0] * 5
    all_retrievd = 0.0

    for movie_name in dataset.movie_names:
        batch = cfg.test_batch_size

        print("Test movie: " + movie_name + "....loading movie data")
        movie_clip_featmaps, movie_clip_sentences = dataset.load_movie_slidingclip(movie_name, 16)
        print("sentences: " + str(len(movie_clip_sentences)))
        print("clips: " + str(len(movie_clip_featmaps)))
        sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
        for k in range(0, len(movie_clip_sentences), batch):
            sent_vec = [x[1] for x in movie_clip_sentences[k:k + batch]]
            length_k = len(sent_vec)
            sent_vec = np.array(sent_vec)
            if length_k < batch:
                padding = np.zeros(shape=[batch - length_k, sent_vec.shape[1]], dtype=np.float32)
                sent_vec = np.concatenate((sent_vec, padding), axis=0)

            batch = cfg.test_batch_size
            for t in range(0, len(movie_clip_featmaps), batch):
                featmap = [x[1] for x in movie_clip_featmaps[t:t + batch]]
                length_t = len(featmap)
                featmap = np.array(featmap)
                visual_clip_name = [x[0] for x in movie_clip_featmaps[t:t + batch]]
                if length_t < batch:
                    padding = np.zeros(shape=[batch - length_t, featmap.shape[1]], dtype=np.float32)
                    featmap = np.concatenate((featmap, padding), axis=0)

                start = np.array([int(x.split("_")[1]) for x in visual_clip_name])
                end = np.array([int(x.split("_")[2].split("_")[0]) for x in visual_clip_name])

                input_feat = np.concatenate((featmap, sent_vec), axis=1)
                name = f"{movie_name}_sent_{k}_clip_{t}_0.bin"
                output_path = os.path.join('./result_Files/', name)
                output_np = np.fromfile(output_path, dtype=np.float32)
                output_np = output_np.reshape(batch, batch, 3)

                sentence_image_mat[k:k + length_k, t:t + length_t] = output_np[:length_k, :length_t, 0]

                reg_end = end + output_np[:length_k, :length_t, 2]
                reg_start = start + output_np[:length_k, :length_t, 1]

                sentence_image_reg_mat[k:k + length_k, t:t + length_t, 0] = reg_start
                sentence_image_reg_mat[k:k + length_k, t:t + length_t, 1] = reg_end

        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]

        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU = IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat,
                                                             sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat,
                                                            sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat,
                                                            sentence_image_reg_mat, sclips, iclips)
            print(movie_name + " IoU=" + str(IoU) + ", R@10: " + str(correct_num_10 / len(sclips)) +
                  "; IoU=" + str(IoU) + ", R@5: " + str(correct_num_5 / len(sclips)) + "; IoU=" +
                  str(IoU) + ", R@1: " + str(correct_num_1 / len(sclips)))
            all_correct_num_10[k] += correct_num_10
            all_correct_num_5[k] += correct_num_5
            all_correct_num_1[k] += correct_num_1
        all_retrievd += len(sclips)
    for k in range(len(IoU_thresh)):
        print("IoU=" + str(IoU_thresh[k]) + ", R@10: " + str(all_correct_num_10[k] / all_retrievd) +
              "; IoU=" + str(IoU_thresh[k]) + ", R@5: " + str(all_correct_num_5[k] / all_retrievd) +
              "; IoU=" + str(IoU_thresh[k]) + ", R@1: " + str(all_correct_num_1[k] / all_retrievd))
    print('Done.')
    print("End.")
