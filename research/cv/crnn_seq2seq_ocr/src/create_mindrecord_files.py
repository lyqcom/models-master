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
"""Create FSNS MindRecord files."""

import codecs
import logging
import os
import numpy as np

from mindspore.mindrecord import FileWriter
from model_utils.config import config

def initialize_vocabulary(vocabulary_path):
    """
    initialize vocabulary from file.
    assume the vocabulary is stored one-item-per-line
    """
    characters_class = 9999

    if os.path.exists(vocabulary_path):
        rev_vocab = []
        with codecs.open(vocabulary_path, 'r', encoding='utf-8') as voc_file:
            rev_vocab = [line.strip() for line in voc_file]

        vocab = {x: y for (y, x) in enumerate(rev_vocab)}

        reserved_char_size = characters_class - len(rev_vocab)
        if reserved_char_size < 0:
            raise ValueError("Number of characters in vocabulary is equal or larger than config.characters_class")

        for _ in range(reserved_char_size):
            rev_vocab.append('')

        # put space at the last position
        vocab[' '] = len(rev_vocab)
        rev_vocab.append(' ')
        logging.info("Initializing vocabulary ends: %s", vocabulary_path)
        return vocab, rev_vocab

    raise ValueError("Initializing vocabulary ends: %s" % vocabulary_path)

def serialize_annotation(img_path, lex, vocab):

    go_id = config.characters_dictionary.go_id
    eos_id = config.characters_dictionary.eos_id

    word = [go_id]
    for special_label in config.labels_not_use:
        if lex == special_label:
            if config.print_no_train_label:
                print("label in for image: %s is special label, related label is: %s, skip ..." % (img_path, lex))
            return None

    for c in lex:
        if c not in vocab:
            return None

        c_idx = vocab.get(c)
        word.append(c_idx)

    word.append(eos_id)
    word = np.array(word, dtype=np.int32)
    return word

def create_fsns_label(image_dir, anno_file_dirs):
    """Get image path and annotation."""

    if not os.path.isdir(image_dir):
        raise ValueError(f'Cannot find {image_dir} dataset path.')

    image_files_dict = {}
    image_anno_dict = {}
    images = []
    img_id = 0

    for anno_file_dir in anno_file_dirs:

        anno_file = open(anno_file_dir, 'r').readlines()

        for line in anno_file:

            file_name = line.split('\t')[0]
            labels = line.split('\t')[1].split('\n')[0]
            image_path = os.path.join(image_dir, file_name)

            if not os.path.isfile(image_path):
                print(f'Cannot find image {image_path} according to annotations.')
                continue

            if labels:
                images.append(img_id)
                image_files_dict[img_id] = image_path
                image_anno_dict[img_id] = labels
                img_id += 1

    return images, image_files_dict, image_anno_dict


def fsns_train_data_to_mindrecord(mindrecord_dir, prefix="data_ocr.mindrecord", file_num=8):

    anno_file_dirs = [config.annotation_file]
    images, image_path_dict, image_anno_dict = create_fsns_label(image_dir=config.data_root,
                                                                 anno_file_dirs=anno_file_dirs)

    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.dirname(current_file_dir) + "/" + config.vocab_path
    vocab, _ = initialize_vocabulary(vocab_path)

    data_schema = {"image": {"type": "bytes"},
                   "label": {"type": "int32", "shape": [-1]},
                   "decoder_input": {"type": "int32", "shape": [-1]},
                   "decoder_mask": {"type": "int32", "shape": [-1]},
                   "decoder_target": {"type": "int32", "shape": [-1]},
                   "annotation": {"type": "string"}}

    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writer = FileWriter(mindrecord_path, file_num)
    writer.add_schema(data_schema, "ocr")

    for img_id in images:

        image_path = image_path_dict[img_id]
        annotation = image_anno_dict[img_id]

        label_max_len = config.max_length
        text_max_len = config.max_length - 2

        if len(annotation) > text_max_len:
            continue
        label = serialize_annotation(image_path, annotation, vocab)

        if label is None:
            continue

        label_len = len(label)
        decoder_input_len = label_max_len

        if label_len <= decoder_input_len:
            label = np.concatenate((label, np.zeros(decoder_input_len - label_len, dtype=np.int32)))
            one_mask_len = label_len - config.go_shift
            target_weight = np.concatenate((np.ones(one_mask_len, dtype=np.float32),
                                            np.zeros(decoder_input_len - one_mask_len, dtype=np.float32)))
        else:
            continue

        decoder_input = (np.array(label).T).astype(np.int32)
        target_weight = (np.array(target_weight).T).astype(np.int32)

        if len(decoder_input) != len(target_weight):
            continue

        target = [decoder_input[i + 1] for i in range(len(decoder_input) - 1)]
        target = (np.array(target)).astype(np.int32)


        with open(image_path, 'rb') as f:
            img = f.read()

        row = {"image": img,
               "label": label,
               "decoder_input": decoder_input,
               "decoder_mask": target_weight,
               "decoder_target": target,
               "annotation": str(annotation)}

        writer.write_raw_data([row])
    writer.commit()


def fsns_val_data_to_mindrecord(mindrecord_dir, prefix="data_ocr.mindrecord", file_num=8):

    anno_file_dirs = [config.val_annotation_file]
    images, image_path_dict, image_anno_dict = create_fsns_label(image_dir=config.val_data_root,
                                                                 anno_file_dirs=anno_file_dirs)

    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    vocab_path = os.path.dirname(current_file_dir) + "/" + config.vocab_path
    vocab, _ = initialize_vocabulary(vocab_path)

    data_schema = {"image": {"type": "bytes"},
                   "decoder_input": {"type": "int32", "shape": [-1]},
                   "decoder_target": {"type": "int32", "shape": [-1]},
                   "annotation": {"type": "string"}}

    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writer = FileWriter(mindrecord_path, file_num)
    writer.add_schema(data_schema, "ocr")

    for img_id in images:

        image_path = image_path_dict[img_id]
        annotation = image_anno_dict[img_id]

        label_max_len = config.max_length
        text_max_len = config.max_length - 2

        if len(annotation) > text_max_len:
            continue
        label = serialize_annotation(image_path, annotation, vocab)

        if label is None:
            continue

        label_len = len(label)
        decoder_input_len = label_max_len

        if label_len <= decoder_input_len:
            label = np.concatenate((label, np.zeros(decoder_input_len - label_len, dtype=np.int32)))
        else:
            continue

        decoder_input = (np.array(label).T).astype(np.int32)

        target = [decoder_input[i + 1] for i in range(len(decoder_input) - 1)]
        target = (np.array(target)).astype(np.int32)


        with open(image_path, 'rb') as f:
            img = f.read()

        row = {"image": img,
               "decoder_input": decoder_input,
               "decoder_target": target,
               "annotation": str(annotation)}

        writer.write_raw_data([row])
    writer.commit()

def create_mindrecord(dataset="fsns", prefix="fsns.mindrecord", is_training=True):
    print("Start creating dataset!")
    if is_training:
        mindrecord_dir = os.path.join(config.mindrecord_dir, "train")
        mindrecord_files = [os.path.join(mindrecord_dir, prefix + "0")]

        if not os.path.exists(mindrecord_files[0]):
            if not os.path.isdir(mindrecord_dir):
                os.makedirs(mindrecord_dir)
            if dataset == "fsns":
                if os.path.isdir(config.data_root):
                    print("Create FSNS Mindrecord files for train pipeline.")
                    fsns_train_data_to_mindrecord(mindrecord_dir=mindrecord_dir, prefix=prefix, file_num=8)
                    print("Create FSNS Mindrecord files for train pipeline Done, at {}".format(mindrecord_dir))
                else:
                    print("{} not exits!".format(config.data_root))
            else:
                print("{} dataset is not defined!".format(dataset))

    if not is_training:
        mindrecord_dir = os.path.join(config.mindrecord_dir, "test")
        mindrecord_files = [os.path.join(mindrecord_dir, prefix + "0")]

        if not os.path.exists(mindrecord_files[0]):
            if not os.path.isdir(mindrecord_dir):
                os.makedirs(mindrecord_dir)
            if dataset == "fsns":
                if os.path.isdir(config.val_data_root):
                    print("Create FSNS Mindrecord files for val pipeline.")
                    fsns_val_data_to_mindrecord(mindrecord_dir=mindrecord_dir, prefix=prefix)
                    print("Create FSNS Mindrecord files for val pipeline Done, at {}".format(mindrecord_dir))
                else:
                    print("{} not exits!".format(config.val_data_root))
            else:
                print("{} dataset is not defined!".format(dataset))

    return mindrecord_files


if __name__ == '__main__':
    create_mindrecord(is_training=True)
    create_mindrecord(is_training=False)
    print("END")
