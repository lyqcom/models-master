# Copyright 2022 Huawei Technologies Co., Ltd
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
"""data process file"""
import os
import json
import pickle
import joblib
from tqdm import tqdm
import numpy as np

from src.reading_comprehension.wiki_link_db import WikiLinkDB
from src.reading_comprehension.dataset import SquadV2Processor, SquadV1Processor
from src.reading_comprehension.feature import convert_examples_to_features
from src.utils.utils import create_dir_not_exist

from mindspore.mindrecord import FileWriter


# load data examples


def load_examples(args, evaluate=False):
    """load examples"""
    if args.with_negative:
        processor = SquadV2Processor()
    else:
        processor = SquadV1Processor()

    if evaluate:
        examples = processor.get_dev_examples(args.data_dir)
    else:
        examples = processor.get_train_examples(args.data_dir)

    bert_model_name = args.model_config.bert_model_name

    segment_b_id = 1
    add_extra_sep_token = False
    if "roberta" in bert_model_name:
        segment_b_id = 0
        add_extra_sep_token = True
    print("Creating features from the dataset...")
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=args.tokenizer,
        entity_vocab=args.entity_vocab,
        wiki_link_db=args.wiki_link_db,
        model_redirect_mappings=args.model_redirect_mappings,
        link_redirect_mappings=args.link_redirect_mappings,
        max_seq_length=args.max_seq_length,
        max_mention_length=args.max_mention_length,
        doc_stride=args.doc_stride,
        max_query_length=args.max_query_length,
        min_mention_link_prob=args.min_mention_link_prob,
        segment_b_id=segment_b_id,
        add_extra_sep_token=add_extra_sep_token,
        is_training=not evaluate,
    )
    a = []
    for feature in features:
        data = {}
        data['unique_id'] = feature.unique_id
        data['example_index'] = feature.example_index
        data['tokens'] = feature.tokens
        data['mentions'] = feature.mentions
        data['token_is_max_context'] = feature.token_is_max_context
        data['word_ids'] = feature.word_ids
        data['word_segment_ids'] = feature.word_segment_ids
        data['word_attention_mask'] = feature.word_attention_mask
        data['entity_ids'] = feature.entity_ids
        data['entity_position_ids'] = feature.entity_position_ids
        data['entity_segment_ids'] = feature.entity_segment_ids
        data['entity_attention_mask'] = feature.entity_attention_mask
        data['start_positions'] = feature.start_positions
        data['end_positions'] = feature.end_positions
        a.append(data)
    b = json.dumps(a)
    f2 = open('new_json.json', 'w')
    f2.write(b)
    f2.close()
    print("over")

    def collate_fn(o):
        """collate fun"""

        def create_padded_sequence(o, attr_name, padding_value, max_len):
            """create padding"""
            value = getattr(o[1], attr_name)
            if attr_name == 'entity_position_ids':
                if len(value) > max_len:
                    return value[:max_len]
                res = value + [[padding_value] * len(value[0])] * (max_len - len(value))
                return res
            if len(value) > max_len:
                return value[:max_len]
            return value + [padding_value] * (max_len - len(value))

        ret = dict(
            word_ids=create_padded_sequence(o, "word_ids", args.tokenizer.pad_token_id, args.max_seq_length),
            word_attention_mask=create_padded_sequence(o, "word_attention_mask", 0, args.max_seq_length),
            word_segment_ids=create_padded_sequence(o, "word_segment_ids", 0, args.max_seq_length),
            entity_ids=create_padded_sequence(o, "entity_ids", 0, args.max_entity_length),
            entity_attention_mask=create_padded_sequence(o, "entity_attention_mask", 0, args.max_entity_length),
            entity_position_ids=create_padded_sequence(o, "entity_position_ids", -1, args.max_entity_length),
            entity_segment_ids=create_padded_sequence(o, "entity_segment_ids", 0, args.max_entity_length),
        )
        if args.no_entity:
            ret["entity_attention_mask"].fill_(0)

        if evaluate:
            ret["example_indices"] = o[0]
        else:
            ret["start_positions"] = o[1].start_positions[0]
            ret["end_positions"] = o[1].end_positions[0]

        return ret

    dataset = []
    for d in tqdm(list(enumerate(features))):
        dataset.append(collate_fn(d))
    if evaluate:
        return dataset, examples, features, processor
    return dataset


def save_train(args):
    """save changed train data"""
    train_data = load_examples(args, False)
    print("train")
    with open(os.path.join(args.data, 'squad_change', 'train.json'), 'w', encoding='utf-8') as f:
        for d in tqdm(train_data):
            f.write(json.dumps(d) + '\n')


# save (examples, features, processor) as a object
class Eval_obj:
    """change eval obj"""

    def __init__(self, examples, features, processor):
        """init fun"""
        self.examples = examples
        self.features = features
        self.processor = processor


def save_eval(args):
    """save changed eval data"""
    eval_data, examples, features, processor = load_examples(args, True)
    # eval_data
    print('eval_data')
    with open(os.path.join(args.data, 'squad_change', 'eval_data.json'), 'w', encoding='utf-8') as f:
        for d in tqdm(eval_data):
            f.write(json.dumps(d) + '\n')
    eval_obj = Eval_obj(examples, features, processor)
    with open(os.path.join(args.data, 'squad_change', 'eval_obj.pickle'), 'wb') as f:
        pickle.dump(eval_obj, f, pickle.HIGHEST_PROTOCOL)


def chang_train_to_mindrecord(args):
    """change train data to mindrecord"""
    print("load train data")
    dataset = []
    with open(os.path.join(args.data, 'squad_change', 'train.json'), 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            dataset.append(json.loads(line))
            line = f.readline()
    print("load complete")
    data_dir = os.path.join(args.data, "mindrecord", "train.mindrecord")
    # write mindrecord
    schema_json = {"word_ids": {"type": "int32", "shape": [-1]},
                   "word_segment_ids": {"type": "int32", "shape": [-1]},
                   "word_attention_mask": {"type": "int32", "shape": [-1]},
                   "entity_ids": {"type": "int32", "shape": [-1]},
                   "entity_position_ids": {"type": "int32", "shape": [args.max_entity_length, 30]},
                   "entity_segment_ids": {"type": "int32", "shape": [-1]},
                   "entity_attention_mask": {"type": "int32", "shape": [-1]},
                   "start_positions": {"type": "int32", "shape": [-1]},
                   "end_positions": {"type": "int32", "shape": [-1]},
                   }

    def get_imdb_data(data):
        """get a iter data"""
        data_list = []
        print('now change，please wait....')
        for each in data:
            data_json = {"word_ids": np.array(each['word_ids'], dtype=np.int32),
                         "word_segment_ids": np.array(each['word_segment_ids'], dtype=np.int32),
                         "word_attention_mask": np.array(each['word_attention_mask'], dtype=np.int32),
                         "entity_ids": np.array(each['entity_ids'], dtype=np.int32),
                         "entity_position_ids": np.array(each['entity_position_ids'], dtype=np.int32),
                         "entity_segment_ids": np.array(each['entity_segment_ids'], dtype=np.int32),
                         "entity_attention_mask": np.array(each['entity_attention_mask'], dtype=np.int32),
                         "start_positions": np.array(each['start_positions'], dtype=np.int32),
                         "end_positions": np.array(each['end_positions'], dtype=np.int32)
                         }

            data_list.append(data_json)
        return data_list

    writer = FileWriter(data_dir, shard_num=4)
    data = get_imdb_data(dataset)
    writer.add_schema(schema_json, "nlp_schema")
    writer.write_raw_data(data)
    writer.commit()
    print("change train data complete")


def chang_eval_to_mindrecord(args):
    """change eval data to mindrecord"""
    print("load eval data")
    dataset = []
    with open(os.path.join(args.data, 'squad_change', 'eval_data.json'), 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            dataset.append(json.loads(line))
            line = f.readline()
    print("load complete")
    data_dir = os.path.join(args.data, "mindrecord", "eval_data.mindrecord")
    # write mindrecord
    schema_json = {"word_ids": {"type": "int32", "shape": [-1]},
                   "word_segment_ids": {"type": "int32", "shape": [-1]},
                   "word_attention_mask": {"type": "int32", "shape": [-1]},
                   "entity_ids": {"type": "int32", "shape": [-1]},
                   "entity_position_ids": {"type": "int32", "shape": [args.max_entity_length, 30]},
                   "entity_segment_ids": {"type": "int32", "shape": [-1]},
                   "entity_attention_mask": {"type": "int32", "shape": [-1]},
                   "example_indices": {"type": "int32", "shape": [-1]}
                   }

    def get_imdb_data(data):
        """get a iter data"""
        data_list = []
        print('now change，please wait....')
        for each in data:
            data_json = {"word_ids": np.array(each['word_ids'], dtype=np.int32),
                         "word_segment_ids": np.array(each['word_segment_ids'], dtype=np.int32),
                         "word_attention_mask": np.array(each['word_attention_mask'], dtype=np.int32),
                         "entity_ids": np.array(each['entity_ids'], dtype=np.int32),
                         "entity_position_ids": np.array(each['entity_position_ids'], dtype=np.int32),
                         "entity_segment_ids": np.array(each['entity_segment_ids'], dtype=np.int32),
                         "entity_attention_mask": np.array(each['entity_attention_mask'], dtype=np.int32),
                         "example_indices": np.array(each['example_indices'], dtype=np.int32)
                         }

            data_list.append(data_json)
        return data_list

    writer = FileWriter(data_dir, shard_num=4)
    data = get_imdb_data(dataset)
    writer.add_schema(schema_json, "nlp_schema")
    writer.write_raw_data(data)
    writer.commit()
    print("change eval data complete")


def build_data_change(args):
    """load wiki"""
    print("load wiki data")
    args.wiki_link_db = WikiLinkDB(args.wiki_link_db_file)
    print("load complete 1")
    args.model_redirect_mappings = joblib.load(args.model_redirects_file)
    print("load complete 2")
    args.link_redirect_mappings = joblib.load(args.link_redirects_file)
    print("load complete 3")
    create_dir_not_exist(os.path.join(args.data, 'squad_change'))
    create_dir_not_exist(os.path.join(args.data, 'mindrecord'))
    save_eval(args)
    save_train(args)
    chang_train_to_mindrecord(args)
    chang_eval_to_mindrecord(args)
