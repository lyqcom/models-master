# Contexts

<!-- TOC -->

- [Contexts](#contexts)
- [MASS: Masked Sequence to Sequence Pre-training for Language Generation Description](#mass-masked-sequence-to-sequence-pre-training-for-language-generation-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
- [Script description](#script-description)
    - [Data Preparation](#data-preparation)
        - [Tokenization](#tokenization)
        - [Byte Pair Encoding](#byte-pair-encoding)
        - [Build Vocabulary](#build-vocabulary)
        - [Generate Dataset](#generate-dataset)
            - [News Crawl Corpus](#news-crawl-corpus)
            - [Gigaword Corpus](#gigaword-corpus)
    - [Configuration](#configuration)
    - [Training & Evaluation process](#training--evaluation-process)
    - [Weights average](#weights-average)
    - [Learning rate scheduler](#learning-rate-scheduler)
- [Environment Requirements](#environment-requirements)
    - [Platform](#platform)
    - [Requirements](#requirements)
- [Get started](#get-started)
    - [Pre-training](#pre-training)
    - [Fine-tuning](#fine-tuning)
    - [Inference](#inference)
    - [Mindir Inference Process](#mindir-inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Performance](#performance)
    - [Results](#results)
        - [Fine-Tuning on Text Summarization](#fine-tuning-on-text-summarization)
        - [Training Performance](#training-performance)
        - [Inference Performance](#inference-performance)
- [Description of random situation](#description-of-random-situation)
- [others](#others)
- [ModelZoo Homepage](#modelzoo-homepage)

<!-- /TOC -->

# MASS: Masked Sequence to Sequence Pre-training for Language Generation Description

[MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-paper-updated-002.pdf) was released by MicroSoft in June 2019.

BERT(Devlin et al., 2018) have achieved SOTA in natural language understanding area by pre-training the encoder part of Transformer(Vaswani et al., 2017) with masked rich-resource text. Likewise, GPT(Raddford et al., 2018) pre-trains the decoder part of Transformer with masked(encoder inputs are masked) rich-resource text. Both of them build a robust language model by pre-training with masked rich-resource text.

Inspired by BERT, GPT and other language models, MicroSoft addressed [MASS: Masked Sequence to Sequence Pre-training for Language Generation](https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-paper-updated-002.pdf) which combines BERT's and GPT's idea. MASS has an important parameter k, which controls the masked fragment length. BERT and GPT are specicl case when k equals to 1 and sentence length.

[Introducing MASS – A pre-training method that outperforms BERT and GPT in sequence to sequence language generation tasks](https://www.microsoft.com/en-us/research/blog/introducing-mass-a-pre-training-method-that-outperforms-bert-and-gpt-in-sequence-to-sequence-language-generation-tasks/)

[Paper](https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-paper-updated-002.pdf): Song, Kaitao, Xu Tan, Tao Qin, Jianfeng Lu and Tie-Yan Liu. “MASS: Masked Sequence to Sequence Pre-training for Language Generation.” ICML (2019).

# Model Architecture

The MASS network is implemented by Transformer, which has multi-encoder layers and multi-decoder layers.
For pre-training, we use the Adam optimizer and loss-scale to get the pre-trained model.
During fine-turning, we fine-tune this pre-trained model with different dataset according to different tasks.
During testing, we use the fine-turned model to predict the result, and adopt a beam search algorithm to
get the most possible prediction results.

# Dataset

Note that you can run the scripts based on the dataset mentioned in original paper or widely used in relevant domain/network architecture. In the following sections, we will introduce how to run the scripts using the related dataset below.

Dataset used:

- [monolingual English data from News Crawl dataset](https://www.statmt.org/wmt16/translation-task.html)(WMT 2019) for pre-training.
- [Gigaword Corpus](https://github.com/harvardnlp/sent-summary)(Graff et al., 2003) for Text Summarization.

# Features

Mass is designed to jointly pre train encoder and decoder to complete the task of language generation.
First of all, through a sequence to sequence framework, mass only predicts the blocked token, which forces the encoder to understand the meaning of the unshielded token, and encourages the decoder to extract useful information from the encoder.
Secondly, by predicting the continuous token of the decoder, the decoder can build better language modeling ability than only predicting discrete token.
Third, by further shielding the input token of the decoder which is not shielded in the encoder, the decoder is encouraged to extract more useful information from the encoder side, rather than using the rich information in the previous token.

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on yaml file.
#          Set other parameters on yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the task"task=train"
# (3) Set the code directory to "/path/mass" on the website UI interface.
# (4) Set the startup file to "train.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a.  Set "enable_modelarts=True" on yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on yaml file.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the task"task=infer" and path of vocab
# (4) Set the code directory to "/path/mass" on the website UI interface.
# (5) Set the startup file to "eval.py" on the website UI interface.
# (6) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (7) Create your job.
```

# Script description

MASS script and code structure are as follow:

```text
├── mass
  ├── README.md                              // Introduction of MASS model.
  ├── config
  │   ├──config.py                           // Configuration instance definition.
  ├── src
  │   ├── model_utils
          │   ├──config.py                   // parameter configuration
          │   ├──device_adapter.py           // device adapter
          │   ├──local_adapter.py            // local adapter
          │   ├──moxing_adapter.py           // moxing adapter
  │   ├──dataset
  │      ├──bi_data_loader.py                // Dataset loader for fine-tune or inferring.
  │      ├──mono_data_loader.py              // Dataset loader for pre-training.
  │   ├──language_model
  │      ├──noise_channel_language_model.p   // Noisy channel language model for dataset generation.
  │      ├──mass_language_model.py           // MASS language model according to MASS paper.
  │      ├──loose_masked_language_model.py   // MASS language model according to MASS released code.
  │      ├──masked_language_model.py         // Masked language model according to MASS paper.
  │   ├──transformer
  │      ├──create_attn_mask.py              // Generate mask matrix to remove padding positions.
  │      ├──transformer.py                   // Transformer model architecture.
  │      ├──encoder.py                       // Transformer encoder component.
  │      ├──decoder.py                       // Transformer decoder component.
  │      ├──self_attention.py                // Self-Attention block component.
  │      ├──multi_head_attention.py          // Multi-Head Self-Attention component.
  │      ├──embedding.py                     // Embedding component.
  │      ├──positional_embedding.py          // Positional embedding component.
  │      ├──feed_forward_network.py          // Feed forward network.
  │      ├──residual_conn.py                 // Residual block.
  │      ├──beam_search.py                   // Beam search decoder for inferring.
  │      ├──transformer_for_infer.py         // Use Transformer to infer.
  │      ├──transformer_for_train.py         // Use Transformer to train.
  │   ├──utils
  │      ├──byte_pair_encoding.py            // Apply BPE with subword-nmt.
  │      ├──dictionary.py                    // Dictionary.
  │      ├──loss_moniter.py                  // Callback of monitering loss during training step.
  │      ├──lr_scheduler.py                  // Learning rate scheduler.
  │      ├──ppl_score.py                     // Perplexity score based on N-gram.
  │      ├──rouge_score.py                   // Calculate ROUGE score.
  │      ├──load_weights.py                  // Load weights from a checkpoint or NPZ file.
  │      ├──initializer.py                   // Parameters initializer.
  ├── vocab
  │   ├──all.bpe.codes                       // BPE codes table(this file should be generated by user).
  │   ├──all_en.dict.bin                     // Learned vocabulary file(this file should be generated by user).
  ├── scripts
  │   ├──run_ascend.sh                       // Ascend train & evaluate model script.
  │   ├──run_gpu.sh                          // GPU train & evaluate model script.
  │   ├──learn_subword.sh                    // Learn BPE codes.
  │   ├──stop_training.sh                    // Stop training.
  ├── requirements.txt                       // Requirements of third party package.
  ├── train.py                               // Train API entry.
  ├── eval.py                                // Infer API entry.
  ├── default_config.yaml                    // parameter configuration
  ├── tokenize_corpus.py                     // Corpus tokenization.
  ├── apply_bpe_encoding.py                  // Applying bpe encoding.
  ├── weights_average.py                     // Average multi model checkpoints to NPZ format.
  ├── news_crawl.py                          // Create News Crawl dataset for pre-training.
  ├── gigaword.py                            // Create Gigaword Corpus.

```

## Data Preparation

The data preparation of a natural language processing task contains data cleaning, tokenization, encoding and vocabulary generation steps.

In our experiments, using [Byte Pair Encoding(BPE)](https://arxiv.org/abs/1508.07909) could reduce size of vocabulary, and relieve the OOV influence effectively.

Vocabulary could be created using `src/utils/dictionary.py` with text dictionary which is learnt from BPE.
For more detail about BPE, please refer to [Subword-nmt lib](https://www.cnpython.com/pypi/subword-nmt) or [paper](https://arxiv.org/abs/1508.07909).

In our experiments, vocabulary was learned based on 1.9M sentences from News Crawl Dataset, size of vocabulary is 45755.

Here, we have a brief introduction of data preparation scripts.

### Tokenization

Using `tokenize_corpus.py` could tokenize corpus whose text files are in format of `.txt`.

Major parameters in `tokenize_corpus.py`:

```bash
--corpus_folder:     Corpus folder path, if multi-folders are provided, use ',' split folders.
--output_folder:     Output folder path.
--tokenizer:         Tokenizer to be used, nltk or jieba, if nltk is not installed fully, use jieba instead.
--pool_size:         Processes pool size.
```

Sample code:

```bash
python tokenize_corpus.py --corpus_folder /{path}/corpus --output_folder /{path}/tokenized_corpus --tokenizer {nltk|jieba} --pool_size 16
```

If you are using `nltk`, make sure you have installed both `nltk` and `nltk data`. Otherwise you could try `jieba`.

### Byte Pair Encoding

After tokenization, BPE is applied to tokenized corpus with provided `all.bpe.codes`.

Apply BPE script can be found in `apply_bpe_encoding.py`.

Major parameters in `apply_bpe_encoding.py`:

```bash
--codes:            BPE codes file.
--src_folder:       Corpus folders.
--output_folder:    Output files folder.
--prefix:           Prefix of text file in `src_folder`.
--vocab_path:       Generated vocabulary output path.
--threshold:        Filter out words that frequency is lower than threshold.
--processes:        Size of process pool (to accelerate). Default: 2.
```

Sample code:

```bash
python apply_bpe_encoding.py --codes /{path}/all.bpe.codes \
    --src_folder /{path}/tokenized_corpus \
    --output_folder /{path}/tokenized_corpus/bpe \
    --prefix tokenized \
    --vocab_path /{path}/vocab_en.dict.bin
    --processes 32
```

### Build Vocabulary

Support that you want to create a new vocabulary, there are two options:

1. Learn BPE codes from scratch, and create vocabulary with multi vocabulary files from `subword-nmt`.
2. Create from an existing vocabulary file which lines in the format of `word frequency`.
3. *Optional*, Create a small vocabulary based on `vocab/all_en.dict.bin` with method of `shink` from `src/utils/dictionary.py`.
4. Persistent vocabulary to `vocab` folder with method `persistence()`.

Major interface of `src/utils/dictionary.py` are as follow:

1. `shrink(self, threshold=50)`: Shrink the size of vocabulary by filter out words frequency is lower than threshold. It returns a new vocabulary.
2. `load_from_text(cls, filepaths: List[str])`: Load existed text vocabulary which lines in the format of `word frequency`.  
3. `load_from_persisted_dict(cls, filepath)`: Load from a persisted binary vocabulary which was saved by calling `persistence()` method.
4. `persistence(self, path)`: Save vocabulary object to binary file.

Sample code:

```python
from src.utils import Dictionary

vocabulary = Dictionary.load_from_persisted_dict("vocab/all_en.dict.bin")
tokens = [1, 2, 3, 4, 5]
# Convert ids to symbols.
print([vocabulary[t] for t in tokens])

sentence = ["Hello", "world"]
# Convert symbols to ids.
print([vocabulary.index[s] for s in sentence])
```

For more detail, please refer to the source file.

### Generate Dataset

As mentioned above, two corpus are used in MASS mode, dataset generation scripts for them are provided.

#### News Crawl Corpus

Script can be found in `news_crawl.py`.

Major parameters in `news_crawl.py`:

```bash
Note that please provide `--existed_vocab` or `--dict_folder` at least one.
A new vocabulary would be created in `output_folder` when pass `--dict_folder`.

--src_folder:       Corpus folders.
--existed_vocab:    Optional, persisted vocabulary file.
--mask_ratio:       Ratio of mask.
--output_folder:    Output dataset files folder path.
--max_len:          Maximum sentence length. If a sentence longer than `max_len`, then drop it.
--suffix:           Optional, suffix of generated dataset files.
--processes:        Optional, size of process pool (to accelerate). Default: 2.
```

Sample code:

```bash
python news_crawl.py --src_folder /{path}/news_crawl \
    --existed_vocab /{path}/mass/vocab/all_en.dict.bin \
    --mask_ratio 0.5 \
    --output_folder /{path}/news_crawl_dataset \
    --max_len 32 \
    --processes 32
```

#### Gigaword Corpus

Script can be found in `gigaword.py`.

Major parameters in `gigaword.py`:

```bash
--train_src:        Train source file path.
--train_ref:        Train reference file path.
--test_src:         Test source file path.
--test_ref:         Test reference file path.
--existed_vocab:    Persisted vocabulary file.
--output_folder:    Output dataset files folder path.
--noise_prob:       Optional, add noise prob. Default: 0.
--max_len:          Optional, maximum sentence length. If a sentence longer than `max_len`, then drop it. Default: 64.
--format:           Optional, dataset format, "mindrecord" or "tfrecord". Default: "tfrecord".
```

Sample code:

```bash
python gigaword.py --train_src /{path}/gigaword/train_src.txt \
    --train_ref /{path}/gigaword/train_ref.txt \
    --test_src /{path}/gigaword/test_src.txt \
    --test_ref /{path}/gigaword/test_ref.txt \
    --existed_vocab /{path}/mass/vocab/all_en.dict.bin \
    --noise_prob 0.1 \
    --output_folder /{path}/gigaword_dataset \
    --max_len 64
```

## Configuration

Almost all of the options and arguments needed could be assigned conveniently, including the training platform, configurations of dataset and model, arguments of optimizer etc. Optional features such as loss scale and checkpoint are also available by setting the options correspondingly.
For more detailed information about the attributes, refer to the file `default_config.yaml`.

## Training & Evaluation process

For training a model, the shell script `run_ascend.sh` or `run_gpu.sh` is all you need. In this scripts, the environment variable is set and the training script `train.py` under `mass` is executed.
You may start a task training with single device or multiple devices by assigning the options and run the command in bash:

Ascend:

```ascend
bash run_ascend.sh [--options]
```

GPU:

```gpu
bash run_gpu.sh [--options]
```

The usage of `run_ascend.sh` is shown as below:

```text
Usage: run_ascend.sh [-h, --help] [-t, --task <CHAR>] [-n, --device_num <N>]
                     [-i, --device_id <N>] [-o, --output <FILE>] [-v, --vocab <FILE>]

options:
    -h, --help               show usage
    -t, --task               select task: CHAR, 't' for train and 'i' for inference".
    -n, --device_num         device number used for training: N, default is 1.
    -i, --device_id          device id used for training with single device: N, 0<=N<=7, default is 0.
    -j, --hccl_json          rank table file used for training with multiple devices: FILE.
    -o, --output             assign output file of inference: FILE.
    -v, --vocab              set the vocabulary.
    -m, --metric             set the metric.
```

Notes: Be sure to assign the hccl_json file while running a distributed-training.

The usage of `run_gpu.sh` is shown as below:

```text
Usage: run_gpu.sh [-h, --help] [-t, --task <CHAR>] [-n, --device_num <N>]
                     [-i, --device_id <N>] [-o, --output <FILE>] [-v, --vocab <FILE>]

options:
    -h, --help               show usage
    -t, --task               select task: CHAR, 't' for train and 'i' for inference".
    -n, --device_num         device number used for training: N, default is 1.
    -i, --device_id          device id used for training with single device: N, 0<=N<=7, default is 0.
    -o, --output             assign output file of inference: FILE.
    -v, --vocab              set the vocabulary.
    -m, --metric             set the metric.
```

The command followed shows a example for training with 2 devices.
Ascend:

```ascend
bash run_ascend.sh --task t --device_num 2 --hccl_json /{path}/rank_table.json
```

ps. Discontinuous device id is not supported in `run_ascend.sh` at present, device id in `rank_table.json` must start from 0.

GPU:

```gpu
bash run_gpu.sh --task t --device_num 2
```

If use a single chip, it would be like this:
Ascend:

```ascend
bash run_ascend.sh --task t --device_num 1 --device_id 0
```

GPU:

```gpu
bash run_gpu.sh --task t --device_num 1 --device_id 0
```

## Weights average

```python
python weights_average.py --input_files your_checkpoint_list --output_file model.npz
```

The input_files is a list of you checkpoints file. To use model.npz as the weights, add its path in default_config.yaml at "default_config.yaml".

```default_config.yaml
{
  ...
    "checkpoint_file_path": "/xxx/xxx/model.npz",
    "save_ckpt_steps": 1000,
    ...
  ...
}
```

## Learning rate scheduler

Two learning rate scheduler are provided in our model:

1. [Polynomial decay scheduler](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1).
2. [Inverse square root scheduler](https://ece.uwaterloo.ca/~dwharder/aads/Algorithms/Inverse_square_root/).

LR scheduler could be config in `default_config.yaml`.

For Polynomial decay scheduler, config could be like:

```default_config.yaml
{
  ...
    "optimizer": "adam",
    "lr": 1e-4,
    "lr_scheduler": "poly",
    "poly_lr_scheduler_power": 0.5,
    "decay_steps": 10000,
    "warmup_steps": 2000,
    "min_lr": 1e-6
  ...
}
```

For Inverse square root scheduler, config could be like:

```default_config.yaml
{
  ...
    "optimizer": "adam",
    "lr": 1e-4,
    "lr_scheduler": "isr",
    "decay_start_step": 12000,
    "warmup_steps": 2000,
    "min_lr": 1e-6
  ...
}
```

More detail about LR scheduler could be found in `src/utils/lr_scheduler.py`.

# Environment Requirements

## Platform

- Hardware（Ascend/GPU）
    - Prepare hardware environment with Ascend or GPU processor.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html)

## Requirements

```txt
nltk
numpy
subword-nmt
rouge
```

<https://www.mindspore.cn/tutorials/experts/en/master/infer/inference.html>

# Get started

MASS pre-trains a sequence to sequence model by predicting the masked fragments in an input sequence. After this, downstream tasks including text summarization and conversation response are candidated for fine-tuning the model and for inference.
Here we provide a practice example to demonstrate the basic usage of MASS for pre-training, fine-tuning a model, and the inference process. The overall process is as follows:

1. Download and process the dataset.
2. Modify the `default_config.yaml` to config the network.
3. Run a task for pre-training and fine-tuning.
4. Perform inference and validation.

## Pre-training

For pre-training a model, config the options in `default_config.yaml` firstly:

- Assign the `pre_train_dataset` under `dataset_config` node to the dataset path.
- Choose the optimizer('momentum/adam/lamb' is available).
- Assign the 'ckpt_prefix' and 'ckpt_path' under `checkpoint_path` to save the model files.
- Set other arguments including dataset configurations and network configurations.
- If you have a trained model already, assign the `checkpoint_file_path` to the checkpoint file.

If you use the ascend chip, run the shell script `run_ascend.sh` as followed:

```ascend
bash run_ascend.sh -t t -n 1 -i 1
```

You can also run the shell script `run_gpu.sh` on gpu as followed:

```gpu
bash run_gpu.sh -t t -n 1 -i 1
```

Get the log and output files under the path `./train_mass_*/`, and the model file under the path assigned in the `default_config.yaml` file.

## Fine-tuning

For fine-tuning a model, config the options in `default_config.yaml` firstly:

- Assign the `fine_tune_dataset` under `dataset_config` node to the dataset path.
- Assign the `checkpoint_file_path` under `checkpoint_path` node to the existed model file generated by pre-training.
- Choose the optimizer('momentum/adam/lamb' is available).
- Assign the `ckpt_prefix` and `checkpoint_file_path` under `checkpoint_path` node to save the model files.
- Set other arguments including dataset configurations and network configurations.

If you use the ascend chip, run the shell script `run_ascend.sh` as followed:

```ascend
bash run_ascend.sh -t t -n 1 -i 1
```

You can also run the shell script `run_gpu.sh` on gpu as followed:

```gpu
bash run_gpu.sh -t t -n 1 -i 1
```

Get the log and output files under the path `./train_mass_*/`, and the model file under the path assigned in the `default_config.yaml` file.

## Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as GPU, Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/tutorials/experts/en/master/infer/inference.html).
For inference, config the options in `default_config.yaml` firstly:

- Assign the `default_config.yaml` under `data_path` node to the dataset path.
- Assign the `default_config.yaml` under `checkpoint_path` node to the model file produced by fine-tuning.
- Choose the optimizer('momentum/adam/lamb' is available).
- Assign the `ckpt_prefix` and `checkpoint_file_path` under `checkpoint_path` node to save the model files.
- Set other arguments including dataset configurations and network configurations.

If you use the ascend chip, run the shell script `run_ascend.sh` as followed:

```bash
bash run_ascend.sh -t i -n 1 -i 1 -o {outputfile}
```

You can also run the shell script `run_gpu.sh` on gpu as followed:

```gpu
bash run_gpu.sh -t i -n 1 -i 1 -o {outputfile}
```

## Mindir Inference Process

### [Export MindIR](#contents)

Export MindIR on local

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# Export on ModelArts
# (1) Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url='s3://dir_to_trained_ckpt/'" on default_config.yaml file.
#          Set "file_name='./mass'" on default_config.yaml file.
#          Set "file_format='MINDIR'" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url='s3://dir_to_trained_ckpt/'" on the website UI interface.
#          Add "file_name='./mass'" on the website UI interface.
#          Add "file_format='MINDIR'" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the code directory to "/path/mass" on the website UI interface.
# (3) Set the startup file to "export.py" on the website UI interface.
# (4) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (5) Create your job.
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [CONFIG] [VOCAB] [OUTPUT] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result in acc.log file.

# Performance

## Results

### Fine-Tuning on Text Summarization

The comparisons between MASS and two other pre-training methods in terms of ROUGE score on the text summarization task
with 3.8M training data are as follows:

| Method         |  RG-1(F)      | RG-2(F)      | RG-L(F)      |
|:---------------|:--------------|:-------------|:-------------|
| MASS           | 38.73       | 19.71      | 35.96      |

### Training Performance

| Parameters                 | Masked Sequence to Sequence Pre-training for Language Generation          |
|:---------------------------|:--------------------------------------------------------------------------|
| Model Version              | v1                                                                        |
| Resource                   | Ascend 910; cpu 2.60GHz, 192cores; memory 755G; OS Euler2.8                            |
| uploaded Date              | 06/21/2021                                                                |
| MindSpore Version          | 1.2.1                                                                     |
| Dataset                    | News Crawl 2007-2017 English monolingual corpus, Gigaword corpus |
| Training Parameters        | Epoch=50, steps=XXX, batch_size=192, lr=1e-4                              |
| Optimizer                  | Adam                                                                      |
| Loss Function              | Label smoothed cross-entropy criterion                                    |
| outputs                    | Sentence and probability                                                  |
| Loss                       | Lower than 2                                                              |
| Accuracy                   | For text summarization, RG-1=45.98. |
| Speed                      | 611.45 sentences/s                                                        |
| Total time                 | --/--                                                                     |
| Params (M)                 | 44.6M                                                                     |

### Inference Performance

| Parameters                 | Masked Sequence to Sequence Pre-training for Language Generation |
|:---------------------------|:-----------------------------------------------------------|
| Model Version              | V1                                                         |
| Resource                   | Ascend 910; OS Euler2.8                                                 |
| uploaded Date              | 06/21/2021                                                 |
| MindSpore Version          | 1.2.1                                                      |
| Dataset                    | Gigaword corpus |
| batch_size                 | ---                                                        |
| outputs                    | Sentence and probability                                   |
| Accuracy                   | RG-1=45.98 for text summarization. |
| Speed                      | ---- sentences/s                                           |
| Total time                 | --/--                                                      |

# Description of random situation

MASS model contains dropout operations, if you want to disable dropout, please set related dropout_rate to 0 in `default_config.yaml`.

# others

The model has been validated on Ascend and GPU environments, not validated on CPU.

# ModelZoo Homepage  

 [Link](https://gitee.com/mindspore/models)
