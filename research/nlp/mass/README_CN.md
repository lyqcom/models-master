# 目录

<!-- TOC -->

- [目录](#目录)
- [掩式序列到序列（MASS）预训练语言生成](#掩式序列到序列mass预训练语言生成)
- [模型架构](#模型架构)
- [数据集](#数据集)
    - [特性](#特性)
    - [脚本说明](#脚本说明)
    - [准备数据集](#准备数据集)
        - [标记化](#标记化)
        - [字节对编码](#字节对编码)
        - [构建词汇表](#构建词汇表)
        - [生成数据集](#生成数据集)
            - [News Crawl语料库](#news-crawl语料库)
            - [Gigaword语料库](#gigaword语料库)
    - [配置](#配置)
    - [训练&评估过程](#训练评估过程)
    - [权重平均值](#权重平均值)
    - [学习速率调度器](#学习速率调度器)
- [环境要求](#环境要求)
    - [平台](#平台)
    - [其他要求](#其他要求)
- [快速上手](#快速上手)
    - [预训练](#预训练)
    - [微调](#微调)
    - [推理](#推理)
    - [Mindir推理](#mindir推理)
        - [导出模型](#导出模型)
        - [在Ascend310执行推理](#在ascend310执行推理)
        - [结果](#结果)
- [性能](#性能)
    - [结果](#结果-1)
        - [文本摘要微调](#文本摘要微调)
        - [训练性能](#训练性能)
        - [推理性能](#推理性能)
- [随机情况说明](#随机情况说明)
- [其他](#其他)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# 掩式序列到序列（MASS）预训练语言生成

[掩式序列到序列（MASS）预训练语言生成](https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-paper-updated-002.pdf)由微软于2019年6月发布。

BERT（Devlin等人，2018年）采用有屏蔽的语料丰富文本预训练Transformer的编码器部分（Vaswani等人，2017年），已在自然语言理解领域实现了性能最优（SOTA）。不仅如此，GPT（Raddford等人，2018年）也采用了有屏蔽的语料丰富文本对Transformer的解码器部分进行预训练（屏蔽了编码器输入）。两者都通过预训练有屏蔽的语料丰富文本来构建一个健壮的语言模型。

受BERT、GPT及其他语言模型的启发，微软致力于在此基础上研究[掩式序列到序列（MASS）预训练语言生成](https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-paper-updated-002.pdf)。MASS的参数k很重要，用来控制屏蔽后的分片长度。BERT和GPT属于特例，k等于1或者句长。

[MASS介绍 — 序列对序列语言生成任务中性能优于BERT和GPT的预训练方法](https://www.microsoft.com/en-us/research/blog/introducing-mass-a-pre-training-method-that-outperforms-bert-and-gpt-in-sequence-to-sequence-language-generation-tasks/)

[论文](https://www.microsoft.com/en-us/research/uploads/prod/2019/06/MASS-paper-updated-002.pdf): Song, Kaitao, Xu Tan, Tao Qin, Jianfeng Lu and Tie-Yan Liu.“MASS: Masked Sequence to Sequence Pre-training for Language Generation.”ICML (2019).

# 模型架构

MASS网络由Transformer实现，Transformer包括多个编码器层和多个解码器层。
预训练中，采用Adam优化器和损失放大来得到预训练后的模型。
微调时，根据不同的任务，采用不同的数据集对预训练的模型进行微调。
测试过程中，通过微调后的模型预测结果，并采用波束搜索算法
获取可能性最高的预测结果。

# 数据集

本文运用数据集包括：

- [News Crawl数据集](https://www.statmt.org/wmt16/translation-task.html)（WMT，2019年）的英语单语数据，用于预训练
- [Gigaword语料库](https://github.com/harvardnlp/sent-summary)（Graff等人，2003年），用于文本摘要

## 特性

MASS设计联合预训练编码器和解码器，来完成语言生成任务。
首先，通过序列到序列的框架，MASS只预测阻塞的标记，迫使编码器理解未屏蔽标记的含义，并鼓励解码器从编码器中提取有用信息。
其次，通过预测解码器的连续标记，可以建立比仅预测离散标记更好的语言建模能力。
第三，通过进一步屏蔽编码器中未屏蔽的解码器的输入标记，鼓励解码器从编码器侧提取更有用的信息，而不是使用前一个标记中的丰富信息。

如果要在modelarts上进行模型的训练，可以参考modelarts的官方指导文档(https://support.huaweicloud.com/modelarts/)
开始进行模型的训练和推理，具体操作如下：

```python
# 在modelarts上使用分布式训练的示例：
# (1) 选址a或者b其中一种方式。
#       a. 设置 "enable_modelarts=True" 。
#          在yaml文件上设置网络所需的参数。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          在modelarts的界面上设置网络所需的参数。
# (2)设置训练任务"task=train"
# (3) 在modelarts的界面上设置代码的路径 "/path/mass"。
# (4) 在modelarts的界面上设置模型的启动文件 "train.py" 。
# (5) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (6) 开始模型的训练。

# 在modelarts上使用模型推理的示例
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件.
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件.
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
# (3) 设置训练任务"task=infer"和vocab的路径
# (4) 在modelarts的界面上设置代码的路径 "/path/mass"。
# (5) 在modelarts的界面上设置模型的启动文件 "eval.py" 。
# (6) 在modelarts的界面上设置模型的数据路径 "Dataset path" ,
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (7) 开始模型的推理。
```

## 脚本说明

MASS脚本及代码结构如下：

```text
├── mass
  ├── README.md                              // MASS模型介绍
  ├── config
  │   ├──config.py                           // 参数配置
  ├── src
      │   ├──model_utils
      │   ├──config.py                       // 参数配置
      │   ├──device_adapter.py               // 设备配置
      │   ├──local_adapter.py                // 本地设备配置
      │   ├──moxing_adapter.py               // modelarts设备配置
  ├──src
  │   ├──dataset
  │      ├──bi_data_loader.py                // 数据集加载器，用于微调或推理
  │      ├──mono_data_loader.py              // 预训练数据集加载器
  │   ├──language_model
  │      ├──noise_channel_language_model.p   // 数据集生成噪声通道语言模型
  │      ├──mass_language_model.py           // 基于MASS论文的MASS语言模型
  │      ├──loose_masked_language_model.py   // 基于MASS发布代码的MASS语言模型
  │      ├──masked_language_model.py         // 基于MASS论文的MASS语言模型
  │   ├──transformer
  │      ├──create_attn_mask.py              // 生成屏蔽矩阵，除去填充部分
  │      ├──transformer.py                   // Transformer模型架构
  │      ├──encoder.py                       // Transformer编码器组件
  │      ├──decoder.py                       // Transformer解码器组件
  │      ├──self_attention.py                // 自注意块组件
  │      ├──multi_head_attention.py          // 多头自注意组件
  │      ├──embedding.py                     // 嵌入组件
  │      ├──positional_embedding.py          // 位置嵌入组件
  │      ├──feed_forward_network.py          // 前馈网络
  │      ├──residual_conn.py                 // 残留块
  │      ├──beam_search.py                   // 推理所用的波束搜索解码器
  │      ├──transformer_for_infer.py         // 使用Transformer进行推理
  │      ├──transformer_for_train.py         // 使用Transformer进行训练
  │   ├──utils
  │      ├──byte_pair_encoding.py            // 使用subword-nmt应用字节对编码（BPE）
  │      ├──dictionary.py                    // 字典
  │      ├──loss_moniter.py                  // 训练步骤中损失监控回调
  │      ├──lr_scheduler.py                  // 学习速率调度器
  │      ├──ppl_score.py                     // 基于N-gram的困惑度评分
  │      ├──rouge_score.py                   // 计算ROUGE得分
  │      ├──load_weights.py                  // 从检查点或者NPZ文件加载权重
  │      ├──initializer.py                   // 参数初始化器
  ├── vocab
  │   ├──all.bpe.codes                       // 字节对编码表（此文件需要用户自行生成）
  │   ├──all_en.dict.bin                     // 已学习到的词汇表（此文件需要用户自行生成）
  ├── scripts
  │   ├──run_ascend.sh                       // Ascend处理器上训练&评估模型脚本
  │   ├──run_gpu.sh                          // GPU处理器上训练&评估模型脚本
  │   ├──learn_subword.sh                    // 学习字节对编码
  │   ├──stop_training.sh                    // 停止训练
  ├── requirements.txt                       // 第三方包需求
  ├── train.py                               // 训练API入口
  ├── eval.py                                // 推理API入口
  ├── default_config.yaml                    // 参数配置
  ├── tokenize_corpus.py                     // 语料标记化
  ├── apply_bpe_encoding.py                  // 应用BPE进行编码
  ├── weights_average.py                     // 将各模型检查点平均转换到NPZ格式
  ├── news_crawl.py                          // 创建预训练所用的News Crawl数据集
  ├── gigaword.py                            // 创建Gigaword语料库

```

## 准备数据集

自然语言处理任务的数据准备过程包括数据清洗、标记、编码和生成词汇表几个步骤。

实验中，使用[字节对编码（BPE）](https://arxiv.org/abs/1508.07909)可以有效减少词汇量，减轻对OOV的影响。

使用`src/utils/dictionary.py`可以基于BPE学习到的文本词典创建词汇表。
有关BPE的更多详细信息，参见[Subword-nmt lib](https://www.cnpython.com/pypi/subword-nmt)或[论文](https://arxiv.org/abs/1508.07909)。

实验中，根据News Crawl数据集的1.9万个句子，学习到的词汇量为45755个单词。

这里我们简单介绍一下准备数据所需的脚本。

### 标记化

使用`tokenize_corpus.py`可以标记`.txt`格式的文本语料。

`tokenize_corpus.py`的主要参数如下：

```bash
--corpus_folder:     Corpus folder path, if multi-folders are provided, use ',' split folders.
--output_folder:     Output folder path.
--tokenizer:         Tokenizer to be used, nltk or jieba, if nltk is not installed fully, use jieba instead.
--pool_size:         Processes pool size.
```

示例代码如下：

```bash
python tokenize_corpus.py --corpus_folder /{path}/corpus --output_folder /{path}/tokenized_corpus --tokenizer {nltk|jieba} --pool_size 16
```

如果你要使用`nltk`，请确保你已经完成了`nltk data`的安装。如果你本地无法完成安装，请尝试使用`jieba`。

### 字节对编码

标记化后，使用提供的`all.bpe.codes`对标记后的语料进行字节对编码处理。

应用BPE所需的脚本为`apply_bpe_encoding.py`。

`apply_bpe_encoding.py`的主要参数如下：

```bash
--codes:            BPE codes file.
--src_folder:       Corpus folders.
--output_folder:    Output files folder.
--prefix:           Prefix of text file in `src_folder`.
--vocab_path:       Generated vocabulary output path.
--threshold:        Filter out words that frequency is lower than threshold.
--processes:        Size of process pool (to accelerate).Default: 2.
```

示例代码如下：

```bash
python apply_bpe_encoding.py --codes /{path}/all.bpe.codes \
    --src_folder /{path}/tokenized_corpus \
    --output_folder /{path}/tokenized_corpus/bpe \
    --prefix tokenized \
    --vocab_path /{path}/vocab_en.dict.bin
    --processes 32
```

### 构建词汇表

如需创建新词汇表，可任选下列方法之一：

1. 重新学习字节对编码，从`subword-nmt`的多个词汇表文件创建词汇表。
2. 基于现有词汇文件创建词汇表，该词汇文件行格式为`word frequency`。
3. *（可选）* 基于`vocab/all_en.dict.bin`，应用`src/utils/dictionary.py`中的`shink`方法创建一个小词汇表。
4. 应用`persistence()`方法将词汇表持久化到`vocab`文件夹。

`src/utils/dictionary.py`的主要接口如下：

1. `shrink(self, threshold=50)`：通过过滤词频低于阈值的单词来缩小词汇量，并返回一个新的词汇表。
2. `load_from_text(cls, filepaths: List[str])`：加载现有文本词汇表，行格式为`word frequency`。  
3. `load_from_persisted_dict(cls, filepath)`：加载通过调用`persistence()`方法保存的持久化二进制词汇表。
4. `persistence(self, path)`：将词汇表对象保存为二进制文件。

示例代码如下：

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

相关信息，参见源文件。

### 生成数据集

如前所述，MASS模式下使用了两个语料数据集，相关数据集生成脚本已提供。

#### News Crawl语料库

数据集生成脚本为`news_crawl.py`。

`news_crawl.py`的主要参数如下：

```bash
Note that please provide `--existed_vocab` or `--dict_folder` at least one.
A new vocabulary would be created in `output_folder` when pass `--dict_folder`.

--src_folder:       Corpus folders.
--existed_vocab:    Optional, persisted vocabulary file.
--mask_ratio:       Ratio of mask.
--output_folder:    Output dataset files folder path.
--max_len:          Maximum sentence length.If a sentence longer than `max_len`, then drop it.
--suffix:           Optional, suffix of generated dataset files.
--processes:        Optional, size of process pool (to accelerate).Default: 2.
```

示例代码如下：

```bash
python news_crawl.py --src_folder /{path}/news_crawl \
    --existed_vocab /{path}/mass/vocab/all_en.dict.bin \
    --mask_ratio 0.5 \
    --output_folder /{path}/news_crawl_dataset \
    --max_len 32 \
    --processes 32
```

#### Gigaword语料库

数据集生成脚本为`gigaword.py`。

`gigaword.py`主要参数如下：

```bash
--train_src:        Train source file path.
--train_ref:        Train reference file path.
--test_src:         Test source file path.
--test_ref:         Test reference file path.
--existed_vocab:    Persisted vocabulary file.
--output_folder:    Output dataset files folder path.
--noise_prob:       Optional, add noise prob.Default: 0.
--max_len:          Optional, maximum sentence length.If a sentence longer than `max_len`, then drop it.Default: 64.
--format:           Optional, dataset format, "mindrecord" or "tfrecord".Default: "tfrecord".
```

示例代码如下：

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

## 配置

`config/`目录下的JSON文件为模板配置文件，
便于为大多数选项及参数赋值，包括训练平台、数据集和模型的配置、优化器参数等。还可以通过设置相应选项，获得诸如损失放大和检查点等可选特性。
有关属性的详细信息，参见`default_config.yaml`文件。

## 训练&评估过程

训练模型时，只需使用shell脚本`run_ascend.sh`或`run_gpu.sh`即可。脚本中设置了环境变量，执行`mass`下的`train.py`训练脚本。
您可以通过选项赋值来启动单卡或多卡训练任务，在bash中运行如下命令：

Ascend处理器：

```ascend
bash run_ascend.sh [--options]
```

GPU处理器：

```gpu
bash run_gpu.sh [--options]
```

`run_ascend.sh`的用法如下：

```text
Usage: run_ascend.sh [-h, --help] [-t, --task <CHAR>] [-n, --device_num <N>]
                     [-i, --device_id <N>] [-j, --hccl_json <FILE>]
                     [-o, --output <FILE>] [-v, --vocab <FILE>]

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

说明：运行分布式训练时，确保已配置`hccl_json`文件。

`run_gpu.sh`的用法如下：

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

运行如下命令进行2卡训练。
Ascend处理器：

```ascend
bash run_ascend.sh --task t --device_num 2 --hccl_json /{path}/rank_table.json
```

注：`run_ascend.sh`暂不支持不连续设备ID，`rank_table.json`中的设备ID必须从0开始。

GPU处理器：

```gpu
bash run_gpu.sh --task t --device_num 2
```

运行如下命令进行单卡训练：
Ascend处理器：

```ascend
bash run_ascend.sh --task t --device_num 1 --device_id 0
```

GPU处理器：

```gpu
bash run_gpu.sh --task t --device_num 1 --device_id 0
```

## 权重平均值

```python
python weights_average.py --input_files your_checkpoint_list --output_file model.npz
```

`input_files`为检查点文件清单。如需使用`model.npz`作为权重文件，请在“checkpoint_file_path”的`default_config.yaml`文件中添加`model.npz`的路径。

```default_config.yaml
{
  ...
    "checkpoint_file_path": "/xxx/xxx/model.npz",
    "save_ckpt_steps": 1000,
    ...
  ...
}
```

## 学习速率调度器

模型中提供了两个学习速率调度器：

1. [多项式衰减调度器](https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1)。
2. [逆平方根调度器](https://ece.uwaterloo.ca/~dwharder/aads/Algorithms/Inverse_square_root/)。

可以在`config/config.json`文件中配置学习率调度器。

多项式衰减调度器配置文件示例如下：

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

逆平方根调度器配置文件示例如下：

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

有关学习率调度器的更多详细信息，参见`src/utils/lr_scheduler.py`。

# 环境要求

## 平台

- 硬件（Ascend或GPU）
    - 使用Ascend或GPU处理器准备硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 更多关于Mindspore的信息，请查看以下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

## 其他要求

```txt
nltk
numpy
subword-nmt
rouge
```

<https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html>

# 快速上手

MASS通过预测输入序列中被屏蔽的片段来预训练序列到序列模型。之后，选择下游的文本摘要或会话应答任务进行模型微调和推理。
这里提供了一个练习示例来演示应用MASS，对模型进行预训练、微调的基本用法，以及推理过程。操作步骤如下：

1. 下载并处理数据集。
2. 修改`default_config.yaml`文件，配置网络。
3. 运行预训练和微调任务。
4. 进行推理验证。

## 预训练

预训练模型时，首先配置`default_config.yaml`中的选项：

- 将`dataset_config`节点下的`pre_train_dataset`配置为数据集路径。
- 选择优化器（可采用'momentum/adam/lamb’）。
- 在`checkpoint_path`下，指定'ckpt_prefix'和'checkpoint_file_path'来保存模型文件。
- 配置其他参数，包括数据集配置和网络配置。
- 如果已经有训练好的模型，请将`checkpoint_file_path`配置为该检查点文件。

如使用Ascend芯片，执行`run_ascend.sh`这个shell脚本：

```ascend
bash run_ascend.sh -t t -n 1 -i 1
```

如使用GPU处理器，执行`run_gpu.sh`这个shell脚本：

```gpu
bash run_gpu.sh -t t -n 1 -i 1
```

日志和输出文件可以在`./train_mass_*/`路径下获取，模型文件可以在`default_config.yaml`配置文件中指定的路径下获取。

## 微调

预训练模型时，首先配置`default_yaml.yaml`中的选项：

- 将`default_config.yaml`下的`data_path`配置为数据集路径。
- 将`default_config.yaml`下的`checkpoint_file_path`赋值给预训练生成的已有模型文件。
- 选择优化器（可采用'momentum/adam/lamb’）。
- 在`default_config.yaml`下，指定'ckpt_prefix'和'checkpoint_path'来保存模型文件。
- 配置其他参数，包括数据集配置和网络配置。

如使用Ascend芯片，执行`run_ascend.sh`这个shell脚本：

```ascend
bash run_ascend.sh -t t -n 1 -i 1
```

如使用GPU处理器，执行`run_gpu.sh`这个shell脚本：

```gpu
bash run_gpu.sh -t t -n 1 -i 1
```

日志和输出文件可以在`./train_mass_*/`路径下获取，模型文件可以在`default_config.yaml`配置文件中指定的路径下获取。

## 推理

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html)。
推理时，请先配置`config.json`中的选项：

- 将`default_config.yaml`节点下的`data_path`配置为数据集路径。
- 选择优化器（可采用'momentum/adam/lamb’）。
- 指定'ckpt_prefix'和'ckpt_path'来保存模型文件。
- 配置其他参数，包括数据集配置和网络配置。

如使用Ascend芯片，执行`run_ascend.sh`这个shell脚本：

```bash
bash run_ascend.sh -t i -n 1 -i 1 -o {outputfile}
```

如使用GPU处理器，执行`run_gpu.sh`这个shell脚本：

```gpu
bash run_gpu.sh -t i -n 1 -i 1 -o {outputfile}
```

## Mindir推理

### [导出模型](#contents)

本地导出mindir

```shell
python export.py --checkpoint_file_path [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

参数checkpoint_file_path为必填项，
`FILE_FORMAT` 必须在 ["AIR", "MINDIR"]中选择。

ModelArts导出mindir

```python
# (1) 把训练好的模型地方到桶的对应位置。
# (2) 选址a或者b其中一种方式。
#       a.  设置 "enable_modelarts=True"
#          设置 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt" 在 yaml 文件。
#          设置 "checkpoint_url=/The path of checkpoint in S3/" 在 yaml 文件。
#          设置 "file_name='./mass'"参数在yaml文件。
#          设置 "file_format='MINDIR'" 参数在yaml文件。
#       b. 增加 "enable_modelarts=True" 参数在modearts的界面上。
#          增加 "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" 参数在modearts的界面上。
#          增加 "checkpoint_url=/The path of checkpoint in S3/" 参数在modearts的界面上。
#          设置 "file_name='./mass'"参数在modearts的界面上。
#          设置 "file_format='MINDIR'" 参数在modearts的界面上。
# (3) 在modelarts的界面上设置代码的路径 "/path/mass"。
# (4) 在modelarts的界面上设置模型的启动文件 "export.py" 。
# 模型的输出路径"Output file path" 和模型的日志路径 "Job log path" 。
# (5) 开始导出mindir。
```

### 在Ascend310执行推理

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

在执行推理前，mindir文件必须通过`export.py`脚本导出。以下展示了使用minir模型执行推理的示例。

```shell
# Ascend310推理
bash run_infer_310.sh [MINDIR_PATH] [CONFIG] [VOCAB] [OUTPUT] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` 表示数据是否需要预处理，取值范围为：'y' 或者 'n'。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，精度计算结果可以在acc.log中看到。

# 性能

## 结果

### 文本摘要微调

下表展示了，相较于其他两种预训练方法，MASS在文本摘要任务中的ROUGE得分情况。
训练数据大小为3.8M。

| 方法| RG-1(F) | RG-2(F) | RG-L(F) |
|:---------------|:--------------|:-------------|:-------------|
| MASS | 38.73 | 19.71 | 35.96 |

### 训练性能

| 参数 | 掩式序列到序列预训练语言生成 |
|:---------------------------|:--------------------------------------------------------------------------|
| 模型版本              | v1                                                                        |
| 资源                   | Ascend 910；CPU 2.60GHz，192核；内存 755GB；系统 Euler2.8              |
| 上传日期              | 2021-06-21                                                 |
| MindSpore版本          | 1.2.1                                                                     |
| 数据集 | News Crawl 2007-2017英语单语语料库、Gigaword语料库 |
| 训练参数 | Epoch=50, steps=XXX, batch_size=192, lr=1e-4 |
| 优化器                  | Adam                                                        |
| 损失函数 | 标签平滑交叉熵准则 |
| 输出 | 句子及概率 |
| 损失                       | 小于2                                                            |
| 准确性 | 文本摘要RG-1=45.98 |
| 速度                      | 611.45句子/秒                              |
| 总时长                 |                               |
| 参数(M)                 | 44.6M                                                          |

### 推理性能

| 参数 | 掩式序列到序列预训练语言生成 |
|:---------------------------|:-----------------------------------------------------------|
|模型版本| V1 |
| 资源                  | Ascend 910；系统 Euler2.8                                                     |
| 上传日期 | 2020-06-21 |
| MindSpore版本 | 1.2.1 |
| 数据集 | Gigaword语料库 |
| batch_size          | ---                                                        |
| 输出 | 句子及概率 |
| 准确度 | 文本摘要RG-1=45.98 |
| 速度                      | ----句子/秒                              |
| 总时长 | --/-- |

# 随机情况说明

MASS模型涉及随机失活（dropout）操作，如需禁用此功能，请在`default_config.yaml`中将dropout_rate设置为0。

# 其他

该模型已在Ascend环境下与GPU环境下得到验证，尚未在CPU环境下验证。

# ModelZoo主页  

 [链接](https://gitee.com/mindspore/models)
