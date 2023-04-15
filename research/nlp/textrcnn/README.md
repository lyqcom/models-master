# TextRCNN

## Contents

- [TextRCNN Description](#textrcnn-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
- [Inference Process](#inference-process)
    - [Export MindIR](#export-mindir)
    - [Infer on Ascend310](#infer-on-ascend310)
    - [result](#result)
- [ModelZoo Homepage](#modelzoo-homepage)

## [TextRCNN Description](#contents)

TextRCNN, a model for text classification, which is proposed by the Chinese Academy of Sciences in 2015.
TextRCNN actually combines RNN and CNN, first uses bidirectional RNN to obtain upper semantic and grammatical information of the input text,
and then uses maximum pooling to automatically filter out the most important feature.
Then connect a fully connected layer for classification.

The TextCNN network structure contains a convolutional layer and a pooling layer. In RCNN, the feature extraction function of the convolutional layer is replaced by RNN. The overall structure consists of  RNN and pooling layer, so it is called RCNN.

[Paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/download/9745/9552):  Siwei Lai, Liheng Xu, Kang Liu, Jun Zhao: Recurrent Convolutional Neural Networks for Text Classification. AAAI 2015: 2267-2273

## [Model Architecture](#contents)

Specifically, the TextRCNN is mainly composed of three parts: a recurrent structure layer, a max-pooling layer, and a fully connected layer. In the paper, the length of the word vector $|e|=50$, the length of the context vector $|c|=50$, the hidden layer size $ H=100$, the learning rate $\alpha=0.01$, the amount of words is $|V|$, the input is a sequence of words, and the output is a vector containing categories.

## [Dataset](#contents)

Dataset used: [Sentence polarity dataset v1.0](http://www.cs.cornell.edu/people/pabo/movie-review-data/)

- Dataset size：10662 movie comments in 2 classes, 9596 comments for train set, 1066 comments for test set.
- Data format：text files. The processed data is in ```./data/```

Dataset used: [Movie Review Data](<http://www.cs.cornell.edu/people/pabo/movie-review-data/>)

- Dataset size：1.18M，5331 positive and 5331 negative processed sentences / snippets.
    - Train：1.06M, 9596 sentences / snippets
    - Test：0.12M, 1066 sentences / snippets

## [Environment Requirements](#contents)

- Hardware: Ascend/GPU
- Framework: [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：[MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html), [MindSpore Python API](https://www.mindspore.cn/docs/en/master/index.html).

## [Quick Start](#contents)

- Preparing environment

```python
  # download the pretrained GoogleNews-vectors-negative300.bin, put it into /tmp
  # you can download from https://code.google.com/archive/p/word2vec/,
  # or from https://pan.baidu.com/s/1NC2ekA_bJ0uSL7BF3SjhIg, code: yk9a

  mv /tmp/GoogleNews-vectors-negative300.bin ./word2vec/
```

- Preparing data

```python
  # split the dataset by the following scripts.
  mkdir -p data/test && mkdir -p data/train
  python data_helpers.py --task dataset_split --data_dir dataset_dir

```

- Running on Ascend

  If you are running the scripts for the first time and , you must prepare dataset and set these parameters in the `default_config.yaml`.

  data_path: "YOUR_SENTENCE_DATA_PATH",     # e.g. ./Sentence-polarity-dataset-v1.0/data
  pos_dir: "DATA_POS_FILE",                 # e.g. ./rt_polaritydata/rt-polarity.pos
  neg_dir: "DATA_NEG_FILE",                 # e.g. ./rt_polaritydata/rt-polarity.neg
  preprocess: "true",
  data_root: "YOUR_SENTENCE_DATA_PATH",     # e.g. ./Sentence-polarity-dataset-v1.0/data
  emb_path: "YOUR_SENTENCE_WORD2VEC_PATH"   # e.g. ./Sentence-polarity-dataset-v1.0/word2vec

```python
# run training
DEVICE_ID=7 python train.py
# or you can use the shell script to train in background
bash scripts/run_train.sh

# run evaluating
DEVICE_ID=7 python eval.py --ckpt_path {checkpoint path}
# or you can use the shell script to evaluate in background
bash scripts/run_eval.sh
```

- Running on GPU

  If you are running the scripts for the first time and , you must prepare dataset and set these parameters in the `default_config.yaml`.

  data_path: "YOUR_SENTENCE_DATA_PATH",     # e.g. ./Sentence-polarity-dataset-v1.0/data
  pos_dir: "DATA_POS_FILE",                 # e.g. ./rt_polaritydata/rt-polarity.pos
  neg_dir: "DATA_NEG_FILE",                 # e.g. ./rt_polaritydata/rt-polarity.neg
  preprocess: "true",
  data_root: "YOUR_SENTENCE_DATA_PATH",     # e.g. ./Sentence-polarity-dataset-v1.0/data
  emb_path: "YOUR_SENTENCE_WORD2VEC_PATH"   # e.g. ./Sentence-polarity-dataset-v1.0/word2vec

```python
# run training
export CUDA_VISIBLE_DEVICES=[DEVICE_ID]
python train.py --device_target GPU
# or you can use the shell script to train in background
bash scripts/run_train_gpu.sh [DEVICE_ID]

# run evaluating
export CUDA_VISIBLE_DEVICES=[DEVICE_ID]
python eval.py --ckpt_path {checkpoint path} --device_target GPU
# or you can use the shell script to evaluate in background
bash scripts/run_eval_gpu.sh [CKPT_FILE] [DEVICE_ID]
```

- Running on ModelArts

  If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training as follows

  You have to prepare the folder 'preprocess'.

  You can change the file name of 'requirements.txt' to 'pip-requirements.txt' for installing some third party libraries automatically on ModelArts.

    - Training standalone on ModelArts

      ```python
      # (1) Upload the code folder to S3 bucket.
      # (2) Click to "create training task" on the website UI interface.
      # (3) Set the code directory to "/{path}/textrcnn" on the website UI interface.
      # (4) Set the startup file to /{path}/textrcnn/train.py" on the website UI interface.
      # (5) Perform a or b.
      #     a. setting parameters in /{path}/textrcnn/default_config.yaml.
      #         1. Set ”enable_modelarts: True“
      #         2. Set ”cell: 'lstm'“(Default is 'gru'. if you want to use lstm, you can do this step)
      #     b. adding on the website UI interface.
      #         1. Set ”enable_modelarts=True“
      #         2. Set ”cell=lstm“(Default is 'gru'. if you want to use lstm, you can do this step)
      # (6) Upload the dataset(the folder 'preprocess') to S3 bucket.
      # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
      # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
      # (9) Under the item "resource pool selection", select the specification of a single cards.
      # (10) Create your job.
      ```

    - evaluating with single card on ModelArts

      ```python
      # (1) Upload the code folder to S3 bucket.
      # (2)  Click to "create training task" on the website UI interface.
      # (3) Set the code directory to "/{path}/textrcnn" on the website UI interface.
      # (4) Set the startup file to /{path}/textrcnn/eval.py" on the website UI interface.
      # (5) Perform a or b.
      #     a. setting parameters in /{path}/textrcnn/default_config.yaml.
      #         1. Set ”enable_modelarts: True“
      #         2. Set ”cell: 'lstm'“(Default is 'gru'. If you want to use lstm, you can do this step)
      #         3. Set ”ckpt_path: './{path}/*.ckpt'“(The *.ckpt file must under the folder 'textrcnn')
      #     b. adding on the website UI interface.
      #         1. Set ”enable_modelarts=True“
      #         2. Set ”cell=lstm“(Default is 'gru'. if you want to use lstm, you can do this step)
      #         3. Set ”ckpt_path=./{path}/*.ckpt“(The *.ckpt file must under the folder 'textrcnn')
      # (6) Upload the dataset(the folder 'preprocess') to S3 bucket.
      # (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
      # (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
      # (9) Under the item "resource pool selection", select the specification of a single cards.
      # (10) Create your job.
      ```

## [Script Description](#contents)

### [Script and Sample Code](#contents)

```python
├── model_zoo
    ├── README.md                       // descriptions about all the models
    ├── textrcnn
        ├── readme.md                   // descriptions about TextRCNN
        ├── ascend310_infer             // application for 310 inference
        ├── scripts
        │   ├──run_train.sh             // shell script for train on Ascend
        │   ├──run_train_gpu.sh         // shell script for train on GPU
        │   ├──run_infer_310.sh         // shell script for 310 infer
        │   └──run_eval.sh              // shell script for evaluation on Ascend
        │   └──run_eval_gpu.sh          // shell script for evaluation on GPU
        ├── src
        │   ├──model_utils
        │   │  ├──config.py             // parsing parameter configuration file of "*.yaml"
        │   │  ├──device_adapter.py     // local or ModelArts training
        │   │  ├──local_adapter.py      // get related environment variables in local training
        │   │  └──moxing_adapter.py     // get related environment variables in ModelArts training
        │   ├──dataset.py               // creating dataset
        │   ├──textrcnn.py              // textrcnn architecture
        │   └──utils.py                 // function related to learning rate
        ├── data_helpers.py             // dataset split script
        ├── default_config.yaml         // parameter configuration
        ├── eval.py                     // evaluation script
        ├── export.py                   // export script
        ├── mindspore_hub_conf.py       // mindspore hub interface
        ├── postprocess.py              // 310infer postprocess script
        ├── preprocess.py               // dataset generation script
        ├── requirements.txt            // some third party libraries that need to be installed
        ├── sample.txt                  // the shell to train and eval the model without '*.sh'
        └── train.py                    // training script
```

### [Script Parameters](#contents)

Parameters for both training and evaluation can be set in `default_config.yaml`

- config for Textrcnn, Sentence polarity dataset v1.0.

  ```python
  num_epochs: 10                  # total training epochs
  lstm_num_epochs: 15             # total training epochs when using lstm
  batch_size: 64                  # training batch size
  cell: 'gru'                     # the RNN architecture, can be 'vanilla', 'gru' and 'lstm'.
  ckpt_folder_path: './ckpt'      # the path to save the checkpoints
  preprocess_path: './preprocess' # the directory to save the processed data
  preprocess: 'false'             # whethere to preprocess the data
  data_path: './data/'            # the path to store the splited data
  lr: 0.001  # 1e-3               # the training learning rate
  lstm_lr_init: 0.002  # 2e-3     # learning rate initial value when using lstm
  lstm_lr_end: 0.0005  # 5e-4     # learning rate end value when using lstm
  lstm_lr_max: 0.003  # 3e-3      # learning eate max value when using lstm
  lstm_lr_warm_up_epochs: 2       # warm up epoch num when using lstm
  lstm_lr_adjust_epochs: 9        # lr adjust in lr_adjust_epoch, after that, the lr is lr_end when using lstm
  emb_path: './word2vec'          # the directory to save the embedding file
  embed_size: 300                 # the dimension of the word embedding
  save_checkpoint_steps: 149      # per step to save the checkpoint
  keep_checkpoint_max: 10         # max checkpoints to save
  ckpt_path: ''                   # relative path of '*.ckpt' to be evaluated relative to the eval.py
  ```

## Inference Process

### [Export MindIR](#contents)

- Export on local

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

- Export on ModelArts (If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start as follows)

```python
# (1) Upload the code folder to S3 bucket.
# (2) Click to "create training task" on the website UI interface.
# (3) Set the code directory to "/{path}/textrcnn" on the website UI interface.
# (4) Set the startup file to /{path}/textrcnn/export.py" on the website UI interface.
# (5) Perform a or b.
#     a. setting parameters in /{path}/textrcnn/default_config.yaml.
#         1. Set ”enable_modelarts: True“
#         2. Set “ckpt_file: ./{path}/*.ckpt”('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Set ”file_name: textrcnn“
#         4. Set ”file_format：MINDIR“
#     b. adding on the website UI interface.
#         1. Add ”enable_modelarts=True“
#         2. Add “ckpt_file=./{path}/*.ckpt”('ckpt_file' indicates the path of the weight file to be exported relative to the file `export.py`, and the weight file must be included in the code directory.)
#         3. Add ”file_name=textrcnn“
#         4. Add ”file_format=MINDIR“
# (7) Check the "data storage location" on the website UI interface and set the "Dataset path" path.
# (8) Set the "Output file path" and "Job log path" to your path on the website UI interface.
# (9) Under the item "resource pool selection", select the specification of a single card.
# (10) Create your job.
# You will see textrcnn.mindir under {Output file path}.
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```bash
============== Accuracy:{} ============== 0.8008
```

### Performance

#### Training Performance

| Model                      | MindSpore + Ascend              | MindSpore + GPU                |
| -------------------------- | ------------------------------- | ------------------------------ |
| Resource                   | Ascend 910; OS Euler2.8         | NV SMX2 V100-32G               |
| Version                    | 1.0.1                           | 1.5.0                          |
| Dataset                    | Sentence polarity dataset v1.0  | Sentence polarity dataset v1.0 |
| epoch_size                 | 10                              | 10                             |
| batch_size                 | 64                              | 64                             |
| loss                       | 0.1720                          | 0.2501                         |
| Speed                      | 1P:23.876 ms/step               | 1p:69.084 ms/step              |

#### Evaluation Performance

| Model                      | MindSpore + Ascend              | MindSpore + GPU                |
| -------------------------- | ------------------------------- | ------------------------------ |
| Resource                   | Ascend 910; OS Euler2.8         | NV SMX2 V100-32G               |
| Version                    | 1.5.0                           | 1.5.0                          |
| Dataset                    | Sentence polarity dataset v1.0  | Sentence polarity dataset v1.0 |
| batch_size                 | 64                              | 64                             |
| Accuracy                   | 0.7930                          | 0.8076                         |

## [ModelZoo Homepage](#contents)

 Please check the official [homepage](https://gitee.com/mindspore/models).
