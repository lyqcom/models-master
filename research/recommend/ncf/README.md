# Contents

- [NCF Description](#NCF-description)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Features](#features)
    - [Mixed Precision](#mixed-precision)
- [Environment Requirements](#environment-requirements)
- [Quick Start](#quick-start)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
    - [Script Parameters](#script-parameters)
    - [Training Process](#training-process)
        - [Training](#training)
        - [Distributed Training](#distributed-training)  
    - [Evaluation Process](#evaluation-process)
        - [Evaluation](#evaluation)
    - [Inference Process](#inference-process)
        - [Export MindIR](#export-mindir)
        - [Infer on Ascend310](#infer-on-ascend310)
        - [result](#result)
- [Model Description](#model-description)
    - [Performance](#performance)  
        - [Evaluation Performance](#evaluation-performance)
        - [Inference Performance](#evaluation-performance)
    - [How to use](#how-to-use)
        - [Inference](#inference)
        - [Continue Training on the Pretrained Model](#continue-training-on-the-pretrained-model)
        - [Transfer Learning](#transfer-learning)
- [Description of Random Situation](#description-of-random-situation)
- [ModelZoo Homepage](#modelzoo-homepage)

# [NCF Description](#contents)

NCF is a general framework for collaborative filtering of recommendations in which a neural network architecture is used to model user-item interactions. Unlike traditional models, NCF does not resort to Matrix Factorization (MF) with an inner product on latent features of users and items. It replaces the inner product with a multi-layer perceptron that can learn an arbitrary function from data.

[Paper](https://arxiv.org/abs/1708.05031):  He X, Liao L, Zhang H, et al. Neural collaborative filtering[C]//Proceedings of the 26th international conference on world wide web. 2017: 173-182.

# [Model Architecture](#contents)

Two instantiations of NCF are Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP). GMF applies a linear kernel to model the latent feature interactions, and and MLP uses a nonlinear kernel to learn the interaction function from data. NeuMF is a fused model of GMF and MLP to better model the complex user-item interactions, and unifies the strengths of linearity of MF and non-linearity of MLP for modeling the user-item latent structures. NeuMF allows GMF and MLP to learn separate embeddings, and combines the two models by concatenating their last hidden layer. [neumf_model.py](neumf_model.py) defines the architecture details.

# [Dataset](#contents)

The [MovieLens datasets](http://files.grouplens.org/datasets/movielens/) are used for model training and evaluation. Specifically, we use two datasets: **ml-1m** (short for MovieLens 1 million) and **ml-20m** (short for MovieLens 20 million).

## ml-1m

ml-1m dataset contains 1,000,209 anonymous ratings of approximately 3,706 movies made by 6,040 users who joined MovieLens in 2000. All ratings are contained in the file "ratings.dat" without header row, and are in the following format:

```cpp
  UserID::MovieID::Rating::Timestamp
```

- UserIDs range between 1 and 6040.
- MovieIDs range between 1 and 3952.
- Ratings are made on a 5-star scale (whole-star ratings only).

## ml-20m

ml-20m dataset contains 20,000,263 ratings of 26,744 movies by 138493 users. All ratings are contained in the file "ratings.csv". Each line of this file after the header row represents one rating of one movie by one user, and has the following format:

```text
userId,movieId,rating,timestamp
```

- The lines within this file are ordered first by userId, then, within user, by movieId.
- Ratings are made on a 5-star scale, with half-star increments (0.5 stars - 5.0 stars).

In both datasets, the timestamp is represented in seconds since midnight Coordinated Universal Time (UTC) of January 1, 1970. Each user has at least 20 ratings.

# [Features](#contents)

## Mixed Precision

The [mixed precision](https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html) training method accelerates the deep learning neural network training process by using both the single-precision and half-precision data formats, and maintains the network precision achieved by the single-precision training at the same time. Mixed precision training can accelerate the computation process, reduce memory usage, and enable a larger model or batch size to be trained on specific hardware.
For FP16 operators, if the input data type is FP32, the backend of MindSpore will automatically handle it with reduced precision. Users could check the reduced-precision operators by enabling INFO log and then searching ‘reduce precision’.

# [Environment Requirements](#contents)

- Hardware(Ascend/GPU/CPU）
    - Prepare hardware environment with Ascend.
- Framework
    - [MindSpore](https://www.mindspore.cn/install/en)
- For more information, please check the resources below：
    - [MindSpore tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
#run data process
bash scripts/run_download_dataset.sh

# run training example on Ascend
bash scripts/run_train_ascend.sh

# run training example on GPU
bash scripts/run_train_gpu.sh

# run training distribute example on Ascend
bash scripts/run_distribute_train.sh /path/hccl.json /path/MovieLens

# run evaluation example on Ascend
bash scripts/run_eval_ascend.sh

# run evaluation example on GPU
bash scripts/run_eval_gpu.sh
```

If you want to run in modelarts, please check the official documentation of [modelarts](https://support.huaweicloud.com/modelarts/), and you can start training and evaluation as follows:

```python
# run distributed training on modelarts example
# (1) First, Perform a or b.
#       a. Set "enable_modelarts=True" on default_config.yaml file.
#          Set other parameters on default_config.yaml file you need.
#       b. Add "enable_modelarts=True" on the website UI interface.
#          Add other parameters on the website UI interface.
# (2) Set the code directory to "/path/ncf" on the website UI interface.
# (3) Set the startup file to "train.py" on the website UI interface.
# (4) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (5) Create your job.

# run evaluation on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a. Set "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on default_config.yaml file.
#       b. Add "checkpoint_file_path='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Add "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the code directory to "/path/ncf" on the website UI interface.
# (4) Set the startup file to "eval.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.

# run export on modelarts example
# (1) Copy or upload your trained model to S3 bucket.
# (2) Perform a or b.
#       a. Set "file_name='ncf'" on default_config.yaml file.
#          Set "file_format='MINDIR'" on default_config.yaml file.
#          Set "ckpt_file='/cache/checkpoint_path/model.ckpt'" on default_config.yaml file.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on default_config.yaml file.
#       b. Add "file_name='ncf'" on the website UI interface.
#          Add "file_format='MINDIR'" on the website UI interface.
#          Add "ckpt_file='/cache/checkpoint_path/model.ckpt'" on the website UI interface.
#          Set "checkpoint_url=/The path of checkpoint in S3/" on the website UI interface.
# (3) Set the code directory to "/path/ncf" on the website UI interface.
# (4) Set the startup file to "export.py" on the website UI interface.
# (5) Set the "Dataset path" and "Output file path" and "Job log path" to your path on the website UI interface.
# (6) Create your job.
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

```text
├── ModelZoo_NCF_ME
    ├── README.md                          // descriptions about NCF
    ├── scripts
    │   ├──ascend_distributed_launcher
    │       ├──__init__.py                      // init file
    │       ├──get_distribute_pretrain_cmd.py   // create distribute shell script
    │   ├──run_train_ascend.sh             // shell script for train on Ascend
    │   ├──run_distribute_train.sh         // shell script for distribute train
    │   ├──run_eval_ascend.sh              // shell script for evaluation on Ascend
    │   ├──run_train_gpu.sh                // shell script for train on GPU
    │   ├──run_eval_gpu.sh                 // shell script for evaluation on GPU
    │   ├──run_download_dataset.sh         // shell script for dataget and process
    │   ├──run_transfer_ckpt_to_air.sh     // shell script for transfer model style
    ├── src
    │   ├──dataset.py                      // creating dataset
    │   ├──ncf.py                          // ncf architecture
    │   ├──config.py                       // parameter analysis
    │   ├──device_adapter.py               // device adapter
    │   ├──local_adapter.py                // local adapter
    │   ├──moxing_adapter.py               // moxing adapter
    │   ├──movielens.py                    // data download file
    │   ├──callbacks.py                    // model loss and eval callback file
    │   ├──constants.py                    // the constants of model
    │   ├──export.py                       // export checkpoint files into geir/onnx
    │   ├──metrics.py                      // the file for auc compute
    │   ├──stat_utils.py                   // the file for data process functions
    ├── default_config.yaml    // parameter configuration
    ├── train.py               // training script
    ├── eval.py                //  evaluation script
```

## [Script Parameters](#contents)

Parameters for both training and evaluation can be set in config.py.

- config for NCF, ml-1m dataset

  ```text
  * `--data_path`: This should be set to the same directory given to the data_download data_dir argument.
  * `--dataset`: The dataset name to be downloaded and preprocessed. By default, it is ml-1m.
  * `--train_epochs`: Total train epochs.
  * `--batch_size`: Training batch size.
  * `--eval_batch_size`: Eval batch size.
  * `--num_neg`: The Number of negative instances to pair with a positive instance.
  * `--layers`： The sizes of hidden layers for MLP.
  * `--num_factors`：The Embedding size of MF model.
  * `--output_path`：The location of the output file.
  * `--eval_file_name` : Eval output file.
  ```

## [Training Process](#contents)

### Training

- on Ascend

  ```bash
  # train single
  bash scripts/run_train_ascend.sh [DATASET_PATH] [CKPT_FILE]
  # train distribute
  bash scripts/run_distribute_train.sh [RANK_TABLE_FILE] [DATA_PATH]
  ```

- on GPU

  ```bash
  bash scripts/run_train_gpu.sh [DATASET_PATH] [CKPT_FILE] [DEVICE_ID]
  ```run_train

- on CPU

  ```bash
  pytrun_trainth=./dataset --dataset=ml-1m --train_epochs=25 --batch_size=256 --output_path=./output/ --crun_trainpoint --device_target=CPU --device_id=0 --num_parallel_workers=2 > train.log 2>&1 &
  ```

  The python command above will run in the background, you can view the results through the file `train.log`. After training, you'll get some checkpoint files under the script folder by default. The loss value will be achieved as follows:

  ```log
  # grep "loss is " train.log
  ds_train.size: 95
  epoch: 1 step: 95, loss is 0.25074288
  epoch: 2 step: 95, loss is 0.23324402
  epoch: 3 step: 95, loss is 0.18286772
  ...  
  ```

  The model checkpoint will be saved in the current directory.

## [Evaluation Process](#contents)

### Evaluation

- evaluation on ml-1m dataset when running on Ascend

  Before running the command below, please check the checkpoint path used for evaluation. Please set the checkpoint path to be the absolute full path, e.g., "checkpoint/ncf-125_390.ckpt".

  ```bash
  bash scripts/run_eval_ascend.sh [DATASET_PATH] [CKPT_FILE] [DEVICE_ID]
  ```

  The above python command will run in the background. You can view the results through the file "eval.log". The accuracy of the test dataset will be as follows:

  ```log
  # grep "accuracy: " eval.log
  HR:0.6846,NDCG:0.410
  ```

- evaluation on ml-1m dataset when running on GPU

  For details, see the above contents `evaluation on ml-1m dataset when running on Ascend`.

  ```bash
  bash scripts/run_eval_gpu.sh [DATASET_PATH] [CKPT_FILE] [DEVICE_ID]
  ```

- evaluation on ml-1m dataset when running on CPU

  ```bash
  python eval.py --data_path=./dataset --dataset=ml-1m --eval_batch_size=160000 --output_path=./output/ --eval_file_name=eval.log --checkpoint_file_path=./ckpt --device_target=CPU --device_id=0 > eval.log 2>&1 &
  ```

  The accuracy of the test dataset will be as follows:

  ```log
  # grep "accuracy: " eval.log
  HR = 0.6975, NDCG = 0.420
  ```

## Inference Process

**Before inference, please refer to [MindSpore Inference with C++ Deployment Guide](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README.md) to set environment variables.**

### [Export MindIR](#contents)

```shell
python export.py --ckpt_file [CKPT_PATH] --file_name [FILE_NAME] --file_format [FILE_FORMAT]
```

The ckpt_file parameter is required,
`FILE_FORMAT` should be in ["AIR", "MINDIR"]

### Infer on Ascend310

Before performing inference, the mindir file must be exported by `export.py` script. We only provide an example of inference using MINDIR model.

```shell
# Ascend310 inference
bash run_infer_310.sh [MINDIR_PATH] [NEED_PREPROCESS] [DEVICE_ID]
```

- `NEED_PREPROCESS` means weather need preprocess or not, it's value is 'y' or 'n'.
- `DEVICE_ID` is optional, default value is 0.

### result

Inference result is saved in current path, you can find result like this in acc.log file.

```grep "accuracy: " acc.log
  HR:0.6846,NDCG:0.410
```

# [Model Description](#contents)

## [Performance](#contents)

### Evaluation Performance

| Parameters                 | Ascend                                                       | GPU                                                       | CPU                                                    |
| -------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |--------------------------------------------------------|
| Model Version              | NCF                                                 | NCF                                                 | NCF                                                    |
| Resource                   | Ascend 910; CPU 2.60GHz, 56cores; Memory 314G; OS Euler2.8             | NV SMX2 V100-32G             | CPU AMD R75800H,3.2GHz,8cores;Memory 16G; 0S Windows11 |
| uploaded Date              | 10/23/2020 (month/day/year)                                  | 08/28/2021 (month/day/year)                                  | 07/27/2022 (month/day/year)                            |
| MindSpore Version          | 1.0.0                                                | 1.4.0                                                | 1.6.1                                                  |
| Dataset                    | ml-1m                                                        | ml-1m                                                        | ml-1m                                                  |
| Training Parameters        | epoch=25, steps=19418, batch_size = 256, lr=0.00382059       | epoch=25, steps=19418, batch_size = 256, lr=0.00382059       | epoch=25, steps=19418, batch_size = 256, lr=0.00382059 |
| Optimizer                  | GradOperation                                                | GradOperation                                                | GradOperation                                          |
| Loss Function              | Softmax Cross Entropy                                        | Softmax Cross Entropy                                        | Softmax Cross Entropy                                  |
| outputs                    | probability                                                  | probability                                                  | probability                                            |
| Speed                      | 1pc: 0.575 ms/step                                          | 1pc: 2.5 ms/step                                          | 1pc: 3.6 ms/step                                       |
| Total time                 | 1pc: 5 mins                       | 1pc: 25 mins                       | 1pc: 29 mins                                           |

### Inference Performance

| Parameters          | Ascend              | CPU                        |
| ------------------- | --------------------------- |----------------------------|
| Model Version       | NCF               | NCF                        |
| Resource            | Ascend 910; OS Euler2.8                  | AMD R75800H; OS Windows11  |
| Uploaded Date       | 10/23/2020 (month/day/year)  | 7/27/2022 (month/day/year) |
| MindSpore Version   | 1.0.0                | 1.6.1                      |
| Dataset             | ml-1m                       | ml-1m                      |
| batch_size          | 256                         | 256                        |
| outputs             | probability                 | probability                |
| Accuracy            | HR:0.6846,NDCG:0.410        | HR:0.6975, NDCG:0.420      |

## [How to use](#contents)

### Inference

If you need to use the trained model to perform inference on multiple hardware platforms, such as Ascend 910 or Ascend 310, you can refer to this [Link](https://www.mindspore.cn/tutorials/experts/en/master/infer/inference.html). Following the steps below, this is a simple example:

<https://www.mindspore.cn/tutorials/experts/en/master/infer/inference.html>

  ```python
  # Load unseen dataset for inference
  dataset = dataset.create_dataset(cfg.data_path, 1, False)

  # Define model
  net = GoogleNet(num_classes=cfg.num_classes)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01,
                 cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'})

  # Load pre-trained model
  param_dict = load_checkpoint(cfg.checkpoint_path)
  load_param_into_net(net, param_dict)
  net.set_train(False)

  # Make predictions on the unseen dataset
  acc = model.eval(dataset)
  print("accuracy: ", acc)
  ```

### Continue Training on the Pretrained Model

  ```python
  # Load dataset
  dataset = create_dataset(cfg.data_path, cfg.epoch_size)
  batch_num = dataset.get_dataset_size()

  # Define model
  net = GoogleNet(num_classes=cfg.num_classes)
  # Continue training if set pre_trained to be True
  if cfg.pre_trained:
      param_dict = load_checkpoint(cfg.checkpoint_path)
      load_param_into_net(net, param_dict)
  lr = lr_steps(0, lr_max=cfg.lr_init, total_epochs=cfg.epoch_size,
                steps_per_epoch=batch_num)
  opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                 Tensor(lr), cfg.momentum, weight_decay=cfg.weight_decay)
  loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
  model = Model(net, loss_fn=loss, optimizer=opt, metrics={'acc'},
                amp_level="O2", keep_batchnorm_fp32=False, loss_scale_manager=None)

  # Set callbacks
  config_ck = CheckpointConfig(save_checkpoint_steps=batch_num * 5,
                               keep_checkpoint_max=cfg.keep_checkpoint_max)
  time_cb = TimeMonitor(data_size=batch_num)
  ckpoint_cb = ModelCheckpoint(prefix="train_googlenet_cifar10", directory="./",
                               config=config_ck)
  loss_cb = LossMonitor()

  # Start training
  model.train(cfg.epoch_size, dataset, callbacks=[time_cb, ckpoint_cb, loss_cb])
  print("train success")
  ```

# [Description of Random Situation](#contents)

In dataset.py, we set the seed inside “create_dataset" function. We also use random seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
