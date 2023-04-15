# 目录

<!-- TOC -->

- [midas描述](#midas描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [特性](#特性)
    - [混合精度](#混合精度)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
    - [评估过程](#评估过程)
    - [ONNX评估过程](#ONNX评估过程)
- [推理过程](#推理过程)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#ModelZoo主页)

<!-- /TOC -->

# midas描述

## 概述

Midas全称为Towards Robust Monocular Depth Estimation:Mixing Datasets for Zero-shot Cross-dataset Transfer,用来估计图片的深度信息，使用了五个不同的训练数据集，五个训练数据集混合策略为多目标优化，其中包括作者自制的3D电影数据集，使用6个和训练集完全不同的测试集进行验证。本次只使用ReDWeb数据集进行训练。

Midas模型网络具体细节可参考[Towards Robust Monocular Depth Estimation:Mixing Datasets for
Zero-shot Cross-dataset Transfer](https://arxiv.org/pdf/1907.01341v3.pdf)，Midas模型网络的Pytorch版本实现，可参考(<https://github.com/intel-isl/MiDaS>)。

## 论文

1. [论文:](https://arxiv.org/pdf/1907.01341v3.pdf) Ranftl*, Katrin Lasinger*, David Hafner, Konrad Schindler, and Vladlen Koltun.

# 模型架构

Midas的总体网络架构如下：
[链接](https://arxiv.org/pdf/1907.01341v3.pdf)

# 数据集

使用的数据集：[ReDWeb](<https://www.paperswithcode.com/dataset/redweb>)

- 数据集大小：
    - 训练集：292M, 3600个图像
- 数据格式：
    - 原图imgs：JPG
    - 深度图RDs：PNG

# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件(Ascend)
    - 准备Ascend处理器搭建硬件环境.
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- 预训练模型

  当开始训练之前需要获取mindspore图像网络预训练模型，使用在resnext101上训练出来的预训练模型[resnext101_32x8d_wsl](<https://download.pytorch.org/models/ig_resnext101_32x8-c38310e5.pth>),下载完pth文件之后,运行`python src/utils/pth2ckpt.py /pth_path/ig_resnext101_32x8-c38310e5.pth`将pth文件转换为ckpt文件.
- 数据集准备

  midas网络模型使用ReDWeb数据集用于训练,使用Sintel,KITTI,TUM数据集进行推理,数据集可通过[ReDWeb](<https://www.paperswithcode.com/dataset/redweb>),[Sintel](http://sintel.is.tue.mpg.de),[Kitti](http://www.cvlibs.net/datasets/kitti/raw_data.php),[TUM](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg2_desk_with_person)官方网站下载使用.
  Sintel数据集需要分别下载原图和深度图，放入到Sintel数据集文件夹中。TUM数据集根据处理函数得到associate.txt进行匹配数据。所有处理函数在preprocess文件夹下，具体可参考preprocess文件夹下的readme.md。
- 下载完数据集之后,按如下目录格式存放数据集和代码并将`midas/mixdata.json`和数据集放到`data/`目录下即可:

    ```path
        └── midas
        └── data
            ├── mixdata.json
            ├── ReDWeb_V1
            |   ├─ Imgs
            |   └─ RDs
            ├── Kitti_raw_data
            |   ├─ 2011_09_26_drive_0002_sync
            |   |   ├─depth
            |   |   └─image
            |   ├─ ...
            |   |   ├─depth
            |   |   └─image
            |   ├─ 2011_10_03_drive_0047_sync
            |   |   ├─depth
            |   |   └─image
            ├── TUM
            |   ├─ rgbd_dataset_freiburg2_desk_with_person
            |   |   ├─ associate.txt
            |   |   ├─ depth.txt
            |   |   ├─ rgb.txt
            |   |   ├─ rgb
            |   |   |   ├─ rgb
            |   |   |   ├─ 1311870426.504412.png
            |   |   |   ├─ ...
            |   |   |   └─ 1311870426.557430.png
            |   |   |   ├─ depth
            |   |   |   ├─ 1311870427.207687.png
            |   |   |   ├─ ...
            |   |   |   └─ 1311870427.376229.png
            ├── Sintel
            |   ├─ depth
            |   ├─ final_left
            |   └─ occlusions
    ```

- Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh 8 ./ckpt/midas_resnext_101_WSL.ckpt

# 单机训练
用法：bash run_standalone_train.sh [DEVICE_ID] [CKPT_PATH]

# 运行评估示例
用法：bash run_eval.sh [DEVICE_ID] [DATA_NAME] [CKPT_PATH]
```

# 脚本说明

## 脚本及样例代码

```shell

└──midas
  ├── README.md
  ├── ascend310_infer
    ├── inc
        └── utils.sh                       # 310头文件
    ├── src
        ├── main.cc                        # 310主函数
        └── utils.cc                       # 310函数
    ├── build.sh                           # 编译310环境
    └── CMakeLists.txt                     # 310推理环境
  ├── scripts
    ├── run_distribute_train.sh            # 启动Ascend分布式训练（8卡）
    ├── run_eval.sh                        # 启动Ascend评估
    ├── run_eval_onnx.sh                   # 启动ONNX评估
    ├── run_standalone_train.sh            # 启动Ascend单机训练（单卡）
    ├── run_train_gpu.sh                   # 启动GPU训练
    └── run_infer_310.sh                   # 启动Ascend的310推理
  ├── src
    ├── utils
        ├── loadImgDepth.py                # 读取数据集
        └── transforms.py                  # 图像处理转换
    ├─config.py                            # 训练配置
    ├── cunstom_op.py                      # 网络操作
    ├── blocks_ms.py                       # 网络组件
    ├── loss.py                            # 损失函数定义
    ├── util.py                            # 读取图片工具
    └── midas_net.py                       # 主干网络定义
  ├── config.yaml                          # 训练参数配置文件
  ├── midas_eval.py                        # 评估网络
  ├── midas_eval_onnx.py                   # ONNX评估网络
  ├── midas_export.py                      # 模型导出
  ├── midas_run.py                         # 模型运行
  ├── postprocess.py                       # 310后处理
  └── midas_train.py                       # 训练网络
```

## 脚本参数

在config.yaml中配置相关参数。

- 配置训练相关参数：

```python
device_target: 'Ascend'                                          #服务器的类型,有CPU,GPU,Ascend
device_id: 7                                                     #卡的编号
run_distribute: False                                            #是否进行分布式并行训练
is_modelarts: False                                              #是否在云上训练
no_backbone_params_lr: 0.00001                                   #1e-5
no_backbone_params_end_lr: 0.00000001                            #1e-8
backbone_params_lr: 0.0001                                       #1e-4
backbone_params_end_lr: 0.0000001                                #1e-7
power: 0.5                                                       #PolynomialDecayLR种控制lr参数
epoch_size: 400                                                  #总epoch
batch_size: 8                                                    #batch_size
lr_decay: False                                                   #是否采用动态学习率
train_data_dir: '/midas/'                       #训练集根路径
width_per_group: 8                                               #网络参数
groups: 32
in_channels: 64
features: 256
layers: [3, 4, 23, 3]
img_width: 384                                                   #输入网络的图片宽度
img_height: 384                                                  #输入网络的图片高度
nm_img_mean: [0.485, 0.456, 0.406]                               #图片预处理正则化参数
nm_img_std: [0.229, 0.224, 0.225]
resize_target: True                                              #如果为True,修改image, mask, target的尺寸，否则只修改image尺寸
keep_aspect_ratio: False                                         #保持纵横比
ensure_multiple_of: 32                                           #图片尺寸为32倍数
resize_method: "upper_bound"                                     #resize模式
```

- 配置验证相关参数：

```python
datapath_TUM: '/data/TUM'                                        #TUM数据集地址
datapath_Sintel: '/data/sintel/sintel-data'                      #Sintel数据集地址
datapath_ETH3D: '/data/ETH3D/ETH3D-data'                         #ETH3D数据集地址
datapath_Kitti: '/data/Kitti_raw_data'                           #Kitti数据集地址
datapath_DIW: '/data/DIW'                                        #DIW数据集地址
datapath_NYU: ['/data/NYU/nyu.mat','/data/NYU/splits.mat']       #NYU数据集地址
ann_file: 'val.json'                                             #存放推理结果的文件地址
ckpt_path: '/midas/ckpt/Midas_0-600_56_1.ckpt'                   #存放推理使用的ckpt地址
data_name: 'all'                                               #需要推理的数据集名称，有 Sintel,Kitti,TUM,DIW,ETH3D,all
```

- 配置运行和导出模型相关参数：

```python
input_path: '/midas/input'                  #输入图片的路径
output_path: '/midas/output'                #模型输出图片的路径
model_weights: '/ckpt/Midas_0-600_56_1.ckpt'#模型参数路径
file_format: "MINDIR"  # ["AIR", "MINDIR"]                          #AIR/MIDIR
```

## 训练过程

### 用法

#### Ascend处理器环境运行

```text
# 分布式训练
用法：bash run_distribute_train.sh 8 ./ckpt/midas_resnext_101_WSL.ckpt
# 单机训练
用法：bash run_standalone_train.sh [DEVICE_ID] [CKPT_PATH]
# 运行评估示例
用法：bash run_eval.sh [DEVICE_ID] [DATA_NAME] [CKPT_PATH]

```

#### GPU处理器环境运行

```text
用法：bash run_train_GPU.sh [DEVICE_NUM] [DEVICE_ID] [CKPT_PATH]
# 分布式训练
用法：bash run_train_GPU.sh 8 0,1,2,3,4,5,6,7 /ckpt/midas_resnext_101_WSL.ckpt
# 单机训练
用法：bash run_train_GPU.sh 1 0 /ckpt/midas_resnext_101_WSL.ckpt
# 运行评估示例
用法：bash run_eval.sh [DEVICE_ID] [DATA_NAME] [CKPT_PATH]

```

### 结果

- 使用ReDWeb数据集训练midas

```text
分布式训练结果（8P）
epoch: 1 step: 56, loss is 579.5216
epoch time: 1497998.993 ms, per step time: 26749.982 ms
epoch: 2 step: 56, loss is 773.3644
epoch time: 74565.443 ms, per step time: 1331.526 ms
epoch: 3 step: 56, loss is 270.76688
epoch time: 63373.872 ms, per step time: 1131.676 ms
epoch: 4 step: 56, loss is 319.71643
epoch time: 61290.421 ms, per step time: 1094.472 ms
...
epoch time: 58586.128 ms, per step time: 1046.181 ms
epoch: 396 step: 56, loss is 8.707727
epoch time: 63755.860 ms, per step time: 1138.498 ms
epoch: 397 step: 56, loss is 8.139318
epoch time: 47222.517 ms, per step time: 843.259 ms
epoch: 398 step: 56, loss is 10.746628
epoch time: 23364.224 ms, per step time: 417.218 ms
epoch: 399 step: 56, loss is 7.4859796
epoch time: 24304.195 ms, per step time: 434.003 ms
epoch: 400 step: 56, loss is 8.2024975
epoch time: 23696.833 ms, per step time: 423.158 ms
```

## 评估过程

### 用法

#### Ascend处理器环境运行

可通过改变config.yaml文件中的"data_name"进行对应的数据集推理，默认为全部数据集。

```bash
# 评估
bash run_eval.sh [DEVICE_ID] [DATA_NAME]
```

### 结果

打开val.json查看推理的结果,如下所示：

```text
{"Kitti": 24.222 "Sintel":0.323 "TUM":15.08 }
```

#### GPU处理器环境运行

```bash
# 评估
bash run_eval.sh [DEVICE_ID] [DATA_NAME] [CKPT_PATH] [DEVICE_TARGET]
```

### 结果

打开val.json查看推理的结果,如下所示：

```text
{"Kitti": 24.222 "Sintel":0.323 "TUM":15.08 }
```

## ONNX评估过程

可通过改变config.yaml文件中的"data_name"进行对应的数据集推理，默认为全部数据集。

评估所需ckpt获取网址：[获取链接](https://mindspore.cn/resources/hub/details?MindSpore/1.7/midas_redweb)

修改config.yaml中model_weights和ckpt_path参数为下载的ckpt文件名；device_target为GPU；file_format为ONNX

修改完成后运行midas_export.py文件，导出ONNX文件，应该导出3个不同输入大小的ONNX文件对应三个数据集。

```bash
# ONNX评估
bash run_eval_onnx.sh [DEVICE_ID] [DATA_NAME] [DEVICE_TARGET]
```

推理结果保存在脚本执行的当前路径，你可以在onnx_val.json中查看精度计算结果。

# 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

## 导出MindIR

```shell
python midas_export.py --img_width [IMG_WIDTH] --img_height [IMG_HEIGHT]
```

参数在config.yaml文件中设置,其中img——width和img_height分别取160，384；384，1280；288，384时分别对应Sintel，Kitti，TUM数据集

### 在Ascend310执行推理

在执行推理前，mindir文件必须通过`midas_export.py`脚本导出。以下展示了使用mindir模型执行推理的示例。

```shell
# Ascend310 inference
bash run_infer_310.sh [MODEL_PATH] [DATA_PATH] [DATASET_NAME] [DEVICE_ID]
```

- `MODEL_PATH` mindir文件路径
- `DATA_PATH` 推理数据集路径
- `DATASET_NAME` 推理数据集名称，名称分别为Kitti，TUM，Sintel。
- `DEVICE_ID` 可选，默认值为0。

### 结果

推理结果保存在脚本执行的当前路径，你可以在result_val.json中看到以下精度计算结果。

```text
{"Kitti": 18.27 "Sintel":0.314 "TUM":13.27 }
```

# 模型描述

## 性能

### 评估性能

#### ReDWeb上性能参数

| Parameters          | Ascend 910                   |V100-PCIE                   |
| ------------------- | --------------------------- |--------------------------- |
| 模型版本       | Midas               | Midas               |
| 资源            | Ascend 910；CPU：2.60GHz，192核；内存：755G                  | Tesla V100-PCIE 32G ， cpu  52cores 2.60GHz，RAM 754G                 |
| 上传日期       | 2021-06-24 |2021-11-30|
| MindSpore版本   | 1.2.0                       |1.6.0.20211125|
| 数据集             | ReDWeb                  |ReDWeb                  |
| 预训练模型            | ResNeXt_101_WSL                  |ResNeXt_101_WSL                  |
| 训练参数 | epoch=400, batch_size=8, no_backbone_lr=1e-4,backbone_lr=1e-5   | epoch=400, batch_size=8, no_backbone_lr=1e-4,backbone_lr=1e-5   |
| 优化器           | Adam                        |Adam                        |
| 损失函数       | 自定义损失函数          | 自定义损失函数          |
| 速度               | 8pc: 423.4 ms/step        |8pc: 920 ms/step  1pc:655ms/step      |
| 训练性能   | "Kitti": 24.222 "Sintel":0.323  "TUM":15.08    |"Kitti"：23.870 "sintel": 0.322569 "TUM": 16.198  |

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。
