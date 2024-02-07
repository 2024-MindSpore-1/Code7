# Contents

* [Contents](#contents)
* [Conformer Description](#conformer-description)
* [Model Architecture](#model-architecture)
* [Dataset](#dataset)
* [Environment Requirements](#environment-requirements)
* [Quick Start](#quick-start)
    * [Prepare the model](#prepare-the-model)
    * [Run the scripts](#run-the-scripts)
* [Script Description](#script-description)
    * [Script and Sample Code](#script-and-sample-code)
        * [Directory structure](#directory-structure)
        * [Script Parameters](#script-parameters)
    * [Training Process](#training-process)
        * [Training on GPU](#training-on-gpu)
            * [Training on multiple GPUs](#training-on-multiple-gpus)
            * [Training on single GPU](#training-on-single-gpu)
            * [Arguments description](#arguments-description)
        * [Training with CPU](#training-with-cpu)
        * [Transfer training](#transfer-training)
    * [Evaluation](#evaluation)
        * [Evaluation process](#evaluation-process)
            * [Evaluation with checkpoint](#evaluation-with-checkpoint)
            * [Evaluation with ONNX](#evaluation-with-onnx)
        * [Evaluation results](#evaluation-results)
    * [Inference](#inference)
        * [Inference with checkpoint](#inference-with-checkpoint)
        * [Inference with ONNX](#inference-with-onnx)
        * [Inference results](#inference-results)
    * [Export](#export)
        * [Export process](#export-process)
        * [Export results](#export-results)
* [Model Description](#model-description)
    * [Performance](#performance)
        * [Training Performance](#training-performance)
* [Description of Random Situation](#description-of-random-situation)
* [ModelZoo Homepage](#modelzoo-homepage)

# [Conformer Description](#contents)

Within Convolutional Neural Network (CNN), the convolution operations are good
at extracting local features but experience difficulty to capture global
representations. Within visual transformer, the cascaded self-attention
modules can capture long-distance feature dependencies but unfortunately
deteriorate local feature details. In this paper, we propose a hybrid network
structure, termed Conformer, to take advantage of convolutional operations
and self-attention mechanisms for enhanced representation learning.
Conformer roots in the Feature Coupling Unit (FCU), which fuses local features
and global representations under different resolutions in an interactive fashion.
Conformer adopts a concurrent structure so that local features and global
representations are retained to the maximum extent.

[Paper](https://arxiv.org/pdf/2105.03889.pdf): Zhiliang Peng, Wei Huang, Shanzhi Gu,
Lingxi Xie, Yaowei Wang, Jianbin Jiao, Qixiang Ye. 2021.

# [Model Architecture](#contents)

Conformer consists of two brunches: CNN and Transformer. CNN collects local
features in a hierarchical manner via convolutional operations and retains
the local cues as feature maps. Visual transformer is believed to aggregate
global representations among the compressed patch embeddings in a soft fashion
by the cascaded self-attention modules. So, Conformer has a concurrent network structure.

Considering the complementarity of the two-style features, within Conformer,
the global context consecutively is fed from the transformer branch to feature
maps, to reinforce the global perception capability of the CNN branch.
Similarly, local features from the CNN branch are progressively fed back to
patch embeddings, to enrich the local details of the transformer branch.
Such a process constitutes the interaction.

In special, Conformer is composed of a stem module, dual branches,
FCUs to bridge dual branches, and two classifiers (a fc layer) for the dual
branches. The stem module, which is a 7×7 convolution with stride 2 followed
by a 3×3 max pooling with stride 2, is used to extract initial local features
(e.g., edge and texture information), which are then fed to the dual branches.
The CNN branch and transformer branch are composed of N (e.g., 12) repeated
convolution and transformer blocks, respectively. Such a concurrent structure
implies that CNN and transformer branch can respectively preserve the local
features and global representations to the maximum extent. FCU is proposed as
a bridge module to fuse local features in the CNN branch with global representations
in the transformer branch. FCU is applied from the second block because the
initialized features of the two branches are the same. Along the branches,
FCU progressively fuses feature maps and patch embeddings in an interactive fashion.

Finally, for the CNN branch, all the features are pooled
and fed to one classifier. For the transformer branch, the
class token is taken out and fed to the other classifier. Dur-
ing training, we use two cross entropy losses to separately
supervise the two classifiers. The importance of the loss
functions are empirically set to be same. During inference,
the outputs of the two classifiers are simply summarized as
the prediction results.

There are following model variants:

* Conformer Tiny
* Conformer Small
* Conformer Base

# [Dataset](#contents)

Note that you can run the scripts based on the dataset mentioned in original
paper or widely used in relevant domain/network architecture. In the following
sections, we will introduce how to run the scripts using the related dataset
below.

Dataset used: [ImageNet2012](http://www.image-net.org/)

* Dataset size：146.6G
    * Train：139.3G，1281167 images
    * Val：6.3G，50000 images
    * Annotations：each image is in label folder
* Data format：images sorted by label folders
    * Note：Data will be processed in imagenet.py

# [Environment Requirements](#contents)

* Install [MindSpore](https://www.mindspore.cn/install/en).
* Download the dataset ImageNet dataset.
* We use ImageNet2012 as training dataset in this example by default, and you
  can also use your own datasets.

For ImageNet-like dataset the directory structure is as follows:

```shell
 .
 └─imagenet
   ├─train
     ├─class1
       ├─image1.jpeg
       ├─image2.jpeg
       └─...
     ├─...
     └─class1000
   ├─val
     ├─class1
     ├─...
     └─class1000
   └─test
```

# [Quick Start](#contents)

## Prepare the model

1. Chose the model by changing the `arch` in `configs/conformer_XXX.yaml`, `XXX` is the corresponding model architecture configuration.
   Allowed options are: `conformer_tiny`, `conformer_small`, `conformer_base`.
2. Change the dataset config in the corresponding config. `configs/conformer_XXX.yaml`.
   Especially, set the correct path to data.
3. Change the hardware setup.
4. Change the artifacts setup to set the correct folders to save checkpoints and mindinsight logs.

Note, that you also can pass the config options as CLI arguments, and they are
preferred over config in YAML.
Also, all possible options must be defined in `yaml` config file.

## Run the scripts

After installing MindSpore via the official website,
you can start training and evaluation as follows.

```shell
# distributed training on GPU
bash run_distribute_train_gpu.sh CONFIG [--num_devices NUM_DEVICES] [--device_ids DEVICE_IDS (e.g. '0,1,2,3')] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]

# standalone training on GPU
bash run_standalone_train_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]

# run eval on GPU
bash run_eval_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

# [Script Description](#contents)

## [Script and Sample Code](#contents)

### Directory structure

```shell
conformer
├── scripts
│   ├── run_distribute_train_gpu.sh                          # shell script for distributed training on GPU
│   ├── run_eval_gpu.sh                                      # shell script for evaluation on GPU
│   ├── run_eval_onnx.sh                                     # shell script for evaluation with ONNX model
│   ├── run_infer_gpu.sh                                     # shell script for inference on GPU
│   ├── run_infer_onnx.sh                                    # shell script for inference with ONNX model
│   └── run_standalone_train_gpu.sh                          # shell script for training on GPU
├── src
│  ├── configs
│  │  ├── conformer_tiny.yaml                             # example of configuration for Conformer Tiny
│  │  ├── conformer_small.yaml                             # example of configuration for Conformer Small
│  │  └── conformer_base.yaml                              # example of configuration for Conformer Base
│  ├── data
│  │  ├── augment
│  │  │  ├── __init__.py
│  │  │  ├── auto_augment.py                                 # augmentation set builder
│  │  │  ├── mixup.py                                        # MixUp augmentation
│  │  │  └── random_erasing.py                               # Random Erasing augmentation
│  │  ├── __init__.py
│  │  └── imagenet.py                                        # wrapper for reading ImageNet dataset
│  ├── layers                                                # layers used in Conformer implementation
│  │  ├── __init__.py
│  │  ├── attention.py
│  │  ├── avg_pool.py
│  │  ├── block.py
│  │  ├── conv_block.py
│  │  ├── convtransblock.py
│  │  ├── custom_identity.py
│  │  ├── drop_path_timm.py
│  │  ├── fcu.py
│  │  ├── med_convblock.py
│  │  └── mlp.py
│  ├── tools
│  │  ├── __init__.py
│  │  ├── callback.py                                        # callback functions (implementation)
│  │  ├── cell.py                                            # tune model layers/parameters
│  │  ├── criterion.py                                       # model training objective function (implementation)
│  │  ├── get_misc.py                                        # initialize optimizers and other arguments for training process
│  │  ├── optimizer.py                                       # model optimizer function (implementation)
│  │  └── schedulers.py                                      # training (LR) scheduling function (implementation)
│  ├── trainer
│  │  ├── __init__.py
│  │  ├── ema.py                                             # EMA implementation
│  │  ├── train_one_step_with_ema.py                         # utils for training with EMA
│  │  └── train_one_step_with_scale_and_clip_global_norm.py  # utils for training with gradient clipping
│  ├── config.py                                             # YAML and CLI configuration parser
│  └── conformer.py                                          # Conformer architecture
├── eval.py                                                  # evaluation script
├── eval_onnx.py                                             # evaluation script for ONNX model
├── export.py                                                # export checkpoint files into MINDIR, ONNX and AIR formats
├── infer.py                                                 # inference script
├── infer_onnx.py                                            # inference script for ONNX model
├── README.md                                                # Conformer descriptions
├── requirements.txt                                         # python requirements
└── train.py                                                 # training script
```

### [Script Parameters](#contents)

```yaml
# ===== Dataset ===== #
dataset: ImageNet
data_url: /data/imagenet/ILSVRC/Data/CLS-LOC/
train_dir: train
val_dir: validation_preprocess
train_num_samples: -1
val_num_samples: -1

# ===== Augmentations ==== #
auto_augment: rand-m9-mstd0.5-inc1
aa_interpolation: bilinear
re_mode: pixel
re_prob: 0.25
re_count: 1
cutmix: 1.0
mixup: 0.8
mixup_prob: 1.0
mixup_mode: batch
mixup_off_epoch: 0.0
switch_prob: 0.5
label_smoothing: 0.1
min_crop: 0.08
crop_pct: 0.96

# ===== Optimizer ======== #
optimizer: adamw
beta: [ 0.9, 0.999 ]
eps: 1.0e-8
base_lr: 0.001
min_lr: 1.0e-5
lr_scheduler: cosine_lr
lr_adjust: 30
lr_gamma: 0.97
momentum: 0.9
weight_decay: 0.05


# ===== Network training config ===== #
epochs: 300
batch_size: 100
is_dynamic_loss_scale: True
loss_scale: 1024
num_parallel_workers: 8
start_epoch: 0
warmup_length: 2
warmup_lr: 0.000007
# Gradient clipping
use_clip_grad_norm: True
clip_grad_norm: 1.0
# Load pretrained setup
exclude_epoch_state: True
seed: 0
# EMA
with_ema: False
ema_decay: 0.9999
pynative_mode: False

# ==== Model arguments ==== #
arch: conformer_tiny_patch16
amp_level: O0
file_format: ONNX
pretrained: ''
image_size: 224
num_classes: 1000
drop: 0.0
drop_block: 0.0
drop_path: 0.1
disable_approximate_gelu: False
use_pytorch_maxpool: False

# ===== Hardware setup ===== #
device_id: 0
device_num: 1
device_target: GPU

# ===== Callbacks setup ===== #
summary_root_dir: /mindspore/summary_dir/
ckpt_root_dir: /mindspore/checkpoints/
best_ckpt_root_dir: /mindspore/best_checkpoints/
ckpt_keep_num: 10
best_ckpt_num: 5
ckpt_save_every_step: 0
ckpt_save_every_seconds: 1800
print_loss_every: 100
summary_loss_collect_freq: 20
model_postfix: 0
collect_input_data: False
dump_graph: False
```

## [Training Process](#contents)

In the examples below the only required argument is YAML config file.

### Training on GPU

#### Training on multiple GPUs

Usage

```shell
# distributed training on GPU
run_distribute_train_gpu.sh CONFIG [--num_devices NUM_DEVICES] [--device_ids DEVICE_IDS (e.g. '0,1,2,3')] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Example

```bash
# Without extra arguments
bash run_distribute_train.sh ../src/configs/conformer_tiny.yaml --num_devices 4 --device_ids 0,1,2,3

# With extra arguments
bash run_distribute_train.sh ../src/configs/conformer_tiny.yaml --num_devices 4 --device_ids 0,1,2,3 --extra --amp_level O0 --batch_size 48 --start_epoch 0 --num_parallel_workers 8
```

#### Training on single GPU

Usage

```shell
# standalone training on GPU
run_standalone_train_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Example

```bash
# Without extra arguments:
bash run_standalone_train.sh ../src/configs/conformer_tiny.yaml --device 0
# With extra arguments:
bash run_standalone_train.sh ../src/configs/conformer_tiny.yaml --device 0 --extra --amp_level O0 --batch_size 48 --start_epoch 0 --num_parallel_workers 8
```

Running the Python scripts directly is also allowed.

```shell
# show help with description of options
python train.py --help

# standalone training on GPU
python train.py --config path/to/config.yaml [OTHER OPTIONS]
```

#### Arguments description

`bash` scripts have the following arguments

* `CONFIG`: path to YAML file with configuration.
* `--num_devices`: the device number for distributed train.
* `--device_ids`: ids of devices to train.
* `--checkpoint`: path to checkpoint to continue training from.
* `--extra`: any other arguments of `train.py`.

By default, training process produces three folders (configured):

* Best checkpoints
* Current checkpoints
* Mindinsight logs

### Training with CPU

**It is recommended to run models on GPU.**

### Transfer training

You can train your own model based on pretrained classification
model. You can perform transfer training by following steps.

1. Convert your own dataset to ImageFolderDataset style. Otherwise, you have to add your own data preprocess code.
2. Change `config_XXX.yaml` according to your own dataset, especially the `num_classes`.
3. Prepare a pretrained checkpoint. You can load the pretrained checkpoint by `pretrained` argument.
4. Build your own bash scripts using new config and arguments for further convenient.

## [Evaluation](#contents)

### Evaluation process

#### Evaluation with checkpoint

Usage

```shell
run_eval_gpu.sh CONFIG [--device DEVICE_ID (default: 0)] [--checkpoint CHECKPOINT] [--extra *EXTRA_ARGS]
```

Examples

```shell

# Without extra args
bash run_eval_gpu.sh  ../src/configs/conformer_tiny.yaml --checkpoint /data/models/conformer_tiny.ckpt

# With extra args
bash run_eval_gpu.sh  ../src/configs/conformer_tiny.yaml --checkpoint /data/models/conformer_tiny.ckpt --extra --data_url /data/imagenet/ --val_dir validation_preprocess
```

Running the Python script directly is also allowed.

```shell
# run eval on GPU
python eval.py --config path/to/config.yaml [OTHER OPTIONS]
```

The Python script has the same arguments as the training script (`train.py`),
but it uses only validation subset of dataset to evaluate.
Also, `--pretrained` is expected.

#### Evaluation with ONNX

Usage

```shell
run_eval_onnx.sh DATA [--onnx_path ONNX_PATH]
```

* `DATA` is a test subset
* `--onnx_path` is path to ONNX model.

Example

```bash
bash run_eval_onnx.sh /data/imagenet/val --onnx_path /data/models/onvmixer_768_32.onnx
```

Also, Python script may be used.

Usage

```shell
eval_onnx.py [-h] [--onnx_path ONNX_PATH] [--image_size IMAGE_SIZE (default: 224)]
             [-m {CPU,GPU} (default: GPU)] [--prefetch PREFETCH (DEFAULT: 16)]
             dataset
```

Example

```shell
python eval_onnx.py /data/imagenet/val --onnx_path conformer_tiny.onnx
```

### Evaluation results

Results will be printed to console.

```shell
# checkpoint evaluation result
eval results: {'Loss': 1.4204, 'Top1-Acc': 0.731066, 'Top5-Acc': 0.90621}

# ONNX evaluation result
eval results: {'Top1-Acc': 0.731066, 'Top5-Acc': 0.90621}
```

## [Inference](#contents)

Inference may be performed with checkpoint or ONNX model.

### Inference with checkpoint

Usage

```shell
run_infer_gpu.sh DATA [--checkpoint CHECKPOINT] [--arch ARCHITECTURE] [--output OUTPUT_JSON_FILE (default: predictions.json)]
```

Example for folder

```shell
bash run_infer_gpu.sh /data/images/cheetah/ --checkpoint /data/models/conformer_tiny_trained.ckpt --arch conformer_tiny
```

Example for single image

```shell
bash run_infer_gpu.sh /data/images/cheetah/ILSVRC2012_validation_preprocess_00001060.JPEG --checkpoint /data/models/conformer_tiny_trained.ckpt --arch conformer_tiny
```

### Inference with ONNX

Usage

```bash
run_infer_onnx.sh DATA [--onnx_path ONNX_PATH] [--output OUTPUT_JSON_FILE (default: predictions.json)]
```

Example

```bash
bash run_infer_onnx.sh /data/images/cheetah/ --onnx_path /data/models/conformer_tiny.onnx
```

### Inference results

Predictions will be output in logs and saved in JSON file. Predictions format
is same for mindspore and ONNX model File content is dictionary where key is
image path and value is class number. It's supported predictions for folder of
images (png, jpeg file in folder root) and single image.

Results for single image in console

```shell
/data/images/cheetah/ILSVRC2012_validation_preprocess_00001060.JPEG (class: 293)
```

Results for single image in JSON file

```json
{
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00001060.JPEG": 293
}
```

Results for directory in console

```shell
/data/images/cheetah/ILSVRC2012_validation_preprocess_00033907.JPEG (class: 293)
/data/images/cheetah/ILSVRC2012_validation_preprocess_00033988.JPEG (class: 293)
/data/images/cheetah/ILSVRC2012_validation_preprocess_00013656.JPEG (class: 293)
/data/images/cheetah/ILSVRC2012_validation_preprocess_00038707.JPEG (class: 293)
```

Results for directory in JSON file

```json
{
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00033907.JPEG": 293,
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00033988.JPEG": 293,
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00013656.JPEG": 293,
 "/data/images/cheetah/ILSVRC2012_validation_preprocess_00038707.JPEG": 293
}
```

## [Export](#contents)

### Export process

Trained checkpoints may be exported to `MINDIR`, `AIR` (currently not checked) and `ONNX`.
When exporting to `ONNX` the results may slightly differ because `AdaptiveAvgPool2d`
is not supported in the current version of `ONNX` and we use wrapper
that implements this layer with `ms.ops.ReduceMean` operation.

Usage

```shell
python export.py --config path/to/config.yaml --file_format FILE_FORMAT --pretrained path/to/checkpoint.ckpt --arch ARCHITECTURE_NAME
```

Example

```shell

# Export to MINDIR
python export.py --config src/configs/conformer_tiny.yaml --file_format MINDIR --pretrained /data/models/conformer_tiny.ckpt --arch conformer_tiny

# Export to ONNX
python export.py --config src/configs/conformer_tiny.yaml --file_format ONNX --pretrained /data/models/conformer_tiny.ckpt --arch conformer_tiny
```

### Export results

Exported models saved in the current directory with name the same as architecture.

# [Model Description](#contents)

## Performance

### Training Performance

| Parameters                 | GPU                             |
|----------------------------|---------------------------------|
| Model Version              | Connformer_tiny_patch16         |
| Resource                   | 4xGPU (NVIDIA GeForce RTX 3090) |
| Uploaded Date              | 07/11/2023 (month/day/year)     |
| MindSpore Version          | 1.9.0                           |
| Dataset                    | ImageNet                        |
| Training Parameters        | src/configs/conformer_tiny.yaml |
| Optimizer                  | AdamW                           |
| Loss Function              | SoftmaxCrossEntropy             |
| Outputs                    | logits                          |
| Accuracy                   | ACC1 [0.586]                    |
| Total time                 | ~1252.3 h                       |
| Params                     | 23008536                        |
| Checkpoint for Fine tuning | 285.8 M                         |
| Scripts                    |                                 |

# [Description of Random Situation](#contents)

We use fixed seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
