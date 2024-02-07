# Contents

* [Contents](#contents)
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

# [ConvMixer Description](#contents)

ConvMixer is an extremely simple model that is similar in spirit to the ViT
and the even-more-basic MLP-Mixer in that it operates directly on patches
as input, separates the mixing of spatial and channel dimensions,
and maintains equal size and resolution throughout the network.
In contrast, however, the ConvMixer uses only standard convolutions
to achieve the mixing steps. Despite its simplicity, the ConvMixer outperforms
the ViT, MLP-Mixer, and some of their variants for similar parameter
counts and data set sizes, in addition to outperforming classical
vision models such as the ResNet.

[Paper](https://openreview.net/pdf?id=TVHS5Y4dNvM): Asher Trockman, J Zico Kolter. 2022.

# [Model Architecture](#contents)

ConvMixer consists of a patch embedding layer followed by repeated
applications of a simple fully-convolutional block (ConvMixer block).
The ConvMixer block itself consists of depthwise convolution
(i.e., grouped convolution with groups equal to the number of channels)
followed by pointwise (i.e., kernel size 1 × 1) convolution.
Each of the convolutions is followed by an activation and post-activation
BatchNorm. After many applications of this block, global pooling is performed
to get a feature vector, which is passed to a softmax classifier.

There are following "most interesting" configurations:

* ConvMixer 768/32
* ConvMixer 1024/20
* ConvMixer 1536/20

Here the naming is follow the rule `Convixer h/d` where `h` is embedding dimension
and `d` is depth, or the number of repetitions of the ConvMixer block.

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

1. Chose the model by changing the `arch` in `configs/convmixer_XXX.yaml`, `XXX` is the corresponding model architecture configuration.
   Allowed options are: `convmixer_768_32`, `convmixer_1024_20`, `convmixer_1536_20`.
2. Change the dataset config in the corresponding config. `configs/convmixer_XXX.yaml`.
   Especially, set the correct path to data.
3. Change the hardware setup.
4. Change the artifacts setup to set the correct folders to save checkpoints and mindinsight logs.

Note, that you also can pass the config options as CLI arguments, and they are
preferred over config in YAML.

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
convmixer
├── scripts
│   ├── run_distribute_train_gpu.sh                          # shell script for distributed training on GPU
│   ├── run_eval_gpu.sh                                      # shell script for evaluation on GPU
│   ├── run_eval_onnx.sh                                     # shell script for evaluation with ONNX model
│   ├── run_infer_gpu.sh                                     # shell script for inference on GPU
│   ├── run_infer_onnx.sh                                    # shell script for inference with ONNX model
│   └── run_standalone_train_gpu.sh                          # shell script for training on GPU
├── src
│  ├── configs
│  │  ├── convmixer_1024_20.yaml                             # example of configuration for ConvMixer-1024/20
│  │  ├── convmixer_1536_20.yaml                             # example of configuration for ConvMixer-1536/20
│  │  └── convmixer_768_32.yaml                              # example of configuration for ConvMixer-768/32
│  ├── data
│  │  ├── augment
│  │  │  ├── __init__.py
│  │  │  ├── auto_augment.py                                 # augmentation set builder
│  │  │  ├── mixup.py                                        # MixUp augmentation
│  │  │  └── random_erasing.py                               # Random Erasing augmentation
│  │  ├── __init__.py
│  │  └── imagenet.py                                        # wrapper for reading ImageNet dataset
│  ├── tools
│  │  ├── __init__.py
│  │  ├── callback.py                                        # callback functions (implementation)
│  │  ├── cell.py                                            # tune model layers/parameters
│  │  ├── criterion.py                                       # model training objective function (implementation)
│  │  ├── get_misc.py                                        # initialize optimizers and other arguments for training process
│  │  ├── optimizer.py                                       # model optimizer function (implementation)
│  │  └── schedulers.py                                      # training (LR) scheduling function (implementation)
│  ├── trainer
│  │  ├── ema.py                                             # EMA implementation
│  │  ├── train_one_step_with_ema.py                         # utils for training with EMA
│  │  └── train_one_step_with_scale_and_clip_global_norm.py  # utils for training with gradient clipping
│  ├── config.py                                             # YAML and CLI configuration parser
│  └── convmixer.py                                          # ConvMixer architecture
├── eval.py                                                  # evaluation script
├── eval_onnx.py                                             # evaluation script for ONNX model
├── export.py                                                # export checkpoint files into MINDIR, ONNX and AIR formats
├── infer.py                                                 # inference script
├── infer_onnx.py                                            # inference script for ONNX model
├── README.md                                                # ConvMixer descriptions
├── requirements.txt                                         # python requirements
└── train.py                                                 # training script
```

### [Script Parameters](#contents)

```yaml
arch: convmixer_768_32

# ===== Dataset ===== #
data_url: /imagenet/ILSVRC/Data/CLS-LOC/
val_dir: validation_preprocess
set: ImageNet
num_classes: 1000

auto_augment: rand-m9-mstd0.5-inc1
interpolation: random
re_prob: 0.25
re_mode: pixel
re_count: 1
cutmix: 0.5
mixup: 0.5
mixup_prob: 1.0
mixup_mode: batch
mixup_off_epoch: 0.0
switch_prob: 0.5
min_crop: 0.875
image_size: 224

# ===== Learning Rate Policy ======== #
optimizer: adamw
base_lr: 0.001
warmup_lr: 0.000007
min_lr: 0.0001
lr_scheduler: cosine_lr
warmup_length: 0

# ===== Network training config ===== #
amp_level: O0
keep_bn_fp32: True
beta: [ 0.9, 0.999 ]
is_dynamic_loss_scale: False
epochs: 300
label_smoothing: 0.1
weight_decay: 0.0001
momentum: 0.9
batch_size: 24

# ===== Gradient clipping ===== #
use_clip_grad_norm: True
clip_grad_norm: 1.0

# ===== Hardware setup ===== #
num_parallel_workers: 4
device_target: GPU

# ===== Artifacts setup ===== #
summary_root_dir: /mindspore/summary_dir/
ckpt_root_dir: /mindspore/checkpoints/
best_ckpt_root_dir: /mindspore/best_checkpoints/
ckpt_keep_num: 10
ckpt_save_every_seconds: 1800
print_loss_every: 50
summary_loss_collect_freq: 10
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
bash run_distribute_train.sh ../src/configs/convmixer_768_32.yaml --num_devices 4 --device_ids 0,1,2,3

# With extra arguments
bash run_distribute_train.sh ../src/configs/convmixer_768_32.yaml --num_devices 4 --device_ids 0,1,2,3 --extra --amp_level O0 --batch_size 48 --start_epoch 0 --num_parallel_workers 8
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
bash run_standalone_train.sh ../src/configs/convmixer_768_32.yaml --device 0
# With extra arguments:
bash run_standalone_train.sh ../src/configs/convmixer_768_32.yaml --device 0 --extra --amp_level O0 --batch_size 48 --start_epoch 0 --num_parallel_workers 8
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
bash run_eval_gpu.sh  ../src/configs/convmixer_768_32.yaml --checkpoint /data/models/convmixer_768_32.ckpt

# With extra args
bash run_eval_gpu.sh  ../src/configs/convmixer_768_32.yaml --checkpoint /data/models/convmixer_768_32.ckpt --extra --data_url /data/imagenet/ --val_dir validation_preprocess
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
python eval_onnx.py /data/imagenet/val --onnx_path convmixer_768_32.onnx
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
bash run_infer_gpu.sh /data/images/cheetah/ --checkpoint /data/models/convmixer_768_32_trained.ckpt --arch convmixer_768_32
```

Example for single image

```shell
bash run_infer_gpu.sh /data/images/cheetah/ILSVRC2012_validation_preprocess_00001060.JPEG --checkpoint /data/models/convmixer_768_32_trained.ckpt --arch convmixer_768_32
```

### Inference with ONNX

Usage

```bash
run_infer_onnx.sh DATA [--onnx_path ONNX_PATH] [--output OUTPUT_JSON_FILE (default: predictions.json)]
```

Example

```bash
bash run_infer_onnx.sh /data/images/cheetah/ --onnx_path /data/models/convmixer_768_32.onnx
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
python export.py --config src/configs/convmixer_768_32.yaml --file_format MINDIR --pretrained /data/models/convmixer_768_32.ckpt --arch convmixer_768_32

# Export to ONNX
python export.py --config src/configs/convmixer_768_32.yaml --file_format ONNX --pretrained /data/models/convmixer_768_32.ckpt --arch convmixer_768_32
```

### Export results

Exported models saved in the current directory with name the same as architecture.

# [Model Description](#contents)

## Performance

### Training Performance

| Parameters                 | GPU                                |
|----------------------------|------------------------------------|
| Model Version              | ConvMixer-768/32                   |
| Resource                   | 4xGPU (NVIDIA GeForce RTX 3090)    |
| Uploaded Date              | 12/26/2023 (month/day/year)        |
| MindSpore Version          | 1.9.0                              |
| Dataset                    | ImageNet                           |
| Training Parameters        | src/configs/convmxier_768_32 .yaml |
| Optimizer                  | AdamW                              |
| Loss Function              | SoftmaxCrossEntropy                |
| Outputs                    | logits                             |
| Accuracy                   | ACC1 [0.731066]                    |
| Total time                 | ~620 h                             |
| Params (M)                 | 21110248                           |
| Checkpoint for Fine tuning | 84.9 M                             |
| Scripts                    |                                    |

# [Description of Random Situation](#contents)

We use fixed seed in train.py.

# [ModelZoo Homepage](#contents)

Please check the official [homepage](https://gitee.com/mindspore/models).
