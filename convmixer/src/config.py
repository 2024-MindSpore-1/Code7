# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the 'License');
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
"""Global args for ConvMixer."""
import argparse
import os
import sys

import yaml

args = None


def _add_augmentation_arguments(parser: argparse.ArgumentParser):
    """Add arguments related to augmentations to parser."""
    parser.add_argument(
        '--re_mode',
        type=str,
        default='pixel',
        choices=['const', 'rand', 'pixel'],
        help='Random erase mode.'
    )
    parser.add_argument(
        '--re_prob',
        type=float,
        default=0.0,
        help='Random erase prob.'
    )
    parser.add_argument(
        '--re_count',
        type=int,
        default=1,
        help='Maximum number of erasing blocks per image, area per box is '
             'scaled by count. per-image count is randomly chosen between '
             '1 and this value.'
    )
    parser.add_argument(
        '--cutmix',
        type=float,
        default=0.0,
        help='Cutmix alpha, cutmix enabled if > 0.'
    )
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.0,
        help='Mixup alpha, mixup enabled if > 0.'
    )
    parser.add_argument(
        '--mixup_off_epoch',
        type=float,
        default=0.0,
        help='Use mixup during training.'
    )
    parser.add_argument(
        '--mixup_mode',
        type=str,
        default='batch',
        choices=['batch', 'pair', 'elem'],
        help='How to apply mixup/cutmix params (per batch, pair '
             '(pair of elements), elem (element).'
    )
    parser.add_argument(
        '--switch_prob',
        type=float,
        default=0.5,
        help='Probability of switching to cutmix instead of mixup when'
             'both are active.'
    )
    parser.add_argument(
        '--auto_augment',
        '--aa',
        type=str,
        default='rand-m9-mstd0.5-inc1',
        help='Use AutoAugment policy.'
    )
    parser.add_argument(
        '--aa_interpolation',
        type=str,
        choices=['bicubic', 'lanczos', 'hamming', 'bilinear'],
        default='bilinear',
        help='Interpolation method for auto-augmentations.'
    )
    parser.add_argument(
        '--label_smoothing',
        type=float,
        help='Label smoothing to use, default 0.1',
        default=0.1
    )
    parser.add_argument(
        '--min_crop',
        type=float,
        default=0.08,
        help='Min random resize scale.'
    )
    parser.add_argument(
        '--crop_pct',
        type=float,
        default=None,
        help='Coef to scale image before center crop in val preprocessing.'
    )


def _add_callback_arguments(parser: argparse.ArgumentParser):
    """Add arguments related to callbacks to parser."""
    parser.add_argument(
        '--summary_root_dir',
        type=str,
        default='./summary_dir/',
        help='Root directory to write summary for Mindinsight.'
    )
    parser.add_argument(
        '--ckpt_root_dir',
        type=str,
        default='./checkpoints/',
        help='Root directory to write checkpoints during training.'
    )
    parser.add_argument(
        '--best_ckpt_root_dir',
        type=str,
        default='./best_checkpoints/',
        help='Root directory to write best checkpoints during training.'
    )
    parser.add_argument(
        '--ckpt_keep_num',
        type=int,
        default=10,
        help='Keep last N checkpoints.'
    )
    parser.add_argument(
        '--ckpt_save_every_step',
        default=0,
        type=int,
        help='Save every N steps. To use saving by time set this to 0 and use '
             '--ckpt_save_every_sec option.'
             'If both are set `step` is preferred.'
    )
    parser.add_argument(
        '--ckpt_save_every_sec',
        default=3600,
        type=int,
        help='Save every N seconds. To use saving by steps set this to 0 and'
             ' use --ckpt_save_every_step option. '
             'If both are set `step` is preferred.'
    )
    parser.add_argument(
        '--model_postfix',
        type=str,
        default='',
        help='Some additional information about the model that will be added '
             'to summary and checkpoints directories names.'

    )
    parser.add_argument(
        '--collect_input_data', action='store_true',
        help='Flag to control collecting input data during training. '
             'Important: data us collected only if dataset_sink_mode is False.'
    )
    parser.add_argument(
        '--print_loss_every', type=int, default=20,
        help='Print loss every step.'
    )
    parser.add_argument(
        '--summary_loss_collect_freq',
        type=int,
        default=1,
        help='Frequency of collecting loss while training.'
    )
    parser.add_argument(
        '--dump_graph',
        action='store_true',
    )


def _add_dataset_arguments(parser: argparse.ArgumentParser):
    """Add arguments related to dataset to parser."""
    parser.add_argument(
        '--set', type=str, default='ImageNet', help='Name of dataset'
    )
    parser.add_argument(
        '--data_url',
        type=str,
        default='./data',
        help='Location of the root data directory.'
    )
    parser.add_argument(
        '--train_dir',
        type=str,
        default='train',
        help='Name of directory which contains training subset',
    )
    parser.add_argument(
        '--val_dir',
        type=str,
        default='val',
        help='Name of directory which contains validation subset',
    )

    parser.add_argument(
        '--train_num_samples',
        type=int,
        default=None,
        help='Number of samples taken from training dataset. '
             'If not set, then all data is used.'
    )
    parser.add_argument(
        '--val_num_samples',
        type=int,
        default=None,
        help='Number of samples taken from validation dataset. '
             'If not set, then all data is used.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        metavar='N',
        help='Mini-batch size, this is the total batch size of all Devices '
             'on the current node when using Data Parallel '
             'or Distributed Data Parallel.'
    )


def _add_optimizer_arguments(parser: argparse.ArgumentParser):
    """Add arguments related to optimizer to parser."""
    parser.add_argument(
        '--optimizer', default='adamw', help='Which optimizer to use.'
    )
    parser.add_argument(
        '--beta',
        default=[0.9, 0.999],
        type=lambda x: [float(a) for a in x.split(',')],
        help='Beta for optimizer.'
    )
    parser.add_argument(
        '--eps', default=1e-8, type=float, help='Optimizer eps.'
    )
    parser.add_argument(
        '--base_lr', default=0.01, type=float, help='Learning rate.',
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default='cosine_lr',
        choices=[
            'multistep_lr', 'cosine_lr', 'constant_lr', 'exp_lr', 'onecycle'
        ],
        help='Schedule for the learning rate.'
    )
    parser.add_argument(
        '--lr_adjust', default=30, type=float, help='Interval to drop lr.'
    )
    parser.add_argument(
        '--lr_gamma', default=0.97, type=int, help='Multistep multiplier.'
    )
    parser.add_argument(
        '--momentum', default=0.9, type=float, metavar='M', help='Momentum.'
    )
    parser.add_argument(
        '--wd',
        '--weight_decay',
        default=0.05,
        type=float,
        metavar='W',
        help='Weight decay.',
        dest='weight_decay'
    )


def _add_training_arguments(parser: argparse.ArgumentParser):
    """Add common training arguments to parser."""
    parser.add_argument(
        '--epochs', default=300, type=int, metavar='N',
        help='Number of total epochs to run.'
    )

    parser.add_argument(
        '--is_dynamic_loss_scale',
        default=1,
        type=int,
        help='Use Dynamic Loss scale update cell.'
    )

    parser.add_argument(
        '-j',
        '--num_parallel_workers',
        default=8,
        type=int,
        metavar='N',
        help='Number of data loading workers.'
    )
    parser.add_argument(
        '--start_epoch',
        default=0,
        type=int,
        metavar='N',
        help='Manual starting epoch number (useful on restarts).'
    )
    parser.add_argument(
        '--warmup_length',
        default=0,
        type=int,
        help='Number of warmup iterations.'
    )
    parser.add_argument(
        '--warmup_lr', default=5e-7, type=float, help='Warm up learning rate.'
    )
    parser.add_argument(
        '--loss_scale',
        default=1024,
        type=int,
        help='The fixed loss scale value used when updating cell with '
             'fixed loss scaling value (not dynamic one).'
    )
    parser.add_argument(
        '--use_clip_grad_norm',
        action='store_true',
        help='Whether to use clip grad norm.'
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        default=1.0,
        help='Clip grad norm value.'
    )
    parser.add_argument(
        '--exclude_epoch_state',
        action='store_true',
        help='Exclude epoch state and learning rate when pretrained is used.'
    )
    parser.add_argument(
        '--seed', default=0, type=int, help='Seed for initializing training.'
    )
    parser.add_argument(
        '--with_ema',
        action='store_true',
        help='Training with ema.'
    )
    parser.add_argument(
        '--ema_decay',
        default=0.9999,
        type=float,
        help='Ema decay.'
    )


def _add_model_arguments(parser: argparse.ArgumentParser):
    """Add arguments related to model to parser."""
    parser.add_argument(
        '-a', '--arch',
        metavar='ARCH',
        default='convmixer_256_3',
        choices=[
            'convmixer_768_32',
            'convmixer_1536_20',
            'convmixer_1024_20',
        ],
        help='Model architecture.'
    )
    parser.add_argument(
        '--amp_level',
        type=str,
        default='O2',
        choices=['O0', 'O1', 'O2', 'O3'],
        help='AMP Level.'
    )
    parser.add_argument(
        '--file_format',
        type=str,
        choices=['AIR', 'MINDIR', 'ONNX'],
        default='ONNX',
        help='File format to export model to.'
    )

    parser.add_argument(
        '--pretrained',
        default=None,
        type=str,
        help='Path to checkpoint to use as a pre-trained model.'
    )
    parser.add_argument(
        '--image_size', default=224, help='Input image size.', type=int
    )
    parser.add_argument(
        '--num_classes', default=1000, type=int, help='Num classes in dataset.'
    )


def _add_device_arguments(parser: argparse.ArgumentParser):
    """Add arguments related to device the model run on to parser."""
    parser.add_argument(
        '--device_id', default=0, type=int, help='Device id.'
    )
    parser.add_argument(
        '--device_num', default=1, type=int, help='Device num.'
    )
    parser.add_argument(
        '--device_target', default='GPU', choices=['GPU', 'Ascend'], type=str
    )


def parse_arguments():
    """
    Create and parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    global args
    parser = argparse.ArgumentParser(
        description='MindSpore ConvMixer Training', add_help=False,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-h', '--help', action='help',
        default=argparse.SUPPRESS,
        help='Show this help message and exit.'
    )
    parser.add_argument(
        '--config',
        help='Config YAML file to use (see configs dir)',
        default=None,
        required=True
    )

    _add_dataset_arguments(parser)
    _add_optimizer_arguments(parser)
    _add_callback_arguments(parser)
    _add_augmentation_arguments(parser)
    _add_training_arguments(parser)
    _add_model_arguments(parser)
    _add_device_arguments(parser)

    args = parser.parse_args()

    get_config()


def _trim_preceding_hyphens(st):
    i = 0
    while st[i] == "-":
        i += 1
    return st[i:]


def _arg_to_varname(st: str):
    st = _trim_preceding_hyphens(st)
    st = st.replace("-", "_")

    return st.split("=")[0]


def _argv_to_vars(argv):
    var_names = []
    for arg in argv:
        if arg.startswith("-") and _arg_to_varname(arg) != "config":
            var_names.append(_arg_to_varname(arg))
    return var_names


def get_config():
    """Get config."""
    global args
    override_args = _argv_to_vars(sys.argv)
    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)

    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f'=> Reading YAML config from {args.config}')

    args.__dict__.update(loaded_yaml)
    print(args)
    os.environ['DEVICE_TARGET'] = args.device_target
    if 'DEVICE_NUM' not in os.environ.keys():
        os.environ['DEVICE_NUM'] = str(args.device_num)
    if 'RANK_SIZE' not in os.environ.keys():
        os.environ['RANK_SIZE'] = str(args.device_num)


def get_args():
    """run and get args"""
    global args
    if args is None:
        parse_arguments()
    return args
