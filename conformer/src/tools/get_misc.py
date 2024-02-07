# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Misc functions for program."""
import os
import datetime as dt
import logging
from pathlib import Path
import shutil
import json
from typing import Optional

from mindspore import context
from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.ops import functional as F
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import ModelCheckpoint, CheckpointConfig, LossMonitor

from ..data.imagenet import ImageNet
from ..conformer import (
    conformer_tiny_patch16, conformer_small_patch16, conformer_base_patch16
)
from ..trainer.train_one_step_with_ema import TrainOneStepWithEMA
from ..trainer.train_one_step_with_scale_and_clip_global_norm import (
    TrainOneStepWithLossScaleCellGlobalNormClip
)
from .callback import (
    SummaryCallbackWithEval,
    BestCheckpointSavingCallback,
    TrainTimeMonitor,
    EvalTimeMonitor,
)


def _get_rank() -> Optional[int]:
    try:
        rank = get_rank()
    except RuntimeError:
        rank = None
    return rank


def set_device(args):
    """Set device and ParallelMode(if device_num > 1)."""
    rank = 0
    # set context and device
    device_target = args.device_target
    device_num = args.device_num

    if device_target == 'Ascend':
        if device_num > 1:
            context.set_context(device_id=int(os.environ['DEVICE_ID']))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True
            )
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    elif device_target == 'GPU':
        if device_num > 1:
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(
                device_num=device_num,
                parallel_mode=ParallelMode.DATA_PARALLEL,
                gradients_mean=True
            )
            rank = get_rank()
        else:
            context.set_context(device_id=args.device_id)
    else:
        raise ValueError('Unsupported platform.')

    return rank


def get_dataset(args, training=True):
    """Get model according to args.dataset."""
    datasets = {
        'imagenet': ImageNet
    }
    dataset_type = args.dataset.lower()
    logging.info('Getting %s dataset', dataset_type)
    dataset = datasets[dataset_type](args, training)

    return dataset


def get_model(
        arch,
        num_classes,
        approximate_gelu,
        use_pytorch_maxpool,
        drop_rate,
        drop_path_rate,
        attn_drop_rate,
        avg_pool_reduce_mean=False,
):
    """"Get model according to args.arch"""
    logging.info('Creating model %s', arch)
    models = {
        'conformer_tiny_patch16': conformer_tiny_patch16,
        'conformer_small_patch16': conformer_small_patch16,
        'conformer_base_patch16': conformer_base_patch16
    }
    model = models.get(arch)
    if model is None:
        raise RuntimeError(f'Unknown model: {arch}')
    return model(
        num_classes,
        approximate_gelu,
        use_pytorch_maxpool,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        attn_drop_rate=attn_drop_rate,
        avg_pool_reduce_mean=avg_pool_reduce_mean,
    )


def load_pretrained(args, model, exclude_epoch_state=True):
    """Load pretrained weights if args.pretrained is given."""
    if os.path.isfile(args.pretrained):
        logging.info('=> loading pretrained weights from %s', args.pretrained)
        param_dict = load_checkpoint(args.pretrained)
        for key, value in param_dict.copy().items():
            if 'trans_cls_head' in key or 'conv_cls_head' in key:
                if value.shape[0] != args.num_classes:
                    logging.info('Removing %s with shape %s', key, value.shape)
                    param_dict.pop(key)
        if exclude_epoch_state:
            state_params = [
                'scale_sense', 'global_step', 'momentum', 'learning_rate',
                'epoch_num', 'step_num'
            ]
            for state_param in state_params:
                if state_param in param_dict:
                    param_dict.pop(state_param)

        load_param_into_net(model, param_dict)
    else:
        logging.warning('No pretrained weights found at %s', args.pretrained)


def get_train_one_step(args, net_with_loss, optimizer):
    """get_train_one_step cell."""
    if args.is_dynamic_loss_scale:
        logging.info('Using DynamicLossScaleUpdateCell')
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(
            loss_scale_value=2 ** 24, scale_factor=2, scale_window=2000
        )
    else:
        logging.info(
            'Using FixedLossScaleUpdateCell, loss_scale_value: %s',
            args.loss_scale
        )
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(
            loss_scale_value=args.loss_scale
        )
    if args.with_ema:
        logging.info('Using EMA. ema_decay: %f', args.ema_decay)
        net_with_loss = TrainOneStepWithEMA(
            net_with_loss,
            optimizer,
            scale_sense=scale_sense,
            with_ema=args.with_ema,
            ema_decay=args.ema_decay)
    elif args.use_clip_grad_norm:
        logging.info(
            'Using gradient clipping by norm, clip_grad_norm: %f',
            args.clip_grad_norm
        )
        net_with_loss = TrainOneStepWithLossScaleCellGlobalNormClip(
            net_with_loss,
            optimizer,
            scale_sense,
            use_global_norm=args.use_clip_grad_norm,
            clip_global_norm_value=args.clip_grad_norm
        )
    else:
        logging.info('Use simple loss scale.')
        net_with_loss = nn.TrainOneStepWithLossScaleCell(
            net_with_loss, optimizer, scale_sense=scale_sense
        )
    return net_with_loss


def get_directories(
        model_name: str,
        summary_root_dir: str,
        ckpt_root_dir: str,
        best_ckpt_root_dir: str,
        postfix: str = '',
        rank: int = 0,
):
    """Get all directories name used in training."""
    summary_root_dir = Path(summary_root_dir)
    ckpt_root_dir = Path(ckpt_root_dir)
    best_ckpt_root_dir = Path(best_ckpt_root_dir)
    prefix = dt.datetime.now().strftime('%y-%m-%d_%H%M%S')
    dir_name = f'{prefix}_{model_name}_{rank}'
    if postfix != '':
        dir_name = f'{dir_name}_{postfix}'

    summary_dir = summary_root_dir / dir_name
    if summary_dir.exists():
        raise RuntimeError(f'Directory {summary_dir} already exist.')

    ckpt_dir = ckpt_root_dir / dir_name
    if ckpt_dir.exists():
        raise RuntimeError(f'Directory {ckpt_dir} already exist.')

    best_ckpt_dir = best_ckpt_root_dir / dir_name
    if best_ckpt_dir.exists():
        raise RuntimeError(f'Directory {best_ckpt_dir} already exist.')

    summary_dir.mkdir(parents=True)
    ckpt_dir.mkdir(parents=True)
    best_ckpt_dir.mkdir(parents=True)

    return str(summary_dir), str(ckpt_dir), str(best_ckpt_dir)


def save_config(args, argv, dir_to_save):
    """Save config for current training."""
    shutil.copy(args.config_path, dir_to_save)
    with open(Path(dir_to_save) / 'all_args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    with open(Path(dir_to_save) / 'cmd.txt', 'w') as f:
        f.write(' '.join(argv))


def config_logging(
        level=logging.INFO,
        log_format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
):
    """
    Configure logging.
    """
    logging.basicConfig(level=level, datefmt=datefmt, format=log_format)


def add_logging_file_handler(
        filename_prefix,
        filemode='a',
        log_format='[%(asctime)s.%(msecs)03d] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S'
):
    rank = _get_rank()
    filename_suffix = '.log'
    if rank is not None:
        filename_suffix = f'_{rank}' + filename_suffix
    filename = filename_prefix + filename_suffix
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_handler = logging.FileHandler(filename, filemode)
    file_handler.setFormatter(logging.Formatter(log_format, datefmt))
    logging.getLogger().addHandler(file_handler)


def get_callbacks(
        arch,
        train_data_size,
        val_data_size,
        ckpt_dir,
        best_ckpt_dir,
        summary_dir,
        rank=0,
        ckpt_save_every_step=0,
        ckpt_save_every_sec=0,
        ckpt_keep_num=10,
        best_ckpt_num=5,
        print_loss_every=1,
        collect_freq=0,
        collect_tensor_freq=None,
        collect_graph=False,
        collect_input_data=False,
        keep_default_action=False,
):
    """
    Get common callbacks.
    """
    if collect_freq == 0:
        collect_freq = train_data_size
    if ckpt_save_every_step == 0 and ckpt_save_every_sec == 0:
        ckpt_save_every_step = train_data_size
    config_ck = CheckpointConfig(
        # To save every epoch use data.train_dataset.get_data_size(),
        save_checkpoint_steps=ckpt_save_every_step,
        save_checkpoint_seconds=ckpt_save_every_sec,
        keep_checkpoint_max=ckpt_keep_num,
        append_info=['epoch_num', 'step_num']
    )
    train_time_cb = TrainTimeMonitor(data_size=train_data_size)
    eval_time_cb = EvalTimeMonitor(data_size=val_data_size)

    best_ckpt_save_cb = BestCheckpointSavingCallback(
        best_ckpt_dir, target_metric='acc_heads', prefix=arch,
        buffer=best_ckpt_num
    )

    ckpoint_cb = ModelCheckpoint(
        prefix=f'{arch}_{rank}',
        directory=ckpt_dir,
        config=config_ck
    )
    loss_cb = LossMonitor(print_loss_every)

    specified = {
        'collect_metric': True,
        'collect_train_lineage': True,
        'collect_eval_lineage': True,
        # 'histogram_regular': '^network.*weight.*',
        'collect_graph': collect_graph,
        # 'collect_dataset_graph': True,
        'collect_input_data': collect_input_data,
    }
    summary_collector_cb = SummaryCallbackWithEval(
        summary_dir=summary_dir,
        collect_specified_data=specified,
        collect_freq=collect_freq,
        keep_default_action=keep_default_action,
        collect_tensor_freq=collect_tensor_freq
    )
    return [
        train_time_cb,
        eval_time_cb,
        ckpoint_cb,
        loss_cb,
        best_ckpt_save_cb,
        summary_collector_cb,
    ]


class WithEvalCellTwoOutputs(nn.WithEvalCell):
    """
    Wraps the forward network with the loss function.
    """

    def construct(self, data, label):
        outputs = self._network(data)
        if self.add_cast_fp32:
            label = F.mixed_precision_cast(mstype.float32, label)
            outputs = F.cast(outputs, mstype.float32)
        if isinstance(outputs, list):
            loss1 = self._loss_fn(outputs[0], label)
            loss2 = self._loss_fn(outputs[1], label)
            loss = loss1 + loss2
        else:
            loss = self._loss_fn(outputs, label)
        return loss, outputs, label
