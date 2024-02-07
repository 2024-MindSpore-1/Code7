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
"""Evaluation script."""
from functools import reduce
import logging

import mindspore as ms
from mindspore import nn
from mindspore import Model
from mindspore import context
from mindspore.common import set_seed

from src.config import get_config
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import (
    get_dataset, set_device, get_model, load_pretrained, get_train_one_step,
    WithEvalCellTwoOutputs, config_logging
)
from src.tools.optimizer import get_optimizer
from src.tools.metrics import ConformerMetricTwoHeadsTopK


config_logging()


def main():
    args = get_config()
    set_seed(args.seed)
    mode = context.PYNATIVE_MODE if args.pynative_mode else context.GRAPH_MODE
    context.set_context(
        mode=mode, device_target=args.device_target
    )
    context.set_context(enable_graph_kernel=False)
    if args.device_target == 'Ascend':
        context.set_context(enable_auto_mixed_precision=True)
    _ = set_device(args)

    net = get_model(
        args.arch,
        args.num_classes,
        not args.disable_approximate_gelu,
        args.use_pytorch_maxpool,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.drop_block,
    )
    net.set_train(False)

    if args.pretrained == '':
        raise RuntimeError('Path to checkpoint (pretrained option) not set.')
    load_pretrained(args, net)

    logging.info(
        'Number of parameters: %d',
        sum(
            reduce(lambda x, y: x * y, params.shape)
            for params in net.trainable_params()
        )
    )
    cast_amp(net, args)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)

    data = get_dataset(args, training=False)

    batch_num = data.val_dataset.get_dataset_size()
    optimizer = get_optimizer(args, net, batch_num)

    net_with_loss = get_train_one_step(args, net_with_loss, optimizer)

    eval_network = WithEvalCellTwoOutputs(
        net, criterion, args.amp_level in ['O2', 'O3', 'auto']
    )
    eval_indexes = [0, 1, 2]
    eval_metrics = {
        'Loss': nn.Loss(),
        'Top1-Acc': ConformerMetricTwoHeadsTopK(nn.Top1CategoricalAccuracy),
        'Top5-Acc': ConformerMetricTwoHeadsTopK(nn.Top5CategoricalAccuracy)
    }

    model = Model(net_with_loss, metrics=eval_metrics,
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)
    loss_monitor_cb = ms.LossMonitor(args.print_loss_every)
    logging.info('=> begin eval')
    results = model.eval(data.val_dataset, callbacks=[loss_monitor_cb],
                         dataset_sink_mode=False)
    logging.info('eval results: %s', results)
    logging.info('eval success')


if __name__ == '__main__':
    main()
