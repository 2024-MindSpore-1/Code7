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

import mindspore as ms
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed

from src.config import get_args
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, \
    load_pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer


def main():
    args = get_args()
    set_seed(args.seed)
    context.set_context(
        mode=context.GRAPH_MODE, device_target=args.device_target
    )
    context.set_context(enable_graph_kernel=False)
    if args.device_target == 'Ascend':
        context.set_context(enable_auto_mixed_precision=True)
    set_device(args)

    net = get_model(args.arch, args.num_classes)
    net.set_train(False)

    if not args.pretrained:
        raise RuntimeError('Path to checkpoint (pretrained option) not set.')
    load_pretrained(args, net)

    print(
        'Number of parameters:',
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
    eval_network = nn.WithEvalCell(
        net, criterion, args.amp_level in ['O2', 'O3', 'auto']
    )
    eval_indexes = [0, 1, 2]
    eval_metrics = {
        'Loss': nn.Loss(),
        'Top1-Acc': nn.Top1CategoricalAccuracy(),
        'Top5-Acc': nn.Top5CategoricalAccuracy()
    }
    model = Model(net_with_loss, metrics=eval_metrics,
                  eval_network=eval_network,
                  eval_indexes=eval_indexes)
    loss_monitor_cb = ms.LossMonitor(args.print_loss_every)
    print(f'=> begin eval')
    results = model.eval(data.val_dataset, callbacks=[loss_monitor_cb])
    print(f'=> eval results: {results}')
    print(f'=> eval success')


if __name__ == '__main__':
    main()
