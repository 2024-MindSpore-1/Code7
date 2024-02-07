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
"""
Script to export model to ONNX, MINDIR or AIR format.

When export to ONNX the results may slightly differ because AdaptiveAvgPool2d
is not support in the current version of ONNX and the wrapper
for ReduceMean is used.
"""
import logging

import numpy as np
from mindspore import (
    Tensor, export, context
)
from mindspore import dtype as mstype

from src.config import get_config
from src.tools.cell import cast_amp
from src.tools.get_misc import get_model, config_logging, load_pretrained


def main():
    args = get_config()
    context.set_context(
        mode=context.GRAPH_MODE, device_target=args.device_target
    )
    if args.device_target in ['Ascend', 'GPU']:
        context.set_context(device_id=args.device_id)
    onnx_export = args.file_format == 'ONNX'
    if onnx_export:
        logging.warning(
            'Warning! When exporting to ONNX, the final accuracy may '
            'differ slightly.'
        )
    net = get_model(
        args.arch,
        args.num_classes,
        not args.disable_approximate_gelu,
        args.use_pytorch_maxpool,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        attn_drop_rate=args.drop_block,
        avg_pool_reduce_mean=onnx_export,
    )
    assert args.pretrained is not None, 'checkpoint_path is None.'
    load_pretrained(args, net, exclude_epoch_state=True)

    cast_amp(net, args)

    net.set_train(False)
    net.to_float(mstype.float32)

    input_arr = Tensor(
        np.zeros([1, 3, args.image_size, args.image_size], np.float32)
    )
    export(
        net, input_arr, file_name=args.arch, file_format=args.file_format
    )


if __name__ == '__main__':
    config_logging()
    main()
