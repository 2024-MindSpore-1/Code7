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
#
# This file has been derived from the https://github.com/pengzhiliang/Conformer
# repository and modified.
# ============================================================================

from functools import partial

import mindspore as ms
import mindspore.nn as nn


class FCUDown(nn.Cell):
    """
    CNN feature maps -> Transformer patch embeddings
    """

    def __init__(
            self,
            inplanes,
            outplanes,
            dw_stride,
            act_layer=nn.GELU(),
            norm_layer=partial(nn.LayerNorm, epsilon=1e-6)
    ):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(
            inplanes,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            pad_mode='pad',
            has_bias=True,
        )
        self.sample_pooling = nn.AvgPool2d(
            kernel_size=dw_stride, stride=dw_stride
        )

        self.ln = norm_layer((outplanes,))
        self.act = act_layer

    def construct(self, x, x_t):
        x = self.conv_project(x)  # [N, C, H, W]

        t = self.sample_pooling(x)
        b, c, h, w = t.shape
        # analog of flatten(2) in Pytorch
        t1 = ms.ops.Reshape()(t, (b, c, h * w))
        x = ms.ops.Transpose()(t1, (0, 2, 1))

        x = self.ln(x)
        x = self.act(x)

        x = ms.ops.concat([x_t[:, 0][:, None, :], x], axis=1)

        return x


class FCUUp(nn.Cell):
    """ Transformer patch embeddings -> CNN feature maps
    """

    def __init__(
            self,
            inplanes,
            outplanes,
            up_stride,
            act_layer=nn.ReLU,
            norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
    ):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(
            inplanes,
            outplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            pad_mode='pad',
            has_bias=True,
        )
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def construct(self, x, H, W):
        B, _, C = x.shape
        # [N, 197, 384] -> [N, 196, 384] -> [N, 384, 196] -> [N, 384, 14, 14]
        x_r = ms.ops.transpose(x[:, 1:], (0, 2, 1))
        x_r = ms.ops.reshape(x_r, (B, C, H, W))

        x_r = self.act(self.bn(self.conv_project(x_r)))

        return ms.ops.ResizeNearestNeighbor(
            size=(H * self.up_stride, W * self.up_stride)
        )(x_r)
