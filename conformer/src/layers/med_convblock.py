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

import mindspore.nn as nn


class Med_ConvBlock(nn.Cell):
    """
    Special case for Convblock with down sampling.
    """

    def __init__(
            self,
            inplanes,
            act_layer=nn.ReLU,
            groups=1,
            norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
            drop_block=None,
            drop_path=None
    ):

        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(
            inplanes, med_planes,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=False
        )
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer()

        self.conv2 = nn.Conv2d(
            med_planes,
            med_planes,
            kernel_size=3,
            stride=1,
            group=groups,
            padding=1,
            has_bias=False
        )
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer()

        self.conv3 = nn.Conv2d(
            med_planes,
            inplanes,
            kernel_size=1,
            stride=1,
            padding=0,
            has_bias=False
        )
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer()

        self.drop_block = drop_block
        self.drop_path = drop_path

    def construct(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x
