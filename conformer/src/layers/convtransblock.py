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

import mindspore.nn as nn

from .conv_block import ConvBlock
from .block import Block
from .med_convblock import Med_ConvBlock
from .fcu import FCUUp, FCUDown


class ConvTransBlock(nn.Cell):
    """
    Basic module for ConvTransformer.

    Keep feature maps for CNN block and patch embeddings for transformer
    encoder block.
    """

    def __init__(
            self,
            inplanes,
            outplanes,
            res_conv,
            stride,
            dw_stride,
            embed_dim,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,  # drop_block
            drop_path_rate=0.,
            last_fusion=False,
            num_med_block=0,
            groups=1,
            trans_act_layer=nn.GELU(),
    ):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(
            inplanes=inplanes,
            outplanes=outplanes,
            res_conv=res_conv,
            stride=stride,
            groups=groups
        )

        if last_fusion:
            self.fusion_block = ConvBlock(
                inplanes=outplanes,
                outplanes=outplanes,
                stride=2,
                res_conv=True,
                groups=groups
            )
        else:
            self.fusion_block = ConvBlock(
                inplanes=outplanes,
                outplanes=outplanes,
                groups=groups
            )

        if num_med_block > 0:
            self.med_block = []
            for _ in range(num_med_block):
                self.med_block.append(
                    Med_ConvBlock(inplanes=outplanes, groups=groups)
                )
            self.med_block = nn.CellList(self.med_block)

        self.squeeze_block = FCUDown(
            inplanes=outplanes // expansion,
            outplanes=embed_dim,
            dw_stride=dw_stride,
            act_layer=trans_act_layer
        )

        self.expand_block = FCUUp(
            inplanes=embed_dim,
            outplanes=outplanes // expansion,
            up_stride=dw_stride
        )

        self.trans_block = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            act_layer=trans_act_layer
        )

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion

    def construct(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(
            x_t, H // self.dw_stride,
            W // self.dw_stride
        )
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t
