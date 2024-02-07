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

"""Implementation of the Conformer model."""
import mindspore as ms
import mindspore.nn as nn
from mindspore.common import initializer as init

from .layers.conv_block import ConvBlock
from .layers.block import Block
from .layers.convtransblock import ConvTransBlock
from .layers.custom_identity import CustomIdentity
from .layers.avg_pool import AvgPoolReduceMean


class PytorchMaxPool2d(nn.Cell):
    """
    MaxPool used the same padding as PyTorch version.

    Used for converted weights.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad = ms.ops.Pad(((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
        )  # 1 / 4 [56, 56]

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        return self.maxpool(self.pad(x))


class Conformer(nn.Cell):
    """
    Conformer model class.
    """

    def __init__(
            self,
            patch_size=16,
            in_chans=3,
            num_classes=1000,
            base_channel=64,
            channel_ratio=4,
            num_med_block=0,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=False,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            approximate_gelu=True,
            use_pytorch_maxpool=False,
            avg_pool_reduce_mean=False,
    ):

        # Transformer
        super().__init__()
        self.num_classes = num_classes
        # num_features for consistency with other models
        self.num_features = self.embed_dim = embed_dim
        assert depth % 3 == 0

        self.cls_token = ms.Parameter(
            init.initializer(
                init.TruncatedNormal(0.02),
                (1, 1, embed_dim),
                ms.float32
            )
        )
        linspace = ms.ops.LinSpace()
        # stochastic depth decay rule
        self.trans_dpr = [
            x for x in linspace(
                ms.Tensor(0, ms.float32),
                ms.Tensor(drop_path_rate, ms.float32),
                depth
            )
        ]
        self.tile = ms.ops.Tile()

        # Classifier head
        self.trans_norm = nn.LayerNorm((embed_dim,))
        self.trans_cls_head = (
            nn.Dense(embed_dim, num_classes) if num_classes > 0
            else CustomIdentity()
        )
        self.pooling = (
            AvgPoolReduceMean() if avg_pool_reduce_mean
            else nn.AdaptiveAvgPool2d((1, 1))
        )

        self.conv_cls_head = nn.Dense(int(256 * channel_ratio), num_classes)

        # Stem stage: get the feature maps by conv block
        # 1 / 2 [112, 112]
        self.conv1 = nn.Conv2d(
            in_chans,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            pad_mode='pad',
            has_bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()

        # 1 / 4 [56, 56]
        self.maxpool = (
            PytorchMaxPool2d() if use_pytorch_maxpool
            else nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        )

        # 1 stage
        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(
            inplanes=64,
            outplanes=stage_1_channel,
            res_conv=True,
            stride=1
        )
        self.trans_patch_conv = nn.Conv2d(
            64,
            embed_dim,
            kernel_size=trans_dw_stride,
            stride=trans_dw_stride,
            padding=0,
            has_bias=True,
        )
        self.trans_1 = Block(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=self.trans_dpr[0],
            act_layer=nn.GELU(approximate=approximate_gelu)
        )

        self.first_conv_trans_stage = 2

        # 2~4 stage
        init_stage = 2
        fin_stage = depth // 3 + 1
        self.conv_trans_blocks = nn.CellList()
        for i in range(init_stage, fin_stage):
            self.conv_trans_blocks.append(
                ConvTransBlock(
                    stage_1_channel,
                    stage_1_channel,
                    False,
                    1,
                    dw_stride=trans_dw_stride,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=self.trans_dpr[i - 1],
                    num_med_block=num_med_block,
                    trans_act_layer=nn.GELU(approximate=approximate_gelu)
                )
            )

        stage_2_channel = int(base_channel * channel_ratio * 2)
        # 5~8 stage
        init_stage = fin_stage  # 5
        fin_stage = fin_stage + depth // 3  # 9
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = (
                stage_1_channel if i == init_stage else stage_2_channel
            )
            res_conv = i == init_stage
            self.conv_trans_blocks.append(
                ConvTransBlock(
                    in_channel,
                    stage_2_channel,
                    res_conv,
                    s,
                    dw_stride=trans_dw_stride // 2,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=self.trans_dpr[i - 1],
                    num_med_block=num_med_block,
                    trans_act_layer=nn.GELU(approximate=approximate_gelu)
                )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        # 9~12 stage
        init_stage = fin_stage  # 9
        fin_stage = fin_stage + depth // 3  # 13
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = (
                stage_2_channel if i == init_stage else stage_3_channel
            )
            res_conv = i == init_stage
            last_fusion = i == depth
            self.conv_trans_blocks.append(
                ConvTransBlock(
                    in_channel,
                    stage_3_channel,
                    res_conv,
                    s,
                    dw_stride=trans_dw_stride // 4,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=self.trans_dpr[i - 1],
                    num_med_block=num_med_block,
                    last_fusion=last_fusion,
                    trans_act_layer=nn.GELU(approximate=approximate_gelu)
                )
            )
        self.fin_stage = fin_stage

        self._init_weights()

    def _init_weights(self):
        for _, m in self.parameters_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(
                    init.initializer(
                        init.TruncatedNormal(sigma=0.02),
                        m.weight.shape,
                        m.weight.dtype
                    )
                )
                if isinstance(m, nn.Dense) and m.bias is not None:
                    m.bias.set_data(
                        init.initializer(
                            init.Constant(0),
                            m.bias.shape,
                            m.bias.dtype
                        )
                    )
            elif isinstance(m, nn.Conv2d):
                m.weight.set_data(
                    init.HeNormal(mode='fan_out', nonlinearity='relu')
                )
            elif (
                    isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm))
            ):
                m.gamma.set_data(
                    init.initializer(
                        init.Constant(1),
                        m.gamma.shape,
                        m.gamma.dtype
                    )
                )
                m.beta.set_data(
                    init.initializer(
                        init.Constant(0),
                        m.beta.shape,
                        m.beta.dtype
                    )
                )

    def construct(self, x):
        B = x.shape[0]
        cls_tokens = self.tile(self.cls_token, (B, 1, 1))

        # stem stage [N, 3, 224, 224] -> [N, 64, 56, 56]
        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        t = self.trans_patch_conv(x_base)
        b, c, h, w = t.shape
        t1 = ms.ops.reshape(t, (b, c, h * w))
        x_t = ms.ops.transpose(t1, (0, 2, 1))
        x_t = ms.ops.concat([cls_tokens, x_t], axis=1)
        x_t = self.trans_1(x_t)

        # 2 ~ final
        for i in range(2, self.fin_stage):
            x, x_t = self.conv_trans_blocks[i - self.first_conv_trans_stage](
                x, x_t
            )

        # conv classification
        x_p = self.pooling(x)
        _, c, h, w = x_p.shape
        x_p = ms.ops.reshape(x_p, (B, c))
        conv_cls = self.conv_cls_head(x_p)

        # trans classification
        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        return [conv_cls, tran_cls]


def conformer_tiny_patch16(
        num_classes,
        approximate_gelu=True,
        use_pytorch_maxpool=False,
        **kwargs,
):
    model = Conformer(
        patch_size=16,
        num_classes=num_classes,
        channel_ratio=1,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        approximate_gelu=approximate_gelu,
        use_pytorch_maxpool=use_pytorch_maxpool,
        **kwargs,
    )
    return model


def conformer_small_patch16(
        num_classes,
        approximate_gelu=True,
        use_pytorch_maxpool=False,
        **kwargs,
):
    model = Conformer(
        patch_size=16,
        num_classes=num_classes,
        channel_ratio=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        approximate_gelu=approximate_gelu,
        use_pytorch_maxpool=use_pytorch_maxpool,
        **kwargs
    )
    return model


def conformer_small_patch32(
        num_classes,
        approximate_gelu=True,
        use_pytorch_maxpool=False,
        **kwargs,
):
    model = Conformer(
        patch_size=32,
        num_classes=num_classes,
        channel_ratio=4,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        approximate_gelu=approximate_gelu,
        use_pytorch_maxpool=use_pytorch_maxpool,
        **kwargs
    )
    return model


def conformer_base_patch16(
        num_classes,
        approximate_gelu=True,
        use_pytorch_maxpool=False,
        **kwargs
):
    model = Conformer(
        patch_size=16,
        num_classes=num_classes,
        channel_ratio=6,
        embed_dim=576,
        depth=12,
        num_heads=9,
        mlp_ratio=4,
        qkv_bias=True,
        approximate_gelu=approximate_gelu,
        use_pytorch_maxpool=use_pytorch_maxpool,
        **kwargs
    )
    return model
