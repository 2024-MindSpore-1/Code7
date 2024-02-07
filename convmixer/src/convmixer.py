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
# This file has been derived from the https://github.com/locuslab/convmixer
# repository and modified.
# ============================================================================
"""Implementation of the ConvMixer model."""
from mindspore.ops import ReduceMean
import mindspore.nn as nn


class Residual(nn.Cell):
    """Residual connection. """

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def construct(self, *inputs, **kwargs):
        x = inputs[0]
        return self.fn(x) + x


class AvgPoolReduceMean(nn.Cell):
    """AvgPool cell implemented on the basis of ReduceMean op."""

    def construct(self, *inputs, **kwargs):
        """Forward pass."""
        x = inputs[0]
        return ReduceMean(True)(x, (2, 3))


class ConvMixer(nn.Cell):
    """ConvMixer model."""

    def __init__(
            self,
            dim,
            depth,
            kernel_size=9,
            patch_size=7,
            n_classes=1000,
            act_type='gelu',
            onnx_export=False,
    ):
        super().__init__()
        if act_type.lower() == 'gelu':
            act = nn.GELU
        elif act_type.lower() == 'relu':
            act = nn.ReLU
        else:
            raise NotImplementedError()

        avg_pool = AvgPoolReduceMean() if onnx_export \
            else nn.AdaptiveAvgPool2d((1, 1))

        self.network = nn.SequentialCell(
            nn.Conv2d(
                3,
                dim,
                kernel_size=patch_size,
                stride=patch_size,
                has_bias=True,
                pad_mode='pad',
                padding=0,
            ),
            act(),
            nn.BatchNorm2d(dim),
            *[nn.SequentialCell(
                Residual(
                    nn.SequentialCell(
                        nn.Conv2d(
                            dim,
                            dim,
                            kernel_size,
                            group=dim,
                            pad_mode='same',
                            has_bias=True
                        ),
                        act(),
                        nn.BatchNorm2d(dim)
                    )
                ),
                nn.Conv2d(
                    dim,
                    dim,
                    kernel_size=1,
                    has_bias=True,
                    pad_mode='pad',
                    padding=0,
                ),
                act(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)],
            avg_pool,
            nn.Flatten(),
            nn.Dense(dim, n_classes)
        )

    def construct(self, *inputs, **kwargs):
        """Forward pass."""
        x = inputs[0]
        x = self.network(x)
        return x


def convmixer_1536_20(num_classes=1000, onnx_export=False):
    """Create ConvMixer-1536/20 model."""
    model = ConvMixer(
        1536,
        20,
        kernel_size=9,
        patch_size=7,
        n_classes=num_classes,
        onnx_export=onnx_export
    )
    return model


def convmixer_1024_20(num_classes=1000, onnx_export=False):
    """Create ConvMixer-1024/20 model."""
    model = ConvMixer(
        1024,
        20,
        kernel_size=9,
        patch_size=14,
        n_classes=num_classes,
        onnx_export=onnx_export
    )
    return model


def convmixer_768_32(num_classes=1000, act_type='relu', onnx_export=False):
    """Create ConvMixer-768/32 model."""
    model = ConvMixer(
        768,
        32,
        kernel_size=7,
        patch_size=7,
        n_classes=num_classes,
        act_type=act_type,
        onnx_export=onnx_export
    )
    return model
