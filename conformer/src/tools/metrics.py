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
"""Custom metrics for two-heads model."""
import logging

from mindspore.nn.metrics.metric import EvaluationBase, Metric
from mindspore.nn.metrics.topk import TopKCategoricalAccuracy


def _check_input_correctness(y_pred):
    if not (isinstance(y_pred, (list, tuple)) and len(y_pred) == 2):
        raise RuntimeError(
            f'Applying to unknown predictions (len={len(y_pred)}'
        )


class ConformerMetric(EvaluationBase):
    """
    Base metric class for two-heads model.
    """
    def __init__(
            self,
            metric_cls: EvaluationBase.__class__,
            eval_type='classification'
    ):
        super().__init__(eval_type)
        self.metric = metric_cls(eval_type)

    def clear(self):
        self.metric.clear()

    def _update(self, *inputs):
        raise NotImplementedError('Must be overridden.')

    def update(self, *inputs):
        if len(inputs) != 2:
            logging.error('Skip batch! Expected 2 inputs. Got %d', len(inputs))
            return
        y_pred, labels = inputs
        if not isinstance(y_pred, tuple) and len(y_pred) != 2:
            logging.error('Skip batch! Expected tuple of len 2. Got %s',
                          type(y_pred))
            return
        try:
            self._update(*inputs)
        except ValueError:
            logging.error(
                'Skip batch! Error in metrics computing. Head-1 shape %s. '
                'Head-2 shape %s. Labels shape %s',
                y_pred[0].shape, y_pred[1].shape, labels.shape,
                exc_info=True
            )
            return

    def eval(self):
        return self.metric.eval()


class ConformerMetricHead1(ConformerMetric):
    def _update(self, y_pred, y):
        _check_input_correctness(y_pred)
        self.metric.update(y_pred[0], y)


class ConformerMetricHead2(ConformerMetric):
    def _update(self, y_pred, y):
        _check_input_correctness(y_pred)
        self.metric.update(y_pred[1], y)


class ConformerMetricTwoHeads(ConformerMetric):
    def _update(self, y_pred, y):
        _check_input_correctness(y_pred)
        self.metric.update(y_pred[0] + y_pred[1], y)


class ConformerMetricTwoHeadsTopK(Metric):
    def __init__(self, metric_cls: TopKCategoricalAccuracy.__class__):
        super().__init__()
        self.metric = metric_cls()

    def update(self, y_pred, y):
        _check_input_correctness(y_pred)
        self.metric.update(y_pred[0] + y_pred[1], y)

    def clear(self):
        self.metric.clear()

    def eval(self):
        return self.metric.eval()
