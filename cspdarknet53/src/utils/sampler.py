# Copyright 2021 Huawei Technologies Co., Ltd
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
choose samples from the dataset
"""
import math
import numpy as np

class DistributedSampler():
    """
    sampling the dataset.

    Args:
    Returns:
        num_samples, number of samples.
    """
    def __init__(self, dataset, rank, group_size, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_len = len(self.dataset)
        self.num_samples = int(math.ceil(self.dataset_len * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_len).tolist()
        else:
            indices = list(range(len(self.dataset_len)))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples
