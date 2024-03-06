# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import random
import pickle as p
from test_suites.nb101.lib import config, model_spec as _model_spec

import numpy as np
from utilities import get_project_root

# Bring ModelSpec to top-level for convenience. See lib/model_spec.py.
ModelSpec = _model_spec.ModelSpec


class OutOfDomainError(Exception):
    """Indicates that the requested graph is outside of the search domain."""

class NASBench:
    """User-facing API for accessing the NASBench dataset."""
    def __init__(self, dataset_file=None, zc_dataset_file=None):
        self.config = config.build_config()
        if dataset_file is None:
            self.data = None
        else:
            self.data = p.load(open(dataset_file, 'rb'))

        root = get_project_root()
        self.zc_data = p.load(open(f'{root}/database/NASBench101/zc_101.p', 'rb'))

    def getModuleHash(self, model_spec):
        return self._hash_spec(model_spec)

    def query(self, phenotype, epochs=108, metric='test_acc'):
        model_spec = ModelSpec(phenotype['matrix'], phenotype['ops'])
        if not self.is_valid(model_spec):
            return -np.inf if metric in ['val_acc', 'test_acc', 'synflow', 'jacob_cov'] else np.inf
        model_hash = self.getModuleHash(model_spec)
        if 'acc' in metric or metric == 'n_params':
            return self.data[f'{epochs}'][model_hash][metric]
        else:
            return self.zc_data[model_hash][metric]

    def query_time(self, phenotype, epochs=108, metric='test_acc'):
        model_spec = ModelSpec(phenotype['matrix'], phenotype['ops'])
        if not self.is_valid(model_spec):
            return 0.0
        model_hash = self.getModuleHash(model_spec)
        if 'acc' in metric:
            return self.data[f'{epochs}'][model_hash]['train_time']

    def isPhenotypeValid(self, phenotype: dict) -> bool:
        model_spec = ModelSpec(phenotype['matrix'], phenotype['ops'])
        return self.is_valid(model_spec)

    def is_valid(self, model_spec):
        """Checks the validity of the model_spec.

        For the purposes of benchmarking, this does not increment the budget
        counters.

        Args:
          model_spec: ModelSpec object.

        Returns:
          True if model is within space.
        """
        try:
            self._check_spec(model_spec)
        except OutOfDomainError:
            return False
        return True

    def _check_spec(self, model_spec):
        """Checks that the model spec is within the dataset."""
        if not model_spec.valid_spec:
            raise OutOfDomainError('invalid spec, provided graph is disconnected.')

        num_vertices = len(model_spec.ops)
        num_edges = np.sum(model_spec.matrix)

        if num_vertices > self.config['module_vertices']:
            raise OutOfDomainError('too many vertices')

        if num_edges > self.config['max_edges']:
            raise OutOfDomainError('too many edges')

        if model_spec.ops[0] != 'input':
            raise OutOfDomainError('first operation should be \'input\'')

        if model_spec.ops[-1] != 'output':
            raise OutOfDomainError('last operation should be \'output\'')

        for op in model_spec.ops[1:-1]:
            if op not in self.config['available_ops']:
                raise OutOfDomainError('unsupported op %s (available ops = %s)'
                                       % (op, self.config['available_ops']))

    def _hash_spec(self, model_spec):
        """Returns the MD5 hash for a provided model_spec."""
        return model_spec.hash_spec(self.config['available_ops'])