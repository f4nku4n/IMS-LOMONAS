import json
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
from numpy import ndarray
from test_suites.utils import define_H
from test_suites.nb101 import NASBench
from evoxbench.modules import Evaluator, Benchmark
from evoxbench.benchmarks import NASBench101SearchSpace

from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

__all__ = ['OurZCNASBench101Evaluator', 'OurZCNASBench101Benchmark']

min_max_values = {
    'cifar10': {
        'err_val': [12.35, 90.74],
        'err_test': [5.68, 90.02],
        'params': [227274, 49979274],
    },
}

time_dict = {
    'synflow': 1.4356617034300945,
    'grad_norm': 2.035946015246093,
    'grasp': 5.570546795804546,
    'jacob_cov': 2.5207841626097856,
    'snip': 2.028758352457235,
    'fisher': 2.610283957422675
}

def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "nb101" / name)

class OurZCNASBench101Evaluator(Evaluator):
    def __init__(self,
                 iepoch=12,
                 objs='err&params',  # objectives to be minimized
                 dataset='cifar10',
                 ):
        super().__init__(objs)
        data_file_path = get_path(f'nb101_data_{dataset}.p')
        self.api = NASBench(data_file_path)
        self.dataset = dataset
        self.iepoch = iepoch
        self.cost_time = 0.0

    @property
    def name(self):
        return 'OurZCNASBench101Evaluator'

    def evaluate(self, archs, objs=None, true_eval=False):  # query the true (mean over multiple runs) performance
        if objs is None:
            objs = self.objs
        objs = objs.split('&')
        batch_stats = []
        for arch in archs:
            stats = OrderedDict()
            if 'err' in objs:
                if true_eval:
                    top1 = self.api.query(arch, epochs=108, metric='test_acc') * 100
                else:
                    top1 = self.api.query(arch, epochs=self.iepoch, metric='val_acc') * 100
                    self.cost_time += self.api.query_time(arch, epochs=self.iepoch, metric='val_acc')
                stats['err'] = 100 - top1
            if 'synflow' in objs:
                stats['synflow'] = -self.api.query(arch, metric='synflow')
                if not np.isinf(stats['synflow']):
                    self.cost_time += time_dict['synflow']
            if 'jacov' in objs:
                stats['jacov'] = -self.api.query(arch, metric='jacob_cov')
                if not np.isinf(stats['jacov']):
                    self.cost_time += time_dict['jacob_cov']
            if 'params' in objs:
                stats['params'] = self.api.query(arch, epochs=108, metric='n_params')
            batch_stats.append(stats)
        return batch_stats

class OurZCNASBench101Benchmark(Benchmark):
    def __init__(self,
                 objs='err&flops',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 normalized_objectives=True,  # whether to normalize the objectives
                 iepoch=12
                 ):

        search_space = NASBench101SearchSpace()
        evaluator = OurZCNASBench101Evaluator(iepoch, 'synflow&jacov&params', dataset)
        super().__init__(search_space, evaluator, normalized_objectives)
        self.objs = objs
        self.dataset = 'cifar10'
        pf_file_path = get_path(f'nb101_pf.json')  # path to the Pareto front json file
        self.pf = np.array(json.load(open(pf_file_path, 'r'))[objs])
        self.pf[:, 0] *= 100
        pf_norm = self.normalize(self.pareto_front)
        self.metric_igd = IGD(pf=pf_norm)
        self.metric_igdP = IGDPlus(pf=pf_norm)
        self.metric_hv = HV(ref_point=self.hv_ref_point)
        self.hv_norm = self.metric_hv(pf_norm)

    @property
    def name(self):
        return 'OurZCNASBench101Benchmark'

    @property
    def pareto_front(self):
        return self.pf

    @property
    def _utopian_point(self):
        list_objs = self.objs.split('&')
        utopian_point = []
        for obj in list_objs:
            if obj == 'err':
                utopian_point.append(min_max_values['cifar10']['err_test'][0])
            else:
                utopian_point.append(min_max_values['cifar10'][obj][0])
        return utopian_point if len(utopian_point) != 0 else None

    @property
    def _nadir_point(self):
        list_objs = self.objs.split('&')
        nadir_point = []
        for obj in list_objs:
            if obj == 'err':
                nadir_point.append(min_max_values['cifar10']['err_test'][1])
            else:
                nadir_point.append(min_max_values['cifar10'][obj][1])
        return nadir_point if len(nadir_point) != 0 else None

    @property
    def _hv_ref_point(self):
        H = define_H(2, len(self.pf))
        r = np.ones(2)
        r = r + 1 / H
        return r

    def evaluate(self, X, **kwargs):
        # convert genotype X to architecture phenotype
        archs = self.search_space.decode(X)

        # query for performance
        batch_stats = self.evaluator.evaluate(archs, **kwargs)

        # convert performance dict to objective matrix
        F = self.to_matrix(batch_stats)
        # print(F)
        # normalize objective matrix
        if self.normalized_objectives:
            F = self.normalize(F)
        return F

    def calc_perf_indicator(self, inputs, indicator='igd'):
        if isinstance(inputs[0], ndarray):
            # re-evaluate the true performance
            F = self.evaluate(inputs, objs=self.objs, true_eval=True)  # use true/mean accuracy
        else:
            batch_stats = self.evaluator.evaluate(inputs, objs=self.objs, true_eval=True)
            F = self.to_matrix(batch_stats)
            if self.normalized_objectives:
                F = self.normalize(F)

        if not self.normalized_objectives:
            F = self.normalize(F)  # in case the benchmark evaluator does not normalize objs by default
        # print(self.normalized_objectives, F)
        # filter out the non-dominated solutions
        nd_front = NonDominatedSorting().do(F, only_non_dominated_front=True)
        F = F[nd_front]

        igd = self.metric_igd(F)
        igdP = self.metric_igdP(F)
        hv = self.metric_hv(F)
        hv_ratio = self.metric_hv(F)/self.hv_norm

        performance = {
            'igd': igd,
            'igd+': igdP,
            'hv': hv,
            'normalized_hv': hv_ratio,
        }
        if indicator != 'all':
            return performance[indicator]
        return performance

    def debug(self):
        archs = self.search_space.sample(10)
        X = self.search_space.encode(archs)
        F = self.evaluate(X, true_eval=False)
        print(F)

        igd = self.calc_perf_indicator(X, 'igd')
        hv = self.calc_perf_indicator(X, 'hv')
        norm_hv = self.calc_perf_indicator(X, 'normalized_hv')

        print(archs)
        print(X)
        print(F)
        print(igd)
        print(hv)
        print(norm_hv)

    def isGenotypeValid(self, genotype):
        phenotype = self.search_space.decode([genotype])[-1]
        return self.evaluator.api.isPhenotypeValid(phenotype)
