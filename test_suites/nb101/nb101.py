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

__all__ = ['OurNASBench101Evaluator', 'OurNASBench101Benchmark']

min_max_values = {
    'cifar10': {
        # 'err': [0, 100],
        'err_val': [12.35, 90.74],
        'err_test': [5.68, 90.02],
        'params': [227274, 49979274],
    },
}

def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "nb101" / name)

class OurNASBench101Evaluator(Evaluator):
    def __init__(self,
                 iepoch=12,
                 objs='err&params',  # objectives to be minimized
                 dataset='cifar10',
                 ):
        super().__init__(objs)
        data_file_path = get_path(f'nb101_data_{dataset}.p')
        zc_data_file_path = get_path(f'nb101_zc_data.p')
        self.api = NASBench(data_file_path)
        self.dataset = dataset
        self.iepoch = iepoch
        self.search_cost = 0.0
        self.evaluation_cost = 0.0

    @property
    def name(self):
        return 'OurNASBench101Evaluator'

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
                    self.search_cost += self.api.query_time(arch, epochs=self.iepoch, metric='val_acc')
                stats['err'] = 100 - top1

            if 'params' in objs:
                stats['params'] = self.api.query(arch, epochs=108, metric='n_params')

            batch_stats.append(stats)

        return batch_stats

class OurNASBench101Benchmark(Benchmark):
    def __init__(self,
                 objs='err&flops',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 normalized_objectives=True,  # whether to normalize the objectives
                 iepoch=12
                 ):

        search_space = NASBench101SearchSpace()
        evaluator = OurNASBench101Evaluator(iepoch, objs, dataset)
        super().__init__(search_space, evaluator, normalized_objectives)
        self.objs = objs
        self.dataset = 'cifar10'
        pf_file_path = get_path(f'nb101_pf_{dataset}.json')  # path to the Pareto front json file
        self.pf = np.array(json.load(open(pf_file_path, 'r'))[objs])

        pf_file_path = get_path(f'nb101_pf_val_{dataset}.json')  # path to the Pareto front json file
        self.pf_val = np.array(json.load(open(pf_file_path, 'r'))[objs])

        pf_norm = self.normalize(self.pareto_front)
        self.metric_igd = IGD(pf=pf_norm)
        self.metric_igdP = IGDPlus(pf=pf_norm)
        self.metric_hv = HV(ref_point=self.hv_ref_point)
        self.hv_norm = self.metric_hv(pf_norm)

        pf_val_norm = self.normalize_val(self.pareto_front_val)
        self.metric_igd_val = IGD(pf=pf_val_norm)
        self.metric_igdP_val = IGDPlus(pf=pf_val_norm)
        self.metric_hv_val = HV(ref_point=self.hv_ref_point_val)
        self.hv_norm_val = self.metric_hv_val(pf_val_norm)

    @property
    def name(self):
        return 'OurNASBench101Benchmark'

    @property
    def pareto_front(self):
        return self.pf

    @property
    def pareto_front_val(self):
        return self.pf_val

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
        H = define_H(self.evaluator.n_objs, len(self.pf))
        r = np.ones(self.evaluator.n_objs)
        r = r + 1 / H
        return r

    @property
    def utopian_point_val(self):
        list_objs = self.objs.split('&')
        utopian_point = []
        for obj in list_objs:
            if obj == 'err':
                utopian_point.append(min_max_values['cifar10']['err_val'][0])
            else:
                utopian_point.append(min_max_values['cifar10'][obj][0])
        return np.array(utopian_point)

    @property
    def nadir_point_val(self):
        list_objs = self.objs.split('&')
        nadir_point = []
        for obj in list_objs:
            if obj == 'err':
                nadir_point.append(min_max_values['cifar10']['err_val'][1])
            else:
                nadir_point.append(min_max_values['cifar10'][obj][1])
        return np.array(nadir_point)

    @property
    def hv_ref_point_val(self):
        H = define_H(self.evaluator.n_objs, len(self.pf_val))
        r = np.ones(self.evaluator.n_objs)
        r = r + 1 / H
        return r

    def evaluate(self, X, **kwargs):
        # convert genotype X to architecture phenotype
        archs = self.search_space.decode(X)

        # query for performance
        batch_stats = self.evaluator.evaluate(archs, **kwargs)

        # convert performance dict to objective matrix
        F = self.to_matrix(batch_stats)

        # normalize objective matrix
        if self.normalized_objectives:
            F = self.normalize(F)
        return F

    def normalize_val(self, F: np.array):
        """ method to normalize the objectives  """
        assert self.utopian_point_val is not None, "Missing Pareto front or utopian point for normalization"
        assert self.nadir_point_val is not None, "Missing Pareto front or nadir point for normalization"
        return (F - self.utopian_point_val) / (self.nadir_point_val - self.utopian_point_val)

    def calc_perf_indicator_val(self, inputs):
        F = np.array(inputs)

        if not self.normalized_objectives:
            F = self.normalize_val(F)  # in case the benchmark evaluator does not normalize objs by default

        igd = self.metric_igd_val(F)
        igdP = self.metric_igdP_val(F)
        hv = self.metric_hv_val(F)
        hv_ratio = self.metric_hv_val(F) / self.hv_norm_val
        performance = {
            'igd': igd,
            'igd+': igdP,
            'hv': hv,
            'normalized_hv': hv_ratio,
        }

        return performance

    def calc_perf_indicator(self, inputs, indicator='igd'):
        if isinstance(inputs[0], ndarray):
            # re-evaluate the true performance
            F = self.evaluate(inputs, true_eval=True)  # use true/mean accuracy
        else:
            batch_stats = self.evaluator.evaluate(inputs, true_eval=True)
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
