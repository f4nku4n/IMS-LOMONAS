import json
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
from numpy import ndarray
from typing import List
import pickle as p
from test_suites.utils import define_H
from test_suites.nb201 import genotype2phenotype, topology_str2structure

from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from evoxbench.modules import Evaluator, Benchmark
from evoxbench.benchmarks import NASBench201SearchSpace

__all__ = ['OurNASBench201Evaluator', 'OurNASBench201Benchmark']

min_max_values = {
    'cifar10': {
        # 'err': [0, 100],
        'err_val': [21.09, 90.21],
        'err_test': [5.63, 90],
        'params': [0.073306, 1.531546],
        'flops': [7.78305, 220.11969],
        'latency': [0.007145, 0.025579],
        'edgegpu_latency': [0.5013847351074219, 11.692700386047363],
        'edgegpu_energy': [2.059187107086182, 48.78041985034943],
        'eyeriss_latency': [2.059187107086182, 48.78041985034943],
        'eyeriss_energy': [0.3469645085, 2.2262379484999997],
        'eyeriss_arithmetic_intensity': [0.9763649200346838, 27.613486168649743],
    },
    'cifar100': {
        # 'err': [0, 100],
        'err_val': [50.76, 99],
        'err_test': [26.49, 99],
        'params': [0.079156, 1.537396],
        'flops': [7.7889, 220.12554],
        'latency': [0.007203, 0.026141],
        'edgegpu_latency': [0.5058526992797852, 11.464014053344727],
        'edgegpu_energy': [0.5058526992797852, 11.464014053344727],
        'eyeriss_latency': [1.68256, 10.52992],
        'eyeriss_energy': [1.68256, 10.52992],
        'eyeriss_arithmetic_intensity': [0.9759234868167821, 27.58100431822556],
    },
    'ImageNet16-120': {
        # 'err': [0, 100],
        'err_val': [74.54, 99.17],
        'err_test': [52.69, 99.17],
        'params': [0.080456, 1.538696],
        'flops': [1.9534, 55.03756],
        'latency': [0.005842, 0.028224],
        'edgegpu_latency': [0.5339527130126953, 11.527729034423828],
        'edgegpu_energy': [2.1862430334091187, 44.128146743774415],
        'eyeriss_latency': [0.4230400000000001, 2.63488],
        'eyeriss_energy': [0.08940838000000001, 0.5696771319999999],
        'eyeriss_arithmetic_intensity': [0.29561138014527844, 8.328928571428571],
    }
}

def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "nb201" / name)

class OurNASBench201Evaluator(Evaluator):
    def __init__(self,
                 iepoch=12,
                 objs='err&flops',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'in16-120']
                 ):
        super().__init__(objs)
        data_file_path = get_path(f'nb201_data_{dataset}.p')
        self.dataset = dataset
        self.data = p.load(open(data_file_path, 'rb'))
        self.allowed_ops: List[str] = ['none', 'skip_connect', 'nor_conv_1x1', 'nor_conv_3x3', 'avg_pool_3x3']
        self.iepoch = iepoch

    @property
    def name(self):
        return 'OurNASBench201Evaluator'

    def decode(self, arch: str):
        # decode architecture phenotype to key in database
        # a sample architecture
        # '|none~0|+|avg_pool_3x3~0|nor_conv_3x3~1|+|nor_conv_1x1~0|nor_conv_3x3~1|nor_conv_1x1~2|'
        ops = []
        for node in arch.split('+'):
            op = [o.split('~')[0] for o in node.split('|') if o]
            ops.extend(op)
        x_ops = ''
        for op in ops:
            x_ops += f'{(np.array(self.allowed_ops) == op).nonzero()[0][0]}'
        return x_ops

    def evaluate(self, archs, objs=None, true_eval=False):  # query the true (mean over multiple runs) performance
        if objs is None:
            objs = self.objs
        # objs = objs.split('&')[:1]
        objs = objs.split('&')
        batch_stats = []
        for arch in archs:
            key = self.decode(arch)
            stats = OrderedDict()
            if true_eval:
                top1 = self.data['200'][key]['test_acc'][-1] * 100
            else:
                top1 = self.data['200'][key]['val_acc'][self.iepoch - 1] * 100

            if 'err' in objs:
                stats['err'] = 100 - top1
            if 'params' in objs:
                stats['params'] = self.data['200'][key]['params']  # in M
            if 'flops' in objs:
                stats['flops'] = self.data['200'][key]['FLOPs']  # in M
            if 'latency' in objs:
                stats['latency'] = self.data['200'][key]['latency']  # in M
            batch_stats.append(stats)

        return batch_stats

class OurNASBench201Benchmark(Benchmark):
    def __init__(self,
                 objs='err&flops',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 normalized_objectives=True,  # whether to normalize the objectives
                 iepoch=12
                 ):

        search_space = NASBench201SearchSpace()
        # obj_performance = objs
        # obj_complexity = objs[4:]
        evaluator = OurNASBench201Evaluator(iepoch, objs, dataset)
        super().__init__(search_space, evaluator, normalized_objectives)
        # self.evaluator_complexity = NASBench201Evaluator(
        #     objs=obj_complexity,
        #     dataset=dataset)
        self.objs = objs
        pf_file_path = get_path(f'nb201_pf_{dataset}.json')  # path to the Pareto front json file
        self.pf = np.array(json.load(open(pf_file_path, 'r'))[objs])
        self.ps = None

        pf_file_path = get_path(f'nb201_pf_val_{dataset}.json')  # path to the Pareto front json file
        self.pf_val = np.array(json.load(open(pf_file_path, 'r'))[objs])
        self.dataset = dataset

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
        return 'OurNASBench201Benchmark'

    @property
    def pareto_front(self):
        return self.pf

    @property
    def pareto_set(self):
        return self.ps

    @property
    def pareto_front_val(self):
        return self.pf_val

    @property
    def _utopian_point(self):
        list_objs = self.objs.split('&')
        utopian_point = []
        for obj in list_objs:
            if obj == 'err':
                utopian_point.append(min_max_values[self.dataset]['err_test'][0])
            else:
                utopian_point.append(min_max_values[self.dataset][obj][0])
        return utopian_point if len(utopian_point) != 0 else None

    @property
    def _nadir_point(self):
        list_objs = self.objs.split('&')
        nadir_point = []
        for obj in list_objs:
            if obj == 'err':
                nadir_point.append(min_max_values[self.dataset]['err_test'][1])
            else:
                nadir_point.append(min_max_values[self.dataset][obj][1])
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
                utopian_point.append(min_max_values[self.dataset]['err_val'][0])
            else:
                utopian_point.append(min_max_values[self.dataset][obj][0])
        return np.array(utopian_point)

    @property
    def nadir_point_val(self):
        list_objs = self.objs.split('&')
        nadir_point = []
        for obj in list_objs:
            if obj == 'err':
                nadir_point.append(min_max_values[self.dataset]['err_val'][1])
            else:
                nadir_point.append(min_max_values[self.dataset][obj][1])
        return np.array(nadir_point)

    @property
    def hv_ref_point_val(self):
        H = define_H(self.evaluator.n_objs, len(self.pf_val))
        r = np.ones(self.evaluator.n_objs)
        r = r + 1 / H
        return r

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
        hv_ratio = self.metric_hv(F) / self.hv_norm

        performance = {
            'igd': igd,
            'igd+': igdP,
            'hv': hv,
            'normalized_hv': hv_ratio,
        }
        if indicator != 'all':
            return performance[indicator]
        return performance

    @staticmethod
    def getGenotypeHash(genotype):
        phenotype = genotype2phenotype(genotype)
        genotypeHash = topology_str2structure(phenotype).to_unique_str(consider_zero=True)
        return genotypeHash

    def debug(self):
        archs = self.search_space.sample(10)
        X = self.search_space.encode(archs)
        F = self.evaluate(X, true_eval=False)
        print(F)

        igd = self.calc_perf_indicator(X, 'igd')
        hv = self.calc_perf_indicator(X, 'hv')
        norm_hv = self.calc_perf_indicator(X, 'normalized_hv')

        # ps_X = self.search_space.encode(self.ps)
        ps_igd = self.calc_perf_indicator(self.pareto_set, 'igd')
        ps_hv = self.calc_perf_indicator(self.pareto_set, 'hv')
        ps_norm_hv = self.calc_perf_indicator(self.pareto_set, 'normalized_hv')

        print(archs)
        print(X)
        print(F)
        print(igd)
        print(hv)
        print(norm_hv)

        print("PF IGD: {}, this number should be really close to 0".format(ps_igd))
        print(ps_hv)
        print("PF normalized HV: {}, this number should be really close to 1".format(ps_norm_hv))

    @staticmethod
    def isGenotypeValid(genotype):
        return True