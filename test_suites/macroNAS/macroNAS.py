import json
import os
from collections import OrderedDict
from pathlib import Path
import numpy as np
from numpy import ndarray
from typing import List
from test_suites.utils import define_H

from pymoo.indicators.igd import IGD
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from evoxbench.modules import SearchSpace, Evaluator, Benchmark

__all__ = ['MacroNASSearchSpace', 'MacroNASEvaluator', 'MacroNASBenchmark']

min_max_values = {
    'cifar10': {
        'err_val': [7.51, 28.59],
        'err_test': [8.10, 28.80],
        'mmacs': [21.31, 239.28],
        'params': [547146, 3400010]
    },
    'cifar100': {
        'err_val': [29.51, 54.82],
        'err_test': [29.49, 54.20],
        'mmacs': [21.54, 239.51],
        'params': [662436, 3515300],
    },
}

def get_path(name):
    return str(Path(os.environ.get("EVOXBENCH_MODEL", os.getcwd())) / "macronas" / name)

class MacroNASSearchSpace(SearchSpace):
    """
        MacroNAS search space
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.allowed_ops: List[str] = ['I', '1', '2']

        # upper and lower bound on the decision variables
        self.n_var = 14
        self.lb = [0] * self.n_var
        self.ub = [2] * self.n_var

        # create the categories for each variable
        self.categories = [list(range(a, b + 1)) for a, b in zip(self.lb, self.ub)]

    @property
    def name(self):
        return 'OurMacroNASSearchSpace'

    def _sample(self):
        x = np.random.choice(len(self.allowed_ops), self.n_var)
        return self._decode(x)

    def _encode(self, arch: str) -> ndarray:
        # encode architecture phenotype to genotype
        # a sample architecture: 'I12II122211I12' -> genotype: 01200122211012
        x_ops = []
        for op in arch:
            if op == 'I':
                x_ops.append(0)
            else:
                x_ops.append(int(op))
        x_ops = np.array(x_ops)
        return x_ops

    def _decode(self, x: ndarray) -> str:
        result = ''
        for op in x:
            if op == 0:
                result += 'I'
            else:
                result += f'{op}'
        return result

    def visualize(self, arch):
        raise NotImplementedError

class MacroNASEvaluator(Evaluator):
    def __init__(self,
                 objs='err&mmacs',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 ):
        super().__init__(objs)
        data_file_path = get_path(f'macronas_data_{dataset}.json')
        self.dataset = dataset
        self.data = json.load(open(data_file_path))

    @property
    def name(self):
        return 'OurMacroNASEvaluator'

    def evaluate(self, archs, objs=None,
                 true_eval=False
                 ):

        if objs is None:
            objs = self.objs
        objs = objs.split('&')
        batch_stats = []
        for arch in archs:
            info = self.data[arch]
            stats = OrderedDict()
            if true_eval:
                top1 = info['test_acc'] # test accuracy
            else:
                top1 = info['val_acc'] # valid accuracy

            if 'err' in objs:
                stats['err'] = 100 - top1
            if 'params' in objs:
                stats['params'] = info['Params']
            if 'mmacs' in objs:
                stats['mmacs'] = info['MMACs']

            batch_stats.append(stats)

        return batch_stats

class MacroNASBenchmark(Benchmark):
    def __init__(self,
                 objs='err&mmacs',  # objectives to be minimized
                 dataset='cifar10',  # ['cifar10', 'cifar100', 'ImageNet16-120']
                 normalized_objectives=True,  # whether to normalize the objectives
                 ):

        search_space = MacroNASSearchSpace()
        evaluator = MacroNASEvaluator(objs, dataset)
        super().__init__(search_space, evaluator, normalized_objectives)
        self.objs = objs
        pf_file_path = get_path(f'macronas_pf_{dataset}.json')  # path to the Pareto front json file
        ps_file_path = get_path(f'macronas_ps_{dataset}.json')  # path to the Pareto set json file
        self.pf = np.array(json.load(open(pf_file_path, 'r'))[objs])
        self.ps = np.array(json.load(open(ps_file_path, 'r'))[objs])

        pf_file_path = get_path(f'macronas_pf_val_{dataset}.json')  # path to the Pareto front json file
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
        return 'OurMacroNASBenchmark'

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
        hv_ratio = self.metric_hv_val(F)/self.hv_norm_val
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

    @staticmethod
    def getGenotypeHash(genotype):
        OPS_LIST = ['I', '1', '2']

        genotypeHash = ''
        for i, x in enumerate(genotype):
            if i in [4, 8, 12]:
                genotypeHash += '|'
            if x != 0:
                genotypeHash += OPS_LIST[x]
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
    def isGenotypeValid():
        return True

if __name__ == '__main__':
    ss = MacroNASSearchSpace()
    samples = ss.sample(2)
    # for sample in samples:
    print(samples, ss.encode(samples))
