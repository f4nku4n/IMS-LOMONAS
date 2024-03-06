import numpy as np
from pymoo.core.problem import Problem

# DEFINE THE PROBLEM
class NASProblem(Problem):
    def __init__(self,
                 benchmark,
                 **kwargs):
        super().__init__(n_var=benchmark.search_space.n_var, n_obj=benchmark.evaluator.n_objs,
                         n_constr=0, xl=benchmark.search_space.lb, xu=benchmark.search_space.ub,
                         type_var=np.int64, **kwargs)
        self.benchmark = benchmark
        self.eval_trend = []
        self.test_suite = kwargs['test_suite']

    @property
    def dataset(self):
        if 'gecco' in self.test_suite:
            return self.benchmark.dataset
        elif 'c10' in self.test_suite:
            return 'CIFAR-10'
        elif 'in1k' in self.test_suite:
            return 'ImageNet1K'
        return ValueError

    def _evaluate(self, x, out, *args, **kwargs):
        F = self.benchmark.evaluate(x, true_eval=False)

        out["F"] = F
        self.eval_trend.append(len(x))

    def reset(self):
        self.eval_trend = []

    def sample(self):
        raw_genotype = self.benchmark.search_space.sample(1)
        genotype = self.benchmark.search_space.encode(raw_genotype)[-1]
        return genotype

    def calc_perf_indicator(self, x, perf_ind, phase='evaluation'):
        if phase == 'search':
            return self.benchmark.calc_perf_indicator_val(x)
        return self.benchmark.calc_perf_indicator(x, perf_ind)

    def isGenotypeValid(self, genotype):
        return self.benchmark.isGenotypeValid(genotype) if 'Our' in self.benchmark.name else True

