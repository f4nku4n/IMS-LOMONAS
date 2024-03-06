import numpy as np
import matplotlib.pyplot as plt
from pymoo.core.callback import Callback

# RESULT
class Result:
    def __init__(self, genotype_list):
        self.genotype_list = np.array(genotype_list)

# CALLBACK
class CustomCallback(Callback):
    """
        This callback is only used for EVOXBENCH
    """
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.data['trend_res'] = []
        self.test_suite = kwargs['test_suite']
        self.verbose = kwargs['verbose']
        self.using_archive = kwargs['using_archive']

    def notify(self, algorithm):
        if self.using_archive:
            genotype_archive = np.array([solution.X for solution in algorithm.archive.archive])
            F_archive = np.array([solution.F for solution in algorithm.archive.archive])
        else:
            genotype_archive = algorithm.opt.get('X')
            F_archive = algorithm.opt.get('F')
        igd_val, igd_plus_val, hv_val, hv_ratio_val = 99999, 99999, 0.0, 0.0
        igd, igd_plus, hv, hv_ratio = 99999, 99999, 0.0, 0.0
        if self.test_suite == 'gecco' and len(genotype_archive) != 0:
            performance = algorithm.problem.calc_perf_indicator(genotype_archive, 'all')
            igd = np.round(performance['igd'], 6)
            igd_plus = np.round(performance['igd+'], 6)
            hv = np.round(performance['hv'], 6)
            hv_ratio = np.round(performance['normalized_hv'], 6)
        if self.verbose:
            content = [algorithm.evaluator.n_eval, igd_val, igd_plus_val, hv_val, hv_ratio_val, igd, igd_plus, hv, hv_ratio]
            print("\033[92m{:<10}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m |".format(*content))
        self.data['trend_res'].append([algorithm.evaluator.n_eval, None, igd_val, igd_plus_val, hv_val, hv_ratio_val, igd, igd_plus, hv, hv_ratio])

class Footprint:
    def __init__(self):
        self.data = {}

class Debugger:
    def __init__(self, **kwargs):
        self.test_suite = kwargs['test_suite']
        self.verbose = kwargs['verbose']

    def __call__(self, **kwargs):
        algorithm = kwargs['algorithm']
        genotype_archive = np.array([s.X for s in algorithm.archive.archive])
        F_archive = np.array([s.F for s in algorithm.archive.archive])

        igd, igd_plus, hv, hv_ratio = 99999, 99999, 0.0, 0.0
        if self.test_suite == 'gecco':
            performance = algorithm.problem.calc_perf_indicator(genotype_archive, 'all')
            igd_plus = np.round(performance['igd+'], 6)
            hv = np.round(performance['hv'], 6)
        algorithm.res_logged.append([algorithm.evaluator.n_eval, None, igd_plus, hv])

def visualize_Elitist_Archive_and_Pareto_Front(AF, POF, xlabel=None, ylabel=None, path=None, figname=None):
    AF_ = np.array(AF)
    AF_[:, 0], AF_[:, 1] = AF_[:, 1], AF_[:, 0].copy()
    AF_ = np.unique(AF_, axis=0)
    X = AF_[:, 0]
    Y = AF_[:, 1]

    X_, Y_ = [], []
    for i in range(len(X)):
        X_.append(X[i])
        Y_.append(Y[i])
        if i < len(X) - 1:
            X_.append(X[i + 1])
            Y_.append(Y[i])

    plt.plot(X_, Y_, '--', c='tab:blue')

    plt.scatter(X, Y, c='tab:blue', s=14, marker='o', label='Approximation Front')

    POF_ = np.array(POF)
    POF_[:, 0], POF_[:, 1] = POF_[:, 1], POF_[:, 0].copy()
    POF_ = np.unique(POF_, axis=0)
    X = POF_[:, 0]
    Y = POF_[:, 1]

    X_, Y_ = [], []
    for i in range(len(X)):
        X_.append(X[i])
        Y_.append(Y[i])
        if i < len(X) - 1:
            X_.append(X[i + 1])
            Y_.append(Y[i])

    plt.plot(X_, Y_, '--', c='tab:red')

    plt.scatter(X, Y, edgecolor='tab:red', facecolor='none', marker='s', label='Pareto-optimal Front')

    plt.legend(loc='best')
    plt.grid(linestyle='--')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    if figname is None:
        figname = 'AF-POF'
    plt.savefig(f'{path}/{figname}.jpg', bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.clf()

if __name__ == '__main__':
    # sol = Solution(1, 2, 3)
    # sol.print()
    pass