"""
Source code for Interleaved Multi-start Scheme Non-dominated Sorting Genetic Algorithm II (IMS-NSGA2)
Authors: Quan Minh Phan, Ngoc Hoang Luong
"""
import random
import numpy as np
from typing import Tuple
from functions import is_front_dominated
from utilities import Result
from core import ElitistArchive
from pymoo.algorithms.moo.nsga2 import NSGA2

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair

def get_genetic_operator(crx_prob=0.9,  # crossover probability
                         crx_eta=20.0,  # SBX crossover eta
                         mut_prob=0.9,  # mutation probability
                         mut_eta=20.0,  # polynomial mutation hyperparameter eta
                         ):
    sampling = IntegerRandomSampling()
    crossover = SBX(prob=crx_prob, eta=crx_eta, repair=RoundingRepair(), vtype=int)
    mutation = PM(prob=mut_prob, eta=mut_eta, repair=RoundingRepair(), vtype=int)
    return sampling, crossover, mutation


class IMS_NSGA2:
    def __init__(self,
                 base=2, init_popsize=10,
                 perform_termination=True,
                 verbose=False):
        self.name = 'IMS-NSGA2'

        self.base = base
        self.init_popsize = init_popsize
        self.perform_termination = perform_termination

        self.res_logged = []
        self.verbose = verbose

    @property
    def hyperparameters(self):
        return {
            'optimizer': self.name,
            'base': self.base,
            'initial_pop': self.init_popsize,
            'stop_inefficient_population': self.perform_termination,
            'sampling': 'IntegerRandomSampling',
            'crossover': f'SimulatedBinaryCrossover (pC = 0.9, etaC = 20)',
            'mutation': f'PolynomialMutation (pM = 1/l, etaM = 20)',
            'selection': 'BinaryTournament',
            'ranking_mechanism': 'NonDominatedSorting_and_CrowdingDistance',
            'eliminate_duplicates': True,
            'using_archive': True
        }

    def set(self, key_value):
        for key, value in key_value.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    ######################################################## Main ######################################################
    def solve(self, problem, seed, max_eval, **kwargs):
        random.seed(seed)
        np.random.seed(seed)
        self._solve(problem, max_eval, **kwargs)

        genotype_archive = [elitist.X for elitist in self.archive.archive]
        res = Result(genotype_archive)
        return res

    def _solve(self, problem, max_eval, **kwargs):
        LOP, LOP_name = [], []  # list of processors
        self.archive = ElitistArchive()

        self.problem = problem
        self.max_eval = max_eval

        self.sampling, self.crossover, self.mutation = get_genetic_operator(crx_prob=0.9, crx_eta=20,
                                                                            mut_prob=1/self.problem.n_var, mut_eta=20)

        hat = 1
        n_iter, id_worker = 1, 1
        pop_size = self.init_popsize

        termination = ('n_eval', max_eval)
        new_processor = self.initialize_processor(pop_size=pop_size, termination=termination, verbose=False)
        # Initialize a new processor
        LOP.append(new_processor)
        LOP_name.append(f'NSGA2 #{id_worker - 1} (pop size = {pop_size})')
        isKilled = [False]

        self.n_eval = 0
        while True:
            if n_iter == self.base ** hat:
                pop_size *= 2
                id_worker += 1
                new_processor = self.initialize_processor(pop_size=pop_size,
                                                          termination=termination, verbose=False)  # Initialize a new processor
                LOP.append(new_processor)  # Add the new processor to the last position of LOP
                LOP_name.append(f'NSGA2 #{id_worker - 1} (pop size = {pop_size})')
                isKilled.append(False)
                hat += 1

            for i, processor in enumerate(LOP):
                if n_iter % (self.base ** i) == 0:
                    # if not isKilled[i]:
                    processor.next()
                    self.n_eval += processor.problem.eval_trend[-1]
                    self.updateArchive(processor.archive)
                    self.debug()
                if self.isTerminated():
                    return

            if self.perform_termination:
                isKilled = self.selection(isKilled, LOP, LOP_name)

                # Reset the counter
                isKilled = np.array(isKilled)
                LOP = np.array(LOP)[~isKilled].tolist()
                LOP_name = np.array(LOP_name)[~isKilled].tolist()
                isKilled = isKilled[~isKilled].tolist()

            n_iter += 1

    ##################################################### Utilities ####################################################
    def isTerminated(self):
        return self.n_eval >= self.max_eval

    def initialize_processor(self, pop_size, **kwargs):
        processor = NSGA2(pop_size=pop_size,
                          sampling=self.sampling, crossover=self.crossover, mutation=self.mutation,
                          archive=ElitistArchive(), eliminate_duplicates=True)
        processor.setup(self.problem, **kwargs)

        return processor

    def updateArchive(self, sub_archive):
        self.archive.add(sub_archive.archive)

    @staticmethod
    def selection(isKilled, LOP, LOP_name) -> Tuple:
        # Only kill processors that are dominated by one of later processors
        for i in range(len(LOP) - 1):
            F_Pi = np.array([s.F for s in LOP[i].archive.archive])
            for j in range(i + 1, len(LOP)):
                F_Pj = np.array([s.F for s in LOP[j].archive.archive])
                if is_front_dominated(front_1=F_Pi, front_2=F_Pj):
                    isKilled[i] = True
                    print(f'-> Kill processor {LOP_name[i]}')
                    break

        return isKilled

    def debug(self):
        genotype_archive = np.array([s.X for s in self.archive.archive])
        F_archive = np.array([s.F for s in self.archive.archive])

        igd_val, igd_plus_val, hv_val, hv_ratio_val = 99999, 99999, 0.0, 0.0
        igd, igd_plus, hv, hv_ratio = 99999, 99999, 0.0, 0.0
        if self.problem.test_suite == 'gecco':
            performance_val = self.problem.calc_perf_indicator(F_archive, 'all', phase='search')
            igd_val = np.round(performance_val['igd'], 6)
            igd_plus_val = np.round(performance_val['igd+'], 6)
            hv_val = np.round(performance_val['hv'], 6)
            hv_ratio_val = np.round(performance_val['normalized_hv'], 6)

            performance = self.problem.calc_perf_indicator(genotype_archive, 'all')
            igd = np.round(performance['igd'], 6)
            igd_plus = np.round(performance['igd+'], 6)
            hv = np.round(performance['hv'], 6)
            hv_ratio = np.round(performance['normalized_hv'], 6)
        if self.verbose:
            content = [self.n_eval, igd_val, igd_plus_val, hv_val, hv_ratio_val, igd, igd_plus, hv, hv_ratio]
            print("\033[92m{:<10}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[96m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m | \033[93m{:^20.6f}\033[00m |".format(*content))
        self.res_logged.append(
            [self.n_eval, None, igd_val, igd_plus_val, hv_val, hv_ratio_val, igd, igd_plus, hv, hv_ratio])
