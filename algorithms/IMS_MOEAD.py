"""
Source code for Interleaved Multi-start Scheme MOEA/D (IMS-MOEAD)
Authors: Quan Minh Phan, Ngoc Hoang Luong
"""
import numpy as np
from core import ElitistArchive
from algorithms import IMS_NSGA2
from pymoo.algorithms.moo.moead import MOEAD

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair

from pymoo.util.ref_dirs import get_reference_directions
from test_suites.utils import define_H

def get_genetic_operator(crx_prob=1.0,  # crossover probability
                         crx_eta=20.0,  # SBX crossover eta
                         mut_prob=0.9,  # mutation probability
                         mut_eta=20.0,  # polynomial mutation hyperparameter eta
                         ):
    sampling = IntegerRandomSampling()
    crossover = SBX(prob=crx_prob, eta=crx_eta, repair=RoundingRepair(), vtype=int)
    mutation = PM(prob=mut_prob, eta=mut_eta, repair=RoundingRepair(), vtype=int)
    return sampling, crossover, mutation


class IMS_MOEAD(IMS_NSGA2):
    def __init__(self,
                 base=2, init_popsize=10,
                 perform_termination=True,
                 verbose=False
                 ):
        super().__init__(base=base, init_popsize=init_popsize, perform_termination=perform_termination, verbose=verbose)
        self.name = 'IMS-MOEAD'

    ######################################################## Main ######################################################
    def _solve(self, problem, max_eval, **kwargs):
        LOP, LOP_name = [], []  # list of processors
        self.archive = ElitistArchive()

        self.problem = problem
        self.max_eval = max_eval

        self.sampling, self.crossover, self.mutation = get_genetic_operator(crx_prob=1.0, crx_eta=20,
                                                                            mut_prob=1/self.problem.n_var, mut_eta=20.0)

        hat = 1
        n_iter, id_worker = 1, 1
        pop_size = self.init_popsize

        termination = ('n_eval', max_eval)
        n_partitions = define_H(self.problem.n_obj, pop_size)
        ref_dirs = get_benchmark_settings(self.problem.n_obj, n_partitions)

        new_processor = self.initialize_processor(pop_size=pop_size, ref_dirs=ref_dirs, termination=termination, verbose=False)
        # Initialize a new processor
        LOP.append(new_processor)
        LOP_name.append(f'MOEAD #{id_worker - 1} (#ref_points = {len(ref_dirs)})')
        isKilled = [False]

        self.n_eval = 0
        while True:
            if n_iter == self.base ** hat:
                pop_size *= 2
                id_worker += 1

                n_partitions = define_H(self.problem.n_obj, pop_size)
                ref_dirs = get_benchmark_settings(self.problem.n_obj, n_partitions)

                new_processor = self.initialize_processor(ref_dirs=ref_dirs, pop_size=pop_size,
                                                          termination=termination, verbose=False)  # Initialize a new processor
                LOP.append(new_processor)  # Add the new processor to the last position of LOP
                LOP_name.append(f'MOEAD #{id_worker - 1} (#ref_points = {len(ref_dirs)})')
                isKilled.append(False)
                hat += 1

            for i, processor in enumerate(LOP):
                if n_iter % (self.base ** i) == 0:
                    # if not isKilled[i]:
                    processor.next()
                    self.n_eval += processor.problem.eval_his[-1]
                    self.updateArchive(processor.problem.archive)
                    self.debug()

                if self.isTerminated():
                    return

            if self.perform_termination:
                isKilled = self.selection_3(isKilled, LOP, LOP_name)

                # Reset the counter
                isKilled = np.array(isKilled)
                LOP = np.array(LOP)[~isKilled].tolist()
                LOP_name = np.array(LOP_name)[~isKilled].tolist()
                isKilled = isKilled[~isKilled].tolist()

            n_iter += 1

    ##################################################### Utilities ####################################################
    def initialize_processor(self, pop_size, **kwargs):
        processor = MOEAD(ref_dirs=kwargs['ref_dirs'], n_neighbors=20,
                          prob_neighbor_mating=0.9,
                          sampling=self.sampling, crossover=self.crossover, mutation=self.mutation,
                          archive=ElitistArchive(), eliminate_duplicates=True)
        processor.setup(self.problem, **kwargs)

        return processor

def get_benchmark_settings(n_obj, n_partitions):
    if n_obj == 2:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    elif n_obj == 3:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    elif n_obj == 4:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    elif n_obj == 5:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)
    elif n_obj == 6:
        ref_dirs = get_reference_directions(
            "multi-layer",
            get_reference_directions("das-dennis", n_obj, n_partitions=4, scaling=1.0),
            get_reference_directions("das-dennis", n_obj, n_partitions=1, scaling=0.5))
    elif n_obj == 8:
        ref_dirs = get_reference_directions(
            "multi-layer",
            get_reference_directions("das-dennis", n_obj, n_partitions=3, scaling=1.0),
            get_reference_directions("das-dennis", n_obj, n_partitions=2, scaling=0.5))
    else:
        raise NotImplementedError

    return ref_dirs