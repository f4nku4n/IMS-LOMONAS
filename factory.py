from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.util.ref_dirs import get_reference_directions

from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.repair.rounding import RoundingRepair

from core import NASProblem, ElitistArchive
from algorithms import LOMONAS
from algorithms import IMS_LOMONAS, IMS_NSGA2, IMS_NSGA3, IMS_MOEAD

def get_genetic_operator(crx_prob=1.0,  # crossover probability
                         crx_eta=30.0,  # SBX crossover eta
                         mut_prob=0.9,  # mutation probability
                         mut_eta=20.0,  # polynomial mutation hyperparameter eta
                         ):
    sampling = IntegerRandomSampling()
    crossover = SBX(prob=crx_prob, eta=crx_eta, repair=RoundingRepair(), vtype=int)
    mutation = PM(prob=mut_prob, eta=mut_eta, repair=RoundingRepair(), vtype=int)
    return sampling, crossover, mutation


def get_benchmark_settings(n_obj):
    n_gen = 100

    if n_obj == 2:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=99)
    elif n_obj == 3:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=13)
    elif n_obj == 4:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=7)
    elif n_obj == 5:
        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=5)
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

    pop_size = ref_dirs.shape[0]
    return pop_size, n_gen, ref_dirs

def nsga2(pop_size,
          crx_prob=0.9,  # crossover probability
          crx_eta=20.0,  # SBX crossover eta
          mut_eta=20.0,  # polynomial mutation hyperparameter eta
          **kwargs):

    sampling, crossover, mutation = get_genetic_operator(crx_prob=crx_prob, crx_eta=crx_eta,
                                                         mut_prob=1/kwargs['n_var'], mut_eta=mut_eta)
    archive = ElitistArchive() if kwargs['using_archive'] else None
    return NSGA2(pop_size=pop_size, sampling=sampling, crossover=crossover, mutation=mutation,
                 archive=archive, eliminate_duplicates=True)


def moead(ref_dirs,
          crx_prob=1.0,  # crossover probability
          crx_eta=20.0,  # SBX crossover eta
          mut_eta=20.0,  # polynomial mutation hyperparameter eta
          neighborhood_size=20,  # neighborhood size
          prob_neighbor_mating=0.9,  # neighborhood selection probability
          **kwargs):

    sampling, crossover, mutation = get_genetic_operator(crx_prob=crx_prob, crx_eta=crx_eta,
                                                         mut_prob=1/kwargs['n_var'], mut_eta=mut_eta)
    archive = ElitistArchive() if kwargs['using_archive'] else None

    return MOEAD(ref_dirs=ref_dirs, n_neighbors=neighborhood_size, prob_neighbor_mating=prob_neighbor_mating,
                 archive=archive, sampling=sampling, crossover=crossover, mutation=mutation)


def nsga3(pop_size,
          ref_dirs,
          crx_prob=1.0,  # crossover probability
          crx_eta=30.0,  # SBX crossover eta
          mut_eta=20.0,  # polynomial mutation hyperparameter eta
          **kwargs):

    sampling, crossover, mutation = get_genetic_operator(crx_prob=crx_prob, crx_eta=crx_eta,
                                                         mut_prob=1/kwargs['n_var'], mut_eta=mut_eta)
    archive = ElitistArchive() if kwargs['using_archive'] else None
    return NSGA3(pop_size=pop_size, ref_dirs=ref_dirs, sampling=sampling, crossover=crossover,
                 archive=archive, mutation=mutation, eliminate_duplicates=True)

def get_algorithm(name, problem, **kwargs):
    pop_size, n_gen, ref_dirs = get_benchmark_settings(problem.n_obj)
    n_var = problem.n_var
    name = name.upper()
    if name == 'NSGA2':
        return nsga2(pop_size, n_var=n_var, using_archive=kwargs['using_archive'])
    elif name == 'NSGA3':
        return nsga3(pop_size, ref_dirs, n_var=n_var, using_archive=kwargs['using_archive'])
    elif name == 'MOEAD':
        return moead(ref_dirs, n_var=n_var, using_archive=kwargs['using_archive'])
    elif name == 'LOMONAS':
        return LOMONAS(nF=kwargs['nF'], neighborhood_check_on_potential_sols=kwargs['neighborhood_check_on_potential_sols'],
                       check_limited_neighbors=kwargs['check_limited_neighbors'], alpha=kwargs['alpha'],
                       evaluator=kwargs['evaluator'], debugger=kwargs['debugger'])
    elif name == 'IMS-LOMONAS':
        return IMS_LOMONAS(base=kwargs['base'],
                           neighborhood_check_on_potential_sols=kwargs['neighborhood_check_on_potential_sols'],
                           check_limited_neighbors=kwargs['check_limited_neighbors'], alpha=kwargs['alpha'],
                           perform_termination=kwargs['perform_termination'],
                           evaluator=kwargs['evaluator'], debugger=kwargs['debugger'])
    elif name == 'IMS-NSGA2':
        return IMS_NSGA2(base=kwargs['base'], perform_termination=kwargs['perform_termination'],
                         init_popsize=kwargs['init_popsize'], verbose=kwargs['verbose'])
    elif name == 'IMS-NSGA3':
        return IMS_NSGA3(base=kwargs['base'], perform_termination=kwargs['perform_termination'],
                         init_popsize=kwargs['init_popsize'], verbose=kwargs['verbose'])
    elif name == 'IMS-MOEAD':
        return IMS_MOEAD(base=kwargs['base'], perform_termination=kwargs['perform_termination'],
                         init_popsize=kwargs['init_popsize'], verbose=kwargs['verbose'])
    else:
        NotImplementedError()

def get_problem(test_suite, pid, **kwargs):
    if test_suite == 'cec-c10':
        from evoxbench.test_suites import c10mop
        benchmark = c10mop(pid)
    elif test_suite == 'cec-in1k':
        from evoxbench.test_suites import in1kmop
        benchmark = in1kmop(pid)
    elif test_suite == 'gecco':
        from test_suites import mop
        benchmark = mop(pid)
    else:
        raise NotImplementedError()
    benchmark.normalized_objectives = False
    return NASProblem(benchmark, test_suite=test_suite, **kwargs)
