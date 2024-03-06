import os
import sys
import json
import logging
import argparse
import pickle as p
import numpy as np
from tqdm import tqdm
from pymoo.optimize import minimize

from evoxbench.database.init import config
from core import Evaluator
from utilities import CustomCallback, Debugger

from factory import get_algorithm, get_problem


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def main(kwargs):
    experiment_stats = []

    database_path, data_path = kwargs.database_path, kwargs.data_path

    config(database_path, data_path)

    pid = kwargs.pid
    test_suite = kwargs.test_suite
    max_eval = kwargs.max_eval
    res_path = kwargs.res_path
    if not os.path.exists(res_path):
        os.mkdir(res_path)

    verbose = bool(kwargs.verbose)
    log_results = bool(kwargs.log_results)
    using_archive = bool(kwargs.using_archive)

    list_hv, list_search_cost = [], []
    n_run = kwargs.n_run

    problem = get_problem(test_suite, pid, using_archive=using_archive)
    debugger = Debugger(test_suite=test_suite, verbose=bool(kwargs.verbose))

    for rid in tqdm(range(n_run)):
        logging.info(f'Run: {rid + 1}')

        problem.reset()
        evaluator = Evaluator(problem)
        algo = get_algorithm(
            name=kwargs.optimizer, problem=problem,
            nF=kwargs.nF, neighborhood_check_on_potential_sols=bool(kwargs.neighborhood_check_on_potential_sols),
            check_limited_neighbors=bool(kwargs.check_limited_neighbors), alpha=kwargs.alpha,
            selection_method=kwargs.selection_method,
            limit_Q_size=bool(kwargs.limit_Q_size),
            base=kwargs.base, init_popsize=kwargs.init_popsize,
            using_archive=using_archive,
            allow_duplicates=bool(kwargs.allow_duplicates),
            perform_termination=bool(kwargs.perform_termination), evaluator=evaluator, debugger=debugger,
            verbose=bool(kwargs.verbose)
        )
        seed = rid * 100
        run_stats = {'run': rid + 1, 'seed': seed, 'max_eval': max_eval}
        search_cost = 0.0

        if kwargs.optimizer in ['lomonas', 'rs', 'ims-lomonas', 'ims-nsga2', 'ims-nsga3', 'ims-moead', 'nsga2-lomonas']:
            res = algo.solve(problem=problem, seed=seed, max_eval=max_eval, test_suite=test_suite, verbose=verbose)
            genotype_list = res.genotype_list
            front_archive = np.array([s.F for s in algo.archive.archive])
            if 'Our' in problem.benchmark.name:
                search_cost = algo.problem.benchmark.evaluator.search_cost
                algo.problem.benchmark.evaluator.search_cost = 0.0
        else:
            res = minimize(problem, algo, ('n_eval', max_eval), seed=seed, verbose=False,
                           callback=CustomCallback(test_suite=test_suite, verbose=verbose, using_archive=using_archive))
            if using_archive:
                genotype_list = np.array([s.X for s in res.algorithm.archive.archive])
                front_archive = np.array([s.F for s in res.algorithm.archive.archive])
            else:
                genotype_list = res.algorithm.opt.get('X')
                front_archive = res.algorithm.opt.get('F')

        hv = np.round(problem.calc_perf_indicator(genotype_list, 'hv'), 6)

        hashKey_list = []
        if '101' in problem.benchmark.name and 'Our' in problem.benchmark.name:
            from test_suites.nb101.api import ModelSpec
            from test_suites.nb101.api import NASBench

            api = NASBench()
            phenotype_list = problem.benchmark.search_space.decode(genotype_list)
            for phenotype in phenotype_list:
                model_spec = ModelSpec(phenotype['matrix'], phenotype['ops'])
                module_hash = api.getModuleHash(model_spec)
                hashKey_list.append(module_hash)

            archive = {'X': genotype_list, 'hashKey': hashKey_list, 'F': front_archive}

        ################################################## Log results #################################################
        if log_results:
            p.dump(archive, open(f'{res_path}/{test_suite}-mop{pid}_run{rid}_{kwargs.optimizer}_archive.p', 'wb'))

        logging.info(f'HV: {hv}')
        logging.info(f'Search Cost: {search_cost} seconds')
        print("-" * 196)

        list_hv.append(hv)
        list_search_cost.append(search_cost)
        run_stats['HV'] = list_hv[-1]

        experiment_stats.append(run_stats)

    logging.info(f'Average HV: {np.round(np.mean(list_hv), 4)} ({np.round(np.std(list_hv), 4)})')
    logging.info(f'Search Cost (avg): {np.round(np.mean(list_search_cost))}')
    print("-" * 196)

    if log_results:
        with open(f'{res_path}/{test_suite}-mop{pid}_{kwargs.optimizer}.json', 'w') as fp:
            json.dump(experiment_stats, fp, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('CEC')

    ''' PROBLEM '''
    parser.add_argument('--pid', type=int, default=1)
    parser.add_argument('--test_suite', type=str, default='cec-c10', choices=['cec-c10', 'cec-in1k', 'gecco'])
    parser.add_argument('--max_eval', type=int, default=3000)

    ''' ALGORITHM '''
    parser.add_argument('--optimizer', type=str, default='lomonas', choices=['lomonas', 'rs', 'nsga2', 'nsga3', 'moead',
                                                                             'ims-lomonas', 'ims-nsga2', 'ims-nsga3',
                                                                             'ims-moead',
                                                                             'nsga2-lomonas'])
    parser.add_argument('--nF', type=int, default=3, help='#kept front for neighborhood check (LOMONAS)')
    parser.add_argument('--check_limited_neighbors', action='store_true',
                        help='only compare to a fixed number of neighbors (LOMONAS)')
    parser.add_argument('--neighborhood_check_on_potential_sols', action='store_true',
                        help='perform neighborhood check on knee and extreme solutions (LOMONAS)')
    parser.add_argument('--limit_Q_size', action='store_true',
                        help='limit #solutions for performing neighborhood (LOMONAS)')
    parser.add_argument('--selection_method', type=str, default='cdistance', choices=['greedy', 'cdistance', 'random'])
    parser.add_argument('--alpha', type=int, default=210)

    parser.add_argument('--base', type=int, default=2, help='for IMS variants')

    parser.add_argument('--using_archive', help='using archive (for MOEAs (pymoo))', action='store_true')
    parser.add_argument('--allow_duplicates', action='store_true', help='random search with duplicates')
    parser.add_argument('--perform_termination', help='terminating inefficient processors (IMS-variants)',
                        action='store_true')
    parser.add_argument('--init_popsize', type=int, default=10, help='for IMS-MOEAs variants')
    parser.add_argument('--verbose', action='store_true')

    ''' ENVIRONMENT '''
    parser.add_argument('--n_run', type=int, default=31)
    parser.add_argument('--database_path', type=str, default=None, help='path for loading api benchmark (database CEC)')
    parser.add_argument('--data_path', type=str, default=None, help='path for loading api benchmark (data CEC)')
    parser.add_argument('--res_path', type=str, default='./res', help='path for saving results')
    parser.add_argument('--log_results', help='log results', action='store_true')
    args = parser.parse_args()

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    main(args)