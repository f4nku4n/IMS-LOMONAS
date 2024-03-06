"""
Source code for Local-search algorithm for Multi-Objective Neural Architecture Search (LOMONAS) (GECCO'23)
Authors: Quan Minh Phan, Ngoc Hoang Luong
"""
import random
import numpy as np
from copy import deepcopy
import itertools
from functions import not_existed, compare_f1_f2
from utilities import Result, Footprint
from core import ElitistArchive, Solution

from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from pymoo.core.population import Population
from pymoo.util.randomized_argsort import randomized_argsort
from pymoo.util.misc import find_duplicates
from sklearn.metrics.pairwise import euclidean_distances

sorter = NonDominatedSorting()
#################################################### LOMONAS #######################################################
class LOMONAS:
    def __init__(self, name='LOMONAS',
                 nF=3, check_limited_neighbors=False, neighborhood_check_on_potential_sols=False, alpha=210,
                 selection_method=None,
                 limit_Q_size=True,
                 archive=None, footprint=None, debugger=None, res_logged=None,
                 evaluator=None, **kwargs):
        """
        - name (str) -> the algorithm name (i.e., LOMONAS)
        - nF (int) -> number of kept front for neighborhood checking
        - check_limited_neighbors (bool) -> checking a limited neighbors when local search?
        - neighborhood_check_on_potential_sols (bool) -> local search on potential or all solutions?
        - alpha (int, [0, 360]) -> angle for checking knee solution or not
        """
        self.name = name

        self.nF = nF
        self.check_limited_neighbors = check_limited_neighbors
        self.neighborhood_check_on_potential_sols = neighborhood_check_on_potential_sols
        self.alpha = alpha

        self.debugger = debugger
        self.footprint = Footprint() if footprint is None else footprint
        self.res_logged = [] if res_logged is None else res_logged

        self.evaluator = evaluator
        self.local_archive = ElitistArchive()
        self.archive = ElitistArchive() if archive is None else archive
        self.last_archive = None

        self.S, self.Q = [], []

        self.problem, self.max_eval = None, None
        self.selection_method = selection_method

        self.limit_Q_size = limit_Q_size

        self.solutions_collector = None
        self.neighbors_collector = None

        if self.selection_method == 'cdistance':
            self.selector = crowding_distance_selection
        elif self.selection_method == 'random':
            self.selector = random_selection
        elif self.selection_method == 'greedy':
            self.selector = greedy_selection
        else:
            raise NotImplementedError

        self.n_eval = 0
        self.NIS = 0

        self.sizeQ_logged = []
        self.last_S_fid, self.last_Q = [], []

    @property
    def hyperparameters(self):
        return {
            'optimizer': self.name,
            'NF': self.nF,
            'check_limited_neighbors': self.check_limited_neighbors,
            'neighborhood_check_on_potential_sols': self.neighborhood_check_on_potential_sols,
            'alpha': self.alpha,
            'limit_Q_size': self.limit_Q_size,
            'selection_method': self.selection_method
        }

    def set(self, key_value):
        for key, value in key_value.items():
            if key in self.__dict__:
                self.__dict__[key] = value

    """-------------------------------------------------- SETUP -----------------------------------------------"""
    def setup(self, problem, max_eval, **kwargs):
        self.problem, self.max_eval = problem, max_eval
        self._setup(**kwargs)

    def _setup(self, **kwargs):
        if self.neighborhood_check_on_potential_sols:  # If only performing neighborhood check on knee and extreme ones
            self.solutions_collector = get_potential_solutions
        else:
            self.solutions_collector = get_all_solutions

        if self.check_limited_neighbors:
            self.neighbors_collector = get_some_neighbors
        else:
            self.neighbors_collector = get_all_neighbors

    """------------------------------------------------- EVALUATE --------------------------------------------"""
    def evaluate(self, solution):
        self.n_eval += 1
        self.evaluator(solution)

        if self.evaluator.n_eval % 100 == 0:
            self.debugger(algorithm=self)
            self.sizeQ_logged.append(len(self.Q))

    """-------------------------------------------------- SOLVE -----------------------------------------------"""
    def solve(self, problem, max_eval, seed, **kwargs):
        random.seed(seed)
        np.random.seed(seed)

        self.setup(problem, max_eval)  # Setup (general)

        self._solve(**kwargs)
        genotype_archive = [elitist.X for elitist in self.archive.archive]
        res = Result(genotype_archive)
        return res

    def _solve(self, **kwargs):
        first = True
        while not self.isTerminated():  # line 5 - 27
            self.initialize(first)  # Sample new starting solution for the next local search
            first = False
            isContinued = True
            while isContinued:
                isContinued = self.neighborhood_checking()

    """-------------------------------------------------- UTILITIES -----------------------------------------------"""
    def isTerminated(self):
        if self.evaluator.n_eval >= self.max_eval:
            return True
        return False

    def update_archive(self, solution):
        self.local_archive.update(solution)
        self.archive.update(solution)

    def initialize(self, first=True):
        start_solution = self.sample_starting_solution(first=first)  # Random a starting solution (line 3)

        # lines 6, 7
        self.S, self.Q = [start_solution], [start_solution]  # approximation set (S) and queue for neighborhood check (Q)
        self.last_archive = deepcopy(self.local_archive)

    def sample_starting_solution(self, first=False):
        if first:
            start_solution = sample_solution(self.footprint.data, self.problem)
            start_solution.set('owner', self.name)
        else:
            # lines 16 - 21
            N = []

            ## Choose one elitist in the archive
            available_idx = list(range(len(self.archive.archive)))
            found_new_start = False
            while len(available_idx) != 0:
                idx = np.random.choice(available_idx)
                available_idx.remove(idx)
                selected_solution = deepcopy(self.archive.archive[idx])
                tmp_N, _ = get_all_neighbors(solution=selected_solution, H={}, problem=self.problem)
                N = [neighbor for neighbor in tmp_N if neighbor.genotypeHash not in self.footprint.data]

                if len(N) != 0:  # If all neighbors of chosen elitist are not visited, choose a random neighbor as new starting solution.
                    found_new_start = True
                    break
            if not found_new_start:  # If not, randomly sampling from the search space.
                start_solution = sample_solution(self.footprint.data, self.problem)
            else:
                idx_selected_neighbor = np.random.choice(len(N))
                start_solution = N[idx_selected_neighbor]
            start_solution.set('owner', self.name)

        self.evaluate(start_solution)
        self.update_archive(start_solution)
        return start_solution

    def neighborhood_checking(self):
        N = self.get_neighbors()  # N: neighboring set, line 9

        # lines 10 - 22
        if len(N) == 0:
            # lines 11 - 15
            for fid in range(1, self.nF):
                self.Q = self.create_Q(fid=fid)

                N = self.get_neighbors()
                if len(N) != 0:
                    break

            if len(N) == 0:
                return False

        # line 23
        for neighbor in N:
            self.evaluate(neighbor)
            self.update_archive(neighbor)
            if self.isTerminated():
                return False

        # lines 24, 25
        self.create_S(N)

        self.Q = self.create_Q(fid=0)

        # For IMS-LOMONAS
        # if is_changed(self.local_archive.archive, self.last_archive.archive):
        #     self.last_archive = deepcopy(self.local_archive)
        #     self.NIS = 0
        # else:
        #     self.NIS += 1
        return True

    def create_S(self, N):
        P = self.S + N
        F_P = [s.F for s in P]
        idx_fronts = sorter.do(np.array(F_P))
        idx_selected = np.zeros(len(F_P), dtype=bool)
        nF = min(len(idx_fronts), self.nF)
        for fid in range(nF):
            idx_selected[idx_fronts[fid]] = True
            for idx in idx_fronts[fid]:
                P[idx].set('rank', fid)
        self.S = np.array(P)[idx_selected].tolist()

        # if self.limit_Q_size:
        self.S = remove_duplicate(self.S)

    def create_Q(self, fid):
        Q, last_S_fid, duplicated = self.solutions_collector(S=self.S, fid=fid, alpha=self.alpha,
                                                             last_S_fid=self.last_S_fid, last_Q=self.last_Q)
        if not duplicated:
            self.last_Q, self.last_S_fid = deepcopy(Q), last_S_fid.copy()

        # # Limit the number of potential solutions for neighborhood checking
        # if self.limit_Q_size:
        #     # n_survive = min(len(Q), int(np.log2(self.max_eval)))
        #     # n_survive = min(len(Q), 1 + 2 * int(np.log10(self.max_eval)))
        #     n_survive = self.problem.n_obj + int(np.log10(self.max_eval))
        #     if n_survive < len(Q):
        #         pop = Population().empty(len(Q))
        #         F = np.array([s.F for s in Q])
        #         pop.set('F', F)
        #
        #         I = self.selector(pop=pop, n_survive=n_survive)
        #         Q = np.array(Q)[I].tolist()

        return Q

    def get_neighbors(self):
        """ Get neighbors of all solutions in queue Q, but discard solutions that has been already in H """
        _H = self.footprint.data
        N = []
        for solution in self.Q:
            tmp_N, _H = self.neighbors_collector(solution, _H, self.problem)

            # Remove duplication
            genotypeHash_S = [s.genotypeHash for s in self.S]
            genotypeHash_N = [s.genotypeHash for s in N]
            for neighbor in tmp_N:
                if not_existed(neighbor.genotypeHash, S=genotypeHash_S, N=genotypeHash_N):
                    neighbor.set('owner', self.name)
                    N.append(neighbor)
                    genotypeHash_N.append(neighbor.genotypeHash)
        self.footprint.data = _H
        return N

#####################################################################################
def is_sol_dominated(solution, archive):
    genotypeHash_archive = [s.genotypeHash for s in archive]
    if solution.genotypeHash not in genotypeHash_archive:
        # Compare to every solutions in Elitist Archive
        for i, elitist in enumerate(archive):
            better_sol = compare_f1_f2(f1=solution.F, f2=elitist.F)
            if better_sol == 1:  # If new solution is dominated by any member, stop the checking process
                return True
        return False
    return True


def is_changed(archive_1, archive_2):
    fitness_archive_1 = np.sum(np.unique([s.F for s in archive_1], axis=0), axis=0)
    fitness_archive_2 = np.sum(np.unique([s.F for s in archive_2], axis=0), axis=0)
    if np.inf in fitness_archive_1 or np.inf in fitness_archive_2:
        return True
    return not np.all(fitness_archive_1 == fitness_archive_2)


def seeking(list_sol, alpha):
    list_sol = np.array(list_sol)
    non_dominated_front = np.array([solution.F for solution in list_sol])

    ids = range(non_dominated_front.shape[-1])
    info_potential_sols_all = []
    for f_ids in itertools.combinations(ids, 2):
        f_ids = np.array(f_ids)
        obj_1, obj_2 = f'{f_ids[0]}', f'{f_ids[1]}'

        _non_dominated_front = non_dominated_front[:, f_ids].copy()

        ids_sol = np.array(list(range(len(list_sol))))
        ids_fr0 = sorter.do(_non_dominated_front, only_non_dominated_front=True)

        ids_sol = ids_sol[ids_fr0]
        _non_dominated_front = _non_dominated_front[ids_fr0]

        sorted_idx = np.argsort(_non_dominated_front[:, 0])

        ids_sol = ids_sol[sorted_idx]
        _non_dominated_front = _non_dominated_front[sorted_idx]

        min_values, max_values = np.min(_non_dominated_front, axis=0), np.max(_non_dominated_front, axis=0)
        _non_dominated_front_norm = (_non_dominated_front - min_values) / (max_values - min_values)

        info_potential_sols = [
            [0, list_sol[ids_sol[0]], f'best_f{obj_1}']  # (idx (in full set), property)
        ]

        l_non_front = len(_non_dominated_front)
        for i in range(l_non_front - 1):
            if np.sum(np.abs(_non_dominated_front[i] - _non_dominated_front[i + 1])) != 0:
                break
            else:
                info_potential_sols.append([i + 1, list_sol[ids_sol[i + 1]], f'best_f{obj_1}'])

        for i in range(l_non_front - 1, -1, -1):
            if np.sum(np.abs(_non_dominated_front[i] - _non_dominated_front[i - 1])) != 0:
                break
            else:
                info_potential_sols.append([i - 1, list_sol[ids_sol[i - 1]], f'best_f{obj_2}'])
        info_potential_sols.append([l_non_front - 1, list_sol[ids_sol[l_non_front - 1]], f'best_f{obj_2}'])

        ## find the knee solutions
        start_idx, end_idx = 0, l_non_front - 1

        for i in range(len(info_potential_sols)):
            if info_potential_sols[i + 1][-1] == f'best_f{obj_2}':
                break
            else:
                start_idx = info_potential_sols[i][0] + 1

        for i in range(len(info_potential_sols) - 1, -1, -1):
            if info_potential_sols[i - 1][-1] == f'best_f{obj_1}':
                break
            else:
                end_idx = info_potential_sols[i][0] - 1

        for i in range(start_idx, end_idx + 1):
            l = None
            h = None
            for m in range(i - 1, -1, -1):
                if np.sum(np.abs(_non_dominated_front[m] - _non_dominated_front[i])) != 0:
                    l = m
                    break
            for m in range(i + 1, l_non_front, 1):
                if np.sum(np.abs(_non_dominated_front[m] - _non_dominated_front[i])) != 0:
                    h = m
                    break

            if (h is not None) and (l is not None):
                position = above_or_below(considering_pt=_non_dominated_front[i],
                                          remaining_pt_1=_non_dominated_front[l],
                                          remaining_pt_2=_non_dominated_front[h])
                if position == -1:
                    angle_measure = calc_angle_measure(considering_pt=_non_dominated_front_norm[i],
                                                            neighbor_1=_non_dominated_front_norm[l],
                                                            neighbor_2=_non_dominated_front_norm[h])
                    if angle_measure > alpha:
                        info_potential_sols.append([i, list_sol[ids_sol[i]], 'knee'])
        info_potential_sols_all += info_potential_sols
    return info_potential_sols_all


def above_or_below(considering_pt, remaining_pt_1, remaining_pt_2):
    """
    This function is used to check if the considering point is above or below
    the line connecting two remaining points.\n
    1: above\n
    -1: below
    """
    orthogonal_vector = remaining_pt_2 - remaining_pt_1
    line_connecting_pt1_and_pt2 = -orthogonal_vector[1] * (considering_pt[0] - remaining_pt_1[0]) \
                                  + orthogonal_vector[0] * (considering_pt[1] - remaining_pt_1[1])
    if line_connecting_pt1_and_pt2 > 0:
        return 1
    return -1


def calc_angle_measure(considering_pt, neighbor_1, neighbor_2):
    """
    This function is used to calculate the angle measure is created by the considering point
    and two its nearest neighbors
    """
    line_1 = neighbor_1 - considering_pt
    line_2 = neighbor_2 - considering_pt
    cosine_angle = (line_1[0] * line_2[0] + line_1[1] * line_2[1]) \
                   / (np.sqrt(np.sum(line_1 ** 2)) * np.sqrt(np.sum(line_2 ** 2)))
    if cosine_angle < -1:
        cosine_angle = -1
    if cosine_angle > 1:
        cosine_angle = 1
    angle = np.arccos(cosine_angle)
    return 360 - np.degrees(angle)


def sample_solution(footprint_data, problem):
    while True:
        solution = Solution(X=problem.sample())
        if solution.genotypeHash not in footprint_data:
            return solution


def get_all_solutions(S, fid, **kwargs):
    Q = []
    Q_genotypeHash = []
    rank_S = np.array([s.get('rank') for s in S])
    S_front_i = np.array(S)[rank_S == fid]

    list_genotypeHash = [s.genotypeHash for s in S_front_i]
    if is_duplicated(list_genotypeHash, kwargs['last_S_fid']):
        return kwargs['last_Q'], list_genotypeHash, True

    for sol in S_front_i:
        if sol.genotypeHash not in Q_genotypeHash:
            Q_genotypeHash.append(sol.genotypeHash)
            Q.append(sol)
    return Q, list_genotypeHash, False


def get_potential_solutions(S, fid, **kwargs):
    alpha = kwargs['alpha']
    Q = []
    Q_genotypeHash = []
    rank_S = np.array([s.get('rank') for s in S])
    S_front_i = np.array(S)[rank_S == fid]

    list_genotypeHash = [s.genotypeHash for s in S_front_i]
    if is_duplicated(list_genotypeHash, kwargs['last_S_fid']):
        return kwargs['last_Q'], list_genotypeHash, True

    info_potential_sols = seeking(S_front_i, alpha)
    potential_sols = [info[1] for info in info_potential_sols]
    for i, sol in enumerate(potential_sols):
        if sol.genotypeHash not in Q_genotypeHash:
            Q_genotypeHash.append(sol.genotypeHash)
            Q.append(sol)
    return Q, list_genotypeHash, False


## Get neighboring architectures
def get_some_neighbors(solution, H, problem):
    X, genotypeHash = solution.X, solution.genotypeHash
    N = []

    if genotypeHash in H:
        if len(H[genotypeHash]) == 0:
            return [], H
        available_idx = H[genotypeHash]
        idx_replace = np.random.choice(available_idx)
        H[genotypeHash].remove(idx_replace)
    else:
        available_idx = list(range(len(X)))
        H[genotypeHash] = available_idx
        idx_replace = np.random.choice(H[genotypeHash])
        H[genotypeHash].remove(idx_replace)

    available_ops = problem.benchmark.search_space.categories[idx_replace]
    available_ops_at_idx_replace = available_ops.copy()
    available_ops_at_idx_replace.remove(X[idx_replace])
    for op in available_ops_at_idx_replace:
        X = solution.X.copy()
        X[idx_replace] = op
        neighbor = Solution(X=X)
        N.append(neighbor)
    return N, H


def get_all_neighbors(solution, H, problem):
    X, genotypeHash = solution.X, solution.genotypeHash
    if genotypeHash in H:
        return [], H
    else:
        H[genotypeHash] = []
    N = []

    available_idx = list(range(len(X)))

    for idx_replace in available_idx:
        available_ops = problem.benchmark.search_space.categories[idx_replace]

        available_ops_at_idx_replace = available_ops.copy()
        available_ops_at_idx_replace.remove(X[idx_replace])
        for op in available_ops_at_idx_replace:
            X = solution.X.copy()
            X[idx_replace] = op
            neighbor = Solution(X=X)
            N.append(neighbor)
    return N, H

def is_duplicated(genotypeHash_list1, genotypeHash_list2):
    if len(genotypeHash_list1) != len(genotypeHash_list1):
        return False
    for genotypeHash in genotypeHash_list1:
        if genotypeHash not in genotypeHash_list2:
            return False
    return True

########################################################################################################################
def calc_crowding_distance(F, filter_out_duplicates=True):
    n_points, n_obj = F.shape

    if n_points <= 2:
        return np.full(n_points, np.inf)

    else:

        if filter_out_duplicates:
            # filter out solutions which are duplicates - duplicates get a zero finally
            is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
        else:
            # set every point to be unique without checking it
            is_unique = np.arange(n_points)

        # index the unique points of the array
        _F = F[is_unique]

        # sort each column and get index
        I = np.argsort(_F, axis=0, kind='mergesort')

        # sort the objective space values for the whole matrix
        _F = _F[I, np.arange(n_obj)]

        # calculate the distance from each point to the last and next
        dist = np.row_stack([_F, np.full(n_obj, np.inf)]) - np.row_stack([np.full(n_obj, -np.inf), _F])

        # calculate the norm for each objective - set to NaN if all values are equal
        norm = np.max(_F, axis=0) - np.min(_F, axis=0)
        norm[norm == 0] = np.nan

        # prepare the distance to last and next vectors
        dist_to_last, dist_to_next = dist, np.copy(dist)
        dist_to_last, dist_to_next = dist_to_last[:-1] / norm, dist_to_next[1:] / norm

        # if we divide by zero because all values in one columns are equal replace by none
        dist_to_last[np.isnan(dist_to_last)] = 0.0
        dist_to_next[np.isnan(dist_to_next)] = 0.0

        # sum up the distance to next and last and norm by objectives - also reorder from sorted list
        J = np.argsort(I, axis=0)
        _cd = np.sum(dist_to_last[J, np.arange(n_obj)] + dist_to_next[J, np.arange(n_obj)], axis=1) / n_obj

        # save the final vector which sets the crowding distance for duplicates to zero to be eliminated
        crowding = np.zeros(n_points)
        crowding[is_unique] = _cd

    return crowding

def crowding_distance_selection(pop, n_survive, **kwargs):
    # get the objective space values and objects
    F = pop.get("F").astype(float, copy=False)

    # calculate the crowding distance of the front
    crowding_of_front = calc_crowding_distance(F)

    # save crowding in the individual class
    for i in range(len(pop)):
        pop[i].set("crowding", crowding_of_front[i])

    # current front sorted by crowding distance if splitting
    I = randomized_argsort(crowding_of_front, order='descending', method='numpy')
    I = I[:n_survive]
    return I

def greedy_selection(pop, n_survive, **kwargs):
    F = pop.get('F')
    front = [{'fitness': [f], 'index': i} for i, f in enumerate(F)]
    selected_solutions = []

    front = list(front)
    front = sorted(front, key=lambda x: -x['fitness'][-1][0])

    cur_selected_indices = set([])

    selected_solutions.append(front[0])
    cur_selected_indices.add(0)
    while len(selected_solutions) < n_survive:
        points1 = np.array([x['fitness'][-1] for x in front])
        points2 = np.array([x['fitness'][-1] for x in selected_solutions])

        distances = euclidean_distances(points1, points2)
        cur_min_distances = np.min(distances, axis=1)

        ind_with_max_dist = -1
        max_dist = -float("inf")
        for j in range(len(front)):
            if j not in cur_selected_indices and cur_min_distances[j] > max_dist:
                max_dist = cur_min_distances[j]
                ind_with_max_dist = j
        selected_solutions.append(front[ind_with_max_dist])
        cur_selected_indices.add(ind_with_max_dist)
    I = [s['index'] for s in selected_solutions]
    return I

# def random_selection(pop, n_survive, **kwargs):
#     I = np.random.choice(range(len(pop)), n_survive, replace=False)
#     return I

def random_selection(pop, n_survive, **kwargs):
    F = pop.get('F')
    all_idx = list(range(len(pop)))
    I = []
    for i in range(F.shape[1]):
        idx = np.argmin(F[:, i])
        I.append(idx)
        try:
            all_idx.remove(idx)
        except ValueError:
            pass
    I_ = np.random.choice(all_idx, n_survive - F.shape[1], replace=False).tolist()
    I += I_
    return I

def remove_duplicate(pop):
    F = np.array([idv.F for idv in pop])
    is_unique = np.where(np.logical_not(find_duplicates(F, epsilon=1e-32)))[0]
    return np.array(pop)[is_unique].tolist()
