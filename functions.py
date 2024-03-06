import numpy as np
from numpy import ndarray

def compare_f1_f2(f1: ndarray, f2: ndarray) -> int:
    """
    Takes in the objective function values of two solution (f1, f2). Returns the better one using Pareto-dominance definition.

    :param f1: the objective function values of the first solution
    :param f2: the objective function values of the second solution
    :return: -1 (no one is better); 0 (f1 is better); or 1 (f2 is better)
    """
    x_better = np.all(f1 <= f2)
    y_better = np.all(f2 <= f1)
    if x_better == y_better:
        return -1
    if y_better:  # False - True
        return 1
    return 0  # True - False


def is_equal(f1: ndarray, f2: ndarray) -> bool:
    """
    Takes in the objective function values of two solution (f1, f2.)
    Returns the better one using Pareto-dominance definition.

    :param f1: the objective function values of the first solution
    :param f2: the objective function values of the second solution
    :return: True or False
    """
    return np.all(f1 == f2)

def is_front_dominated(front_1: ndarray, front_2: ndarray) -> bool:
    """
    Takes in the objective function values of two fronts (front_1, front_2).
    Returns the better one using Pareto-dominance definition.

    :param front_1: the objective function values of the first front
    :param front_2: the objective function values of the second front
    :return: True or False
    """
    checklist = []
    for i, f_s1 in enumerate(front_1):
        res = 'non'
        for f_s2 in front_2:
            better_sol = compare_f1_f2(f1=f_s1, f2=f_s2)
            if better_sol == 1:
                res = 'dom'
                break
            elif better_sol == -1:
                if is_equal(f1=f_s1, f2=f_s2):
                    res = 'eq'
        checklist.append(res)
    checklist = np.array(checklist)
    if np.all(checklist == 'dom'):
        return True
    if np.any(checklist == 'non'):
        return False
    return True

def get_front_zero_indices(F: ndarray) -> ndarray:
    """
    Takes in a set of objective function values F.
    Returns the indices of solutions that are on the front zero (non-dominated front).

    :param F: the set of objective function values
    :return: indices of solutions on the front zero.
    """
    l = len(F)
    r = np.zeros(l, dtype=int)
    for i in range(l):
        if r[i] == 0:
            for j in range(i + 1, l):
                better_sol = compare_f1_f2(F[i], F[j])
                if better_sol == 0:
                    r[j] = 1
                elif better_sol == 1:
                    r[i] = 1
                    break
    return r == 0

def not_existed(genotypeHash: str, **kwargs) -> bool:
    """
    Takes in the fingerprint of a solution and a set of checklists.
    Return True if the current solution have not existed on the set of checklists.

    :param genotypeHash: the fingerprint of the considering solution
    :return: True or False
    """
    return np.all([genotypeHash not in kwargs[L] for L in kwargs])


