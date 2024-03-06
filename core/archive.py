import numpy as np
from functions import compare_f1_f2, is_equal
from pymoo.core.individual import Individual

class ElitistArchive:
    """
        Note: No limit the size
    """
    def __init__(self):
        self.archive = []
        self.genotypeHash_archive = []

    def add(self, list_solution):
        if isinstance(list_solution, Individual):
            list_solution = [list_solution]
        for solution in list_solution:
            self.update(solution)
        return self

    def update(self, solution, **kwargs):
        length = len(self.archive)
        notDominated = np.ones(length).astype(bool)

        genotypeHash_solution = ''.join(map(str, solution.X))
        if genotypeHash_solution not in self.genotypeHash_archive:
            # Compare to every solutions in Elitist Archive
            for i, elitist in enumerate(self.archive):
                better_sol = compare_f1_f2(f1=solution.F, f2=elitist.F)
                if better_sol == 0:  # Filter out members that are dominated by new solution
                    notDominated[i] = False
                elif better_sol == 1:  # If new solution is dominated by any member, stop the checking process
                    return
            self.archive.append(solution)
            self.genotypeHash_archive.append(genotypeHash_solution)
            notDominated = np.append(notDominated, True)
            # Update Elitist Archive
            self.archive = np.array(self.archive)[notDominated].tolist()
            self.genotypeHash_archive = np.array(self.genotypeHash_archive)[notDominated].tolist()

    def isDominated(self, other) -> bool:
        """
        Check whether the current archive is dominated by the other or not.
        Returns the better one using Pareto-dominance definition.

        :param other: the comparing archive
        :return: True or False
        """
        fitness_self = np.array([s.F for s in self.archive])
        fitness_other = np.array([s_.F for s_ in other.archive])

        checklist = []
        for i, f_s1 in enumerate(fitness_self):
            res = 'non'
            for f_s2 in fitness_other:
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