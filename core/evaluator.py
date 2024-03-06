class Evaluator:
    def __init__(self, problem):
        self.problem = problem
        self.cache = {}
        self.n_eval = 0

    def __call__(self, solution):
        self.n_eval += 1
        try:
            solution.F = self.cache[solution.genotypeHash]
        except KeyError:
            f = self.problem.evaluate(solution.X)
            self.cache[solution.genotypeHash] = f
            solution.F = f