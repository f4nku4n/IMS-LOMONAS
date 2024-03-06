import copy

class Solution:
    def __init__(self, X=None, F=None, **kwargs) -> None:
        # TODO: You can replace with another mechanism to get the genotypeHash
        self.X = X
        self.F = F
        self.genotypeHash = ''.join(map(str, X))
        self.data = kwargs

    def set(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self.data[key] = value

    def copy(self):
        solution = copy.copy(self)
        solution.data = self.data.copy()
        return solution

    def get(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        if key in self.data:
            return self.data[key]
        return None

    def print(self):
        for key in self.__dict__:
            print(f'{key}: {self.__dict__[key]}')