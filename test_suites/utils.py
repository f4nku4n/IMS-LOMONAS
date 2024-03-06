import math

def nCr(n, r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def define_H(n_obj, n_sol):
    H = 0
    while True:
        H += 1
        if nCr(H + n_obj - 1, n_obj - 1) > n_sol:
            return H - 1
