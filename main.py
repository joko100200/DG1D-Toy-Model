import math
import numpy as np
import DG1DSolver

D = 100 # number of cells
N = 50 # number of basis functions
L = 100 # number of evaluation points for basis functions on epsilon [-1,1]
a = -5.0 # left boundary
b = 5.0 # right boundary

x_grid = np.linspace(a, b, D + 1)
solver = DG1DSolver.DG1DSolver(x_grid, N, L)

solver.initialize_solution(DG1DSolver.gaussian)
solver.error_in_u0(DG1DSolver.gaussian)






