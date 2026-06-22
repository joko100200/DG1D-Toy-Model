import os
import sys
import inspect
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import DG1DSolver

# -------------------------
# PARAMETERS
# -------------------------

N           = 5
L           = 100
left_bound  = -60.0
right_bound = 80.0
R           = 60.0
P           = 4
T           = 600
cfl         = 0.1
D           = 500

M = 1.0
ll = 2

x_grid = np.linspace(left_bound, right_bound, D + 1)
solver = DG1DSolver.DG1DSolver(x_grid, N, L, R, P, M, ll, "probes/TmpTest.csv")

solver.initialize_solution(DG1DSolver.gaussian_pulse)
if os.path.isfile("probes/TmpTest.csv"):
    print("Loading file")
    coarse_probe = np.loadtxt("probes/TmpTest.csv", delimiter=" ", skiprows=1)
    solver._probe_buffer = coarse_probe.tolist()
else:
    print("RUnning sim")
    #solver.plot_solution(0.0, plot_exact=False)
    #exit()
    solver.run(T, cfl)

solver.plot_RingDown()
#solver.plot_tail()
#solver.FFT_scri()


