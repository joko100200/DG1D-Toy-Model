import numpy as np
import DG1DSolver
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------
N = 5                 # polynomial degree (fixed)
L = 100                # quadrature resolution
left_bound = -10.0
right_bound = 10.0

T = 1.0
cfl = 0.2

D_values = [20, 40, 80, 160, 320]

errors = []
hs = []

# -------------------------
# CONVERGENCE LOOP
# -------------------------
for D in D_values:

    print("\n==============================")
    print(f"Running D = {D}, N = {N}")
    print("==============================")

    x_grid = np.linspace(left_bound, right_bound, D + 1)

    solver = DG1DSolver.DG1DSolver(x_grid, N, L)

    # initial condition
    solver.initialize_solution(DG1DSolver.gaussian)

    # time integration
    solver.run(T, cfl)

    # L2 error at final time (IMPORTANT: periodic exact solution assumed inside)
    err = solver.L2_error(T)

    h = (right_bound - left_bound) / D

    errors.append(err)
    hs.append(h)

    print(f"h = {h:.6e}, L2 error = {err:.6e}")

# -------------------------
# COMPUTE CONVERGENCE ORDER
# -------------------------
hs = np.array(hs)
errors = np.array(errors)

orders = np.log(errors[:-1] / errors[1:]) / np.log(2)

print("\n==============================")
print("Observed convergence rates")
print("==============================")

for i, p in enumerate(orders):
    print(f"D={D_values[i]} -> {D_values[i+1]} : p ≈ {p:.3f}")

# -------------------------
# LOG-LOG PLOT
# -------------------------
plt.figure()
plt.loglog(hs, errors, marker='o')
plt.gca().invert_xaxis()
plt.xlabel("h")
plt.ylabel("L2 error")
plt.title(f"DG Convergence Study (N={N})")
plt.grid(True, which="both")
plt.savefig(f"convergence_plot(N={N})(cfl={cfl}).png", dpi=300)
plt.show()