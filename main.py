import numpy as np
import DG1DSolver
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS
# -------------------------
N = 6
L = 100
left_bound = -10.0
right_bound = 20.0

T = 5.0        # keep T small relative to domain so periodic wrapping
cfl = 0.1      # doesn't corrupt the exact solution at fine grids

D_values = [20, 40, 80, 160, 320, 640]

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

    solver.initialize_solution(DG1DSolver.initial_state)

    solver.run(T, cfl)

    err = solver.L2_error(T)
    h   = (right_bound - left_bound) / D

    errors.append(err)
    hs.append(h)

    print(f"h = {h:.6e}")

# -------------------------
# COMPUTE CONVERGENCE ORDER
# -------------------------
hs     = np.array(hs)
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
plt.title(f"DG Wave Equation Convergence (N={N})")
plt.grid(True, which="both")
plt.savefig(f"graphs/waveconvergence_plot_V(N={N})(cfl={cfl}).png", dpi=300)
plt.show()

#---------------------------
# Final Graph Plotting
#---------------------------
solver.plot_solution(T, f"graphs/waveconvergence_solution_V(N={N})(cfl={cfl}).png")