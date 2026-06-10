import os
import sys
from datetime import datetime
import inspect
import numpy as np
import DG1DSolver
import matplotlib.pyplot as plt

class Tee:
    def __init__(self, filepath, mode="w"):
        self.file = open(filepath, mode)
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# -------------------------
# PARAMETERS
# -------------------------
N = 4
L = 100
left_bound = -7.0
right_bound = 30.0
R = 15.0
P = 4

T = 50.0
cfl = 0.1
D = 640

D_values = [20, 40, 80, 160, 320]
errors = []
hs = []

fine_filepath = "probes/h_refinement_probe_file_fine.csv"
run_tag = f"N{N}_T{T}_cfl{cfl}_Dmax{max(D_values)}_Finepath{fine_filepath[-7:-4]}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

complete_filename = f"convergence_{run_tag}_{timestamp}"
log_file = f"logs/{complete_filename}.txt"

os.makedirs("logs", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
os.makedirs("probelogs", exist_ok=True)
sys.stdout = Tee(log_file)

print("================================")
print("DG CONVERGENCE RUN")
print("================================")
print(f"N = {N}")
print(f"D values = {D_values}")
print(f"T = {T}")
print(f"CFL = {cfl}")
print(f"Domain = [{left_bound}, {right_bound}]")

src = inspect.getsource(DG1DSolver.initial_state)

keep_keywords = ["f =", "fx =", "g ="]

filtered_lines = [
    line for line in src.splitlines()
    if any(k in line for k in keep_keywords)
]

print("\n".join(filtered_lines))

if not os.path.isfile(fine_filepath):
    print("fine_filepath not found. Running fine h_refinement for convergence testing...")
    print(f"D={D}, N={N}, T={T}, cfl={cfl}, R={R}, P={P}, left_bound={left_bound}, right_bound={right_bound}")
    print(f"Fine filepath is '{fine_filepath}'")

    x_ref = np.linspace(left_bound, right_bound, D + 1)
    s_ref = DG1DSolver.DG1DSolver(x_ref, N, L, R, P, fine_filepath)
    s_ref.initialize_solution(DG1DSolver.initial_state)
    s_ref.runDEBUG(T, cfl)

    fine_probe = np.array(s_ref._probe_buffer)
    print("Completed propagation")

else:
    print(f"Loading {fine_filepath}")
    fine_probe = np.loadtxt(fine_filepath, delimiter=" ", skiprows=1)  # (M, 4) array of (t, U, q, p)

print("================================\n")

# -------------------------
# CONVERGENCE LOOP
# -------------------------
for Dp in D_values:

    if os.path.isfile(f"probes/h_refinement{Dp}.csv"):
        coarse_probe = np.loadtxt(f"probes/h_refinement{Dp}.csv", delimiter=" ", skiprows=1)
        err = DG1DSolver.DG1DSolver.L2_error_probe_state_diff(coarse_probe, fine_probe)
        h = np.log10(Dp)
        print(f"L2 Error for D={Dp} compared to D={D}: {err}")
        errors.append(err)
        hs.append(h)
        continue

    print("\n==============================")
    print(f"Running D = {Dp}, N = {N}")
    print("==============================")

    x_grid = np.linspace(left_bound, right_bound, Dp + 1)

    solver = DG1DSolver.DG1DSolver(x_grid, N, L, R, P, f"probes/h_refinement{Dp}.csv")

    solver.initialize_solution(DG1DSolver.initial_state)

    solver.runDEBUG(T, cfl)

    err = solver.L2_error_probe_state(fine_probe)
    h   = (right_bound - left_bound) / Dp

    print(f"L2 Error for D={Dp} compared to D={D}: {err}")

    errors.append(err)
    hs.append(h)

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
plt.savefig(f"graphs/{complete_filename}_Log_Log.png", dpi=300)
plt.show()

exit()

#---------------------------
# Final Graph Plotting
#---------------------------
solver.plot_solution(T, f"graphs/{complete_filename}_solution.png")

# -------------------------
# P-REFINEMENT STUDY
# (hold D fixed, vary N)
# -------------------------

print("\n==============================")
print("P-refinement study (fixed D)")
print("==============================")

D_fixed = 160  # choose reasonably fine mesh
x_grid = np.linspace(left_bound, right_bound, D_fixed + 1)

N_values = [2, 3, 4, 5, 6]

p_errors = []

for Np in N_values:

    print("\n------------------------------")
    print(f"Running D = {D_fixed}, N = {Np}")
    print("------------------------------")

    solver = DG1DSolver.DG1DSolver(x_grid, Np, L, R, P, f"p_refinement{Np}.csv")
    solver.initialize_solution(DG1DSolver.initial_state)
    solver.runDEBUG(T, cfl)

    err = solver.L2_error_self(solver.u_fine)
    p_errors.append(err)

print("\n==============================")
print("P-refinement exponential rates")
print("==============================")

alpha_values = []

for i in range(len(N_values) - 1):
    alpha = np.log(p_errors[i] / p_errors[i+1]) / (N_values[i+1] - N_values[i])
    alpha_values.append(alpha)
    print(f"N={N_values[i]} -> {N_values[i+1]} : alpha ≈ {alpha:.6f}")

# -------------------------
# PLOT P-CONVERGENCE
# -------------------------

plt.figure()
plt.semilogy(N_values, p_errors, marker='o')
plt.xlabel("Polynomial degree N")
plt.ylabel("L2 error")
plt.title(f"DG p-convergence (fixed D={D_fixed})")
plt.grid(True, which="both")
plt.savefig(f"graphs/{complete_filename}_p_convergence.png", dpi=300)
plt.show()