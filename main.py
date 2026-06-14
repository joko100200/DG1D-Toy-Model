import os
import sys
import inspect
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

import DG1DSolver


# -------------------------
# LOGGING UTILITY
# -------------------------

class Tee:
    """
    Used to log print out data to store and review.
    """
    def __init__(self, filepath, mode="w"):
        self.file   = open(filepath, mode)
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

N           = 4
L           = 100
left_bound  = 1.0
right_bound = 50.0
R           = 30.0
P           = 4
T           = 60.0
cfl         = 0.001
D           = 160

# -------------------------
# OUTPUT DIRECTORIES
# -------------------------

os.makedirs("logs",   exist_ok=True)
os.makedirs("graphs", exist_ok=True)
os.makedirs("probes", exist_ok=True)


run_tag   = f"N{N}_T{T}_cfl{cfl}_D{D}_P{P}"
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file  = f"logs/convergence_{run_tag}_{timestamp}.txt"

sys.stdout = Tee(log_file)

# -------------------------
# PRINT RUN SUMMARY
# -------------------------

print("================================")
print("DG CONVERGENCE RUN")
print("================================")
print(f"N = {N}")
print(f"D = {D}")
print(f"T = {T}")
print(f"CFL = {cfl}")
print(f"Domain = [{left_bound}, {right_bound}]")
print(f"R = {R}, P = {P}")

src            = inspect.getsource(DG1DSolver.initial_state)
keep_keywords  = ["f ="]
filtered_lines = [l for l in src.splitlines() if any(k in l for k in keep_keywords)]
print("\n".join(filtered_lines))


# -------------------------
# H-REFINEMENT LOOP
# -------------------------

hs     = []
errors = []
D_values    = [20, 40, 80, 160, 320]

for Dp in D_values:
    probe_path = f"probes/h_refinement{Dp}{run_tag}.csv"
    x_grid     = np.linspace(left_bound, right_bound, Dp + 1)
    h          = (right_bound - left_bound) / Dp

    if os.path.isfile(probe_path):
        print(f"\n{'='*30}")
        print(f"Probe '{probe_path}' found. Loading...")
        print(f"\n{'='*30}")
        solver = DG1DSolver.DG1DSolver(x_grid, N, L, R, P, probe_path)
        coarse_probe = np.loadtxt(probe_path, delimiter=" ", skiprows=1)
        solver._probe_buffer = coarse_probe.tolist()
    else:
        print(f"\n{'='*30}")
        print(f"Running D = {Dp}, N = {N}")
        print(f"{'='*30}")

        solver = DG1DSolver.DG1DSolver(x_grid, N, L, R, P, probe_path)
        solver.initialize_solution(DG1DSolver.initial_state)
        solver.run(T, cfl)

    err = solver.L2_error_probe_state_diff()
    print(f"L2 Error for D={Dp} compared to exact: {err}")

    errors.append(err)


# -------------------------
# H-REFINEMENT RATES
# -------------------------

errors = np.array(errors)   # (num_D, 3)

orders_scri = np.log(errors[:-1, 0] / errors[1:, 0]) / np.log(2)
orders_in   = np.log(errors[:-1, 1] / errors[1:, 1]) / np.log(2)
orders_mid  = np.log(errors[:-1, 2] / errors[1:, 2]) / np.log(2)

print("\n==============================")
print("Observed convergence rates")
print("==============================")
print("Scri")
for i, p in enumerate(orders_scri):
    print(f"D={D_values[i]:4d} -> {D_values[i+1]:4d} : p ≈ {p:.3f}")
print("Second-to-last")
for i, p in enumerate(orders_in):
    print(f"D={D_values[i]:4d} -> {D_values[i+1]:4d} : p ≈ {p:.3f}")
print("Mid")
for i, p in enumerate(orders_mid):
    print(f"D={D_values[i]:4d} -> {D_values[i+1]:4d} : p ≈ {p:.3f}")


# -------------------------
# H-REFINEMENT PLOT
# -------------------------

graph_base = f"graphs/convergence_{run_tag}_{timestamp}"

labels = ["Scri", "second-to-last", "Mid"]
D_values = np.asarray(D_values)
fig, ax = plt.subplots()
for i, label in enumerate(labels):
    ax.plot(np.log10(D_values), np.log10(errors[:, i]), marker='o', label=label)
ref = errors[-1,0] * (D_values[-1]/D_values)**8
ax.plot(np.log10(D_values), np.log10(ref), "--", label=r"$D^{-8}$")
ref = errors[-1,-1] * (D_values[-1]/D_values)**5
ax.plot(np.log10(D_values), np.log10(ref), "--", label=r"$D^{-5}$")
ax.set_xlabel("$log_{10}$(No. of subdomains)")
ax.set_ylabel("$log_{10}$(Relative L2 error)")
ax.set_title(f"h-refinement convergence (N={N})")
ax.legend()
ax.grid(True, which="both")
fig.savefig(f"{graph_base}_h_refinement.png", dpi=300)
plt.show()
plt.close()

# -------------------------
# P-REFINEMENT STUDY
# -------------------------

print("\n==============================")
print("P-refinement study (fixed D)")
print("==============================")

N_values = [2, 3, 4, 5, 6]
x_grid   = np.linspace(left_bound, right_bound, D + 1)

p_errors = []

for Np in N_values:

    probe_path = f"probes/p_refinement_N{Np}{run_tag}.csv"

    if os.path.isfile(probe_path):
        print(f"\n{'='*30}")
        print(f"Probe '{probe_path}' found. Loading...")
        print(f"\n{'='*30}")
        solver = DG1DSolver.DG1DSolver(x_grid, Np, L, R, P, probe_path)
        coarse_probe = np.loadtxt(probe_path, delimiter=" ", skiprows=1)
        solver._probe_buffer = coarse_probe.tolist()
    else:
        print(f"\n{'='*30}")
        print(f"Running D = {D}, N = {Np}")
        print(f"\n{'='*30}")
        solver = DG1DSolver.DG1DSolver(x_grid, Np, L, R, P, probe_path)
        solver.initialize_solution(DG1DSolver.initial_state)
        solver.run(T, cfl)
    
    err = solver.L2_error_probe_state_diff()
    print(f"L2 Error for N={Np} compared to exact: {err}")
    p_errors.append(err)   # use scri error for p-refinement


# -------------------------
# P-REFINEMENT RATES
# -------------------------

print("\n==============================")
print("P-refinement exponential rates")
print("==============================")
p_errors = np.array(p_errors)   # (num_D, 3)

print("Scri")
for i in range(len(N_values) - 1):
    alpha1 = np.log(p_errors[i, 0] / p_errors[i+1, 0]) / (N_values[i+1] - N_values[i])
    print(f"N={N_values[i]} -> {N_values[i+1]} : alpha ≈ {alpha1:.6f}")
print("Second-to-last")
for i in range(len(N_values) - 1):
    alpha2 = np.log(p_errors[i, 1] / p_errors[i+1, 1]) / (N_values[i+1] - N_values[i])
    print(f"N={N_values[i]} -> {N_values[i+1]} : alpha ≈ {alpha2:.6f}")
print("Mid")
for i in range(len(N_values) - 1):
    alpha3 = np.log(p_errors[i, 2] / p_errors[i+1, 2]) / (N_values[i+1] - N_values[i])
    print(f"N={N_values[i]} -> {N_values[i+1]} : alpha ≈ {alpha3:.6f}")



# -------------------------
# P-REFINEMENT PLOT
# -------------------------

graph_base = f"graphs/convergence_{run_tag}_{timestamp}"

labels = ["Scri", "Second-to-last", "Mid"]
fig, ax = plt.subplots()
for i, label in enumerate(labels):
    ax.semilogy(N_values, p_errors[:, i], marker='o', label=label)
ax.set_xlabel("Polynomial degree N")
ax.set_ylabel("Relative L2 error")
ax.set_title(f"p-refinement convergence (D={D})")
ax.legend()
ax.grid(True, which="both")
fig.savefig(f"{graph_base}_p_refinement.png", dpi=300)
plt.show()
plt.close()