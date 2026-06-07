# Discontinuous Galerkin Solver for the 1D Wave Equation

A high-order nodal discontinuous Galerkin (DG) solver for the 1D wave equation,
using Gauss-Lobatto-Legendre (GLL) quadrature, upwind fluxes, and RK4 time integration.
Achieves order $N+1$ convergence in $h$-refinement and exponential convergence in $p$-refinement.

---

## Convergence Results

| D (elements) | L2 Error | Convergence Rate |
|---|---|---|
| 20  | 3.53e+00 | —     |
| 40  | 3.18e-02 | 6.80  |
| 80  | 3.02e-04 | 6.72  |
| 160 | 2.33e-06 | 7.02  |
| 320 | 1.83e-08 | 6.99  |
| 640 | 1.53e-10 | 6.90  |

*N=6, T=5.0, domain [-10, 20], CFL=0.1. Expected rate: N+1 = 7.*

| N (polynomial order) | L2 Error |
|---|---|
| 2 | 1.05e+00 |
| 3 | 2.08e-02 |
| 4 | 1.00e-03 |
| 5 | 5.09e-05 |
| 6 | 2.33e-06 |

*D=160 fixed. Exponential decay confirms spectral convergence in p.*

---

## Method

### 1. Problem and State Vector

We solve the second-order wave equation

$$\partial_{tt} U = \partial_{xx} U - V(x)U$$

rewritten as a first-order system with $q = \partial_x U$ and $p = \partial_t U$:

$$\partial_t \begin{pmatrix} U \\ q \\ p \end{pmatrix} = \begin{pmatrix} p \\ \partial_x p \\ \partial_x q - V(x)U \end{pmatrix}$$

The domain $[x_L, x_R]$ is partitioned into $D$ elements $I_e = [x_e, x_{e+1}]$
with half-width $h_e = (x_{e+1} - x_e)/2$.

---

### 2. Reference Element and Basis

Each element is mapped to the reference element $[-1, 1]$ via

$$x = x_e + h_e(\xi + 1), \qquad \xi \in [-1, 1]$$

On the reference element we place $N+1$ GLL nodes $\lbrace \xi_j \rbrace_{j=0}^{N}$
with quadrature weights $\lbrace w_j \rbrace$. The solution on element $e$ is
approximated by the degree-$N$ nodal interpolant

$$U_h\big|_{I_e}(\xi, t) = \sum_{j=0}^{N} U_j^e(t)\, \phi_j(\xi)$$

where $\phi_j$ are Lagrange basis functions satisfying $\phi_j(\xi_m) = \delta_{jm}$,
so the degrees of freedom are simply nodal values $U_j^e = U_h(\xi_j, t)$.
Identical expansions hold for $q_h$ and $p_h$.

---

### 3. Weak Formulation

Multiplying the $p$-equation by test function $\phi_i \in \mathcal{P}^N$ and
integrating over the reference element, then integrating by parts:

$$h_e \int_{-1}^{1} \partial_t p_h\, \phi_i\, d\xi = -\int_{-1}^{1} q_h\, \phi_i'\, d\xi + \Big[\hat{q}\, \phi_i\Big]_{-1}^{1}$$

where $\hat{q}$ is the numerical flux at element boundaries.
The $q$-equation is identical with $p \leftrightarrow q$.

---

### 4. Matrix Form

The mass matrix under GLL quadrature is diagonal:

$$M_{ij} = \int_{-1}^{1} \phi_i\, \phi_j\, d\xi \approx \sum_m w_m\, \phi_i(\xi_m)\, \phi_j(\xi_m) = w_i\, \delta_{ij}$$

The volume term is evaluated as:

$$\big[S^T \mathbf{q}\big]_i \approx \big[(\mathbf{w} \odot \mathbf{q})^T D_\Phi\big]_i$$

where $(D_\Phi)_{im} = \phi_i'(\xi_m)$ is the differentiation matrix and
$\odot$ denotes elementwise multiplication. The boundary vector is

$$(\mathbf{b}_p)_i = \begin{cases} -\hat{q}_L & i = 0 \\ +\hat{q}_R & i = N \\ 0 & \text{otherwise} \end{cases}$$

The semi-discrete system on element $e$ is:

$$h_e M\, \dot{\mathbf{p}}^e = \mathbf{b}_p^e - (\mathbf{w} \odot \mathbf{q}^e)^T D_\Phi - h_e M(\mathbf{V}^e \odot \mathbf{U}^e)$$

$$h_e M\, \dot{\mathbf{q}}^e = \mathbf{b}_q^e - (\mathbf{w} \odot \mathbf{p}^e)^T D_\Phi$$

$$\dot{\mathbf{U}}^e = \mathbf{p}^e$$

Since $M$ is diagonal, applying $M^{-1}$ is trivial:

$$\dot{\mathbf{p}}^e = \frac{1}{h_e} M^{-1} \Big[\mathbf{b}_p^e - (\mathbf{w} \odot \mathbf{q}^e)^T D_\Phi\Big] - \mathbf{V}^e \odot \mathbf{U}^e$$

$$\dot{\mathbf{q}}^e = \frac{1}{h_e} M^{-1} \Big[\mathbf{b}_q^e - (\mathbf{w} \odot \mathbf{p}^e)^T D_\Phi\Big]$$

---

### 5. Upwind Fluxes

For the wave system, the exact upwind fluxes at left and right interfaces are:

$$\hat{q}_L = \tfrac{1}{2}(q_- + q_+) - \tfrac{1}{2}(p_- - p_+)$$

$$\hat{q}_R = \tfrac{1}{2}(q_- + q_+) - \tfrac{1}{2}(p_- - p_+)$$

$$\hat{p}_L = \tfrac{1}{2}(p_- + p_+) - \tfrac{1}{2}(q_- - q_+)$$

$$\hat{p}_R = \tfrac{1}{2}(p_- + p_+) - \tfrac{1}{2}(q_- - q_+)$$

where $\pm$ denotes the left/right element trace at each interface.

---

### 6. Time Integration

The semi-discrete system $\dot{\mathbf{u}} = \mathcal{L}(\mathbf{u})$ is advanced
with classical RK4. The timestep is set by the CFL condition:

$$\Delta t = \frac{c_{\mathrm{CFL}} \cdot \min_e h_e}{2N + 1}$$

where $2N+1$ accounts for GLL node clustering near element boundaries.
All convergence runs used $c_{\mathrm{CFL}} = 0.1$.

---

## Usage

```python
import numpy as np
from dg1d import DG1DSolver, initial_state, effective_potential

x_grid = np.linspace(-10.0, 20.0, 81)   # 80 elements
solver  = DG1DSolver(x_grid, N=6, L=50)
solver.initialize_solution(initial_state)
solver.run(T=5.0, cfl=0.1)
solver.plot_solution(t=5.0)
```

### Convergence Tests

The convergence results reported above can be reproduced by running:

```bash
python main.py
```

This script performs both an $h$-refinement study (fixed $N$, increasing $D$) and a
$p$-refinement study (fixed $D$, increasing $N$), printing L2 errors and observed
convergence rates for each run.