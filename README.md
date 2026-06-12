# Discontinuous Galerkin Solver for the Hyperboloidal Wave Equation

A high-order nodal discontinuous Galerkin (DG) solver for the one-dimensional
wave equation with a hyperboloidal layer. The code uses Gauss-Lobatto-Legendre
(GLL) quadrature, Lagrange nodal basis functions, upwind numerical fluxes, and
classical RK4 time integration.

The hyperboloidal layer compactifies future null infinity ($\mathscr{I}^+$)
to a finite coordinate location, allowing outgoing radiation to leave the domain
without artificial outer boundary conditions.

This implementation is based on methods from the literature listed in the References section below.

The solver demonstrates:

- $N+1$ convergence at generic grid points.
- Superconvergent behavior approaching $2N$ at $\mathscr{I}^+$.
- Exponential convergence under $p$-refinement.
- Stable propagation of outgoing waves through a hyperboloidal compactification layer.

---

## Problem

The code evolves the scalar wave equation

$\partial_{tt}U = \partial_{xx}U - V(x)U$

written as a first-order system

$\partial_t U = -p,$

$\partial_t q = -\partial_x p,$

$\partial_t p = -\partial_x q + V(x)U,$

where

$q = \partial_x U,\qquad p = -\partial_t U.$

For convergence studies the default potential is

$V(x)=\frac{6}{x^2},$

corresponding to the $l=2$ centrifugal barrier of the flat-space scalar wave equation.

---

## Hyperboloidal Layer

The computational coordinate $\rho$ is related to the physical radius $r$ through

$r = \frac{\rho}{\Omega(\rho)},$

with compactification function

$\Omega(\rho) = 1 - \left(\frac{\rho-R}{s-R}\right)^P.$

The layer begins at $\rho=R$ and future null infinity is located at

$\rho=s.$

Outgoing radiation reaches $\mathscr{I}^+$ in finite computational time and leaves
the domain without numerical reflection.

---

## Spatial Discretization

The computational domain is partitioned into $D$ elements

$I_e=[x_e,x_{e+1}],$

each mapped to the reference element

$\xi \in [-1,1].$

Within each element the solution is approximated by degree-$N$ Lagrange
polynomials defined on Gauss-Lobatto-Legendre nodes.

For a nodal basis,

$\phi_i(\xi_j)=\delta_{ij},$

so the DG coefficients are simply nodal values.

Because quadrature and interpolation use the same GLL nodes, the mass matrix is diagonal,

$M_{ij}=w_i\delta_{ij},$

making application of $M^{-1}$ an elementwise division by quadrature weights.

---

## Numerical Fluxes

The first-order wave system is evolved using exact upwind fluxes.

At each interface,

$\hat q = \frac12(q_-+q_+) + \frac12(p_- - p_+),$

$\hat p = \frac12(p_-+p_+) + \frac12(q_- - q_+).$

These fluxes provide stable communication between neighboring elements and
enforce outgoing-wave behavior at the outer boundary.

---

## Time Integration

The semi-discrete DG system

$\frac{d\mathbf u}{dt} = \mathcal L(\mathbf u)$

is advanced using classical fourth-order Runge-Kutta (RK4).

Convergence studies are typically performed with a fixed timestep to isolate
spatial discretization error.

---

## Verification

The primary verification problem uses the exact outgoing solution of

$U_{tt} = U_{xx} - \frac{6}{x^2}U.$

The exact solution is

$U(t,r) = f''(u) + \frac{3}{r}f'(u) + \frac{3}{r^2}f(u),$

with

$u = t - r + x_0,$

and

$f(u)=\sin(u)e^{-u^2}.$

At future null infinity,

$U|_{\mathscr I^+} = f''(u),$

since the $1/r$ and $1/r^2$ terms vanish.

---

## Convergence Results

### h-refinement

For fixed polynomial degree $N=4$, the solver exhibits the expected
$N+1$ convergence rate at generic grid points and approximately $2N$
superconvergence at future null infinity.

Example:

| D (elements) | Relative L2 Error at $\mathscr{I}^+$ |
|---|---|
| 20  | 9.33e-01 |
| 40  | 2.98e-01 |
| 80  | 7.64e-03 |
| 160 | 3.86e-05 |
| 320 | 1.59e-07 |

Observed rates approach

$p \approx 8,$

consistent with superconvergent DG behavior for $N=4$.

---

### p-refinement

For fixed mesh resolution, errors decrease exponentially with increasing
polynomial degree.

Example:

| N | Relative L2 Error at $\mathscr{I}^+$ |
|---|---|
| 2 | 2.64e-01 |
| 3 | 4.36e-03 |
| 4 | 3.86e-05 |
| 5 | 2.62e-07 |
| 6 | 1.62e-09 |

This confirms spectral convergence of the DG discretization.

---

## Features

- Nodal DG formulation
- Gauss-Lobatto-Legendre quadrature
- Diagonal mass matrix
- Strong-form DG differentiation
- Exact upwind fluxes
- Hyperboloidal compactification
- Direct extraction of waveforms at $\mathscr{I}^+$
- h-refinement studies
- p-refinement studies
- Exact-solution verification

---

## Future Work

Planned extensions include:

- Pöschl-Teller scattering potentials
- Quasinormal mode extraction
- Black-hole perturbation potentials
- Distributionally forced wave equations
- Teukolsky-equation-inspired test problems
- Investigation of long-range potential behavior under hyperboloidal compactification

---

## Usage

```python
import numpy as np
from HyperboloidalDG import HyperboloidalWaveDG

x_grid = np.linspace(1.0, 50.0, 161)

solver = HyperboloidalWaveDG(
    x_grid=x_grid,
    N=4,
    R=25.0,
    P=4
)

solver.initialize_solution(initial_state)
solver.run(T=60.0)

## References

The implementation of the DG scheme and hyperboloidal layer was heavily influenced by:

Vishal, M., Field, S. E., Rink, K., Gottlieb, S., & Khanna, G. (2024).
Toward exponentially-convergent simulations of extreme-mass-ratio inspirals: A time-domain solver for the scalar Teukolsky equation with singular source terms.
*Physical Review D*, 110(10), 104009.
DOI: 10.1103/PhysRevD.110.104009
URL: http://dx.doi.org/10.1103/PhysRevD.110.104009

Vishal, M., Field, S. E., Gottlieb, S., & Ryan, J. (2025).
Superconvergent discontinuous Galerkin method for the scalar Teukolsky equation on hyperboloidal domains: Efficient waveform and self-force computation.
*General Relativity and Gravitation*, 57(7).
DOI: 10.1007/s10714-025-03435-9
URL: http://dx.doi.org/10.1007/s10714-025-03435-9

Field, S. E., Gottlieb, S., Khanna, G., & McClain, E. (2022).
Discontinuous Galerkin Method for Linear Wave Equations Involving Derivatives of the Dirac Delta Distribution.
In *Spectral and High Order Methods for Partial Differential Equations ICOSAHOM 2020+1* (pp. 307–321).
Springer International Publishing.
DOI: 10.1007/978-3-031-20432-6_19
URL: http://dx.doi.org/10.1007/978-3-031-20432-6_19

Many of the verification tests, error measures, and convergence expectations used in this repository follow the methodology in Vishal et al. (2024).