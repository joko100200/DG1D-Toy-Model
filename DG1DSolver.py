import numpy as np
import numpy.typing as npt
from numpy.polynomial.legendre import Legendre
from collections.abc import Callable

class DG1DSolver:
    """
    Discontinuous Galerkin solver for the 1D wave equation.

    Solves the first-order system:
        d/dt p = d/dx q
        d/dt q = d/dx p

    with unit wave speed, using upwind fluxes and RK4 time integration.

    State is stored as u of shape (D, N+1, 2) where:
        u[:, :, 0] = p  (time derivative of displacement)
        u[:, :, 1] = q  (spatial derivative of displacement)
    """

    def __init__(self, x_grid: npt.NDArray[np.float64], N: int, L: int):
        self.D = len(x_grid) - 1
        self.N = N
        self.L = L

        self.x = x_grid
        self.h = (self.x[1:] - self.x[:-1]) / 2.0

        self.lagrange_basis_matrix()
        self.quad_weights = self.gauss_lobatto_weights()
        self.compute_MassMatrix()
        self.calculate_D_Phi()

        self.x_nodes = self.reconstruct_x_at_nodes()

        # Since we are not using a time dependent potential we pre compute for efficiency
        self.V_at_nodes = effective_potential(self.x_nodes)

    def lagrange_basis_matrix(self):
        """
        Lagrange basis functions on the reference element [-1, 1].

        Stores
        ------
        Phi_q : (N+1, N+1)   Phi_q[j, m] = phi_j(xi_m)
        Phi_plot : (N+1, L)  Phi_plot[j, k] = phi_j(epsilon_k) on dense grid
        """
        self.xi_nodes = self.gauss_lobatto_nodes()
        self.grid_points = np.linspace(-1.0, 1.0, self.L, dtype=np.float64)

        Phi_q    = np.zeros((self.N + 1, self.N + 1))
        Phi_plot = np.zeros((self.N + 1, self.L))

        for j in range(self.N + 1):
            lj_q = np.ones(self.N + 1)
            lj_p = np.ones(self.L)
            xj = self.xi_nodes[j]
            for m, xm in enumerate(self.xi_nodes):
                if m != j:
                    lj_q *= (self.xi_nodes   - xm) / (xj - xm)
                    lj_p *= (self.grid_points - xm) / (xj - xm)
            Phi_q[j, :]    = lj_q
            Phi_plot[j, :] = lj_p

        self.Phi_q    = Phi_q
        self.Phi_plot = Phi_plot

    def reconstruct_x(self) -> npt.NDArray[np.float64]:
        """Physical coordinates on the dense grid. Shape (D, L)."""
        x_grid = np.asarray(self.x, dtype=np.float64)
        epsilon = self.grid_points
        x_reconstructed = np.zeros((self.D, self.L), dtype=np.float64)
        for j in range(self.D):
            x_left  = x_grid[j]
            x_right = x_grid[j + 1]
            dx = x_right - x_left
            x_reconstructed[j, :] = x_left + 0.5 * dx * (epsilon + 1.0)
        return x_reconstructed

    def reconstruct_x_at_nodes(self) -> npt.NDArray[np.float64]:
        """Physical coordinates at GL nodes. Shape (D, N+1)."""
        x_grid = np.asarray(self.x, dtype=np.float64)
        x_reconstructed_at_nodes = np.zeros((self.D, self.N + 1), dtype=np.float64)
        for j in range(self.D):
            x_left  = x_grid[j]
            x_right = x_grid[j + 1]
            dx = x_right - x_left
            x_reconstructed_at_nodes[j, :] = x_left + 0.5 * dx * (self.xi_nodes + 1.0)
        return x_reconstructed_at_nodes

    def initialize_solution(
        self,
        init_fn: Callable[
            [npt.NDArray[np.float64]],
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ]
    ) -> None:
        """
        Project initial conditions onto the DG basis.

        Parameters
        ----------
        init_fn : callable
            Function of x returning (f, fx, g) where
                f = U(x, 0)
                fx = ∂U/∂x(x, 0)
                g = q(x, 0)

        Stores
        ------
        self.u : (D, N+1, 3)
            u[:, :, 0] = coefficients for U
            u[:, :, 1] = coefficients for q = ∂U/∂x
            u[:, :, 2] = coefficients for p = ∂U/∂t 
        """
        f_vals, fx_vals, g_vals = init_fn(self.x_nodes)   # each (D, N+1)

        self.u = np.zeros((self.D, self.N + 1, 3), dtype=np.float64)

        for component, vals in enumerate([f_vals, fx_vals, g_vals]):
            b = np.zeros((self.D, self.N + 1))
            for j in range(self.D):
                for i in range(self.N + 1):
                    b[j, i] = np.sum(self.quad_weights * vals[j, :] * self.Phi_q[i, :])
            self.u[:, :, component] = (self.inv_M @ b.T).T

    def compute_MassMatrix(self) -> None:
        """
        Reference mass matrix M_ij = integral phi_i phi_j dxi and its inverse.
        """
        M = np.zeros((self.N + 1, self.N + 1), dtype=np.float64)
        for i in range(self.N + 1):
            for j in range(self.N + 1):
                M[i, j] = np.sum(self.quad_weights * self.Phi_q[i, :] * self.Phi_q[j, :])
        self.M     = M
        self.inv_M = np.linalg.inv(M)

    def error_in_u0(self, init_fn : Callable[[npt.NDArray[np.float64]], tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]]) -> None:
        """Print projection error at t=0 for both components."""
        f_exact, fx_exact, g_exact = init_fn(self.x_nodes)
        U_h = self.u[:, :, 0]
        q_h = self.u[:, :, 1]
        p_h = self.u[:, :, 2]
        err_p = np.sqrt(np.mean((U_h - f_exact)**2))
        err_q = np.sqrt(np.mean((q_h - fx_exact)**2))
        err_g = np.sqrt(np.mean((p_h - g_exact)**2))
        print(f"Projection error U: {err_p:.6e}")
        print(f"Projection error q: {err_q:.6e}")
        print(f"Projection error p: {err_g:.6e}")

    def gauss_lobatto_nodes(self) -> npt.NDArray[np.float64]:
        if self.N == 1:
            return np.array([-1.0, 1.0], dtype=np.float64)
        P  = Legendre.basis(self.N)
        dP = P.deriv()
        interior = dP.roots()
        return np.concatenate(([-1.0], np.sort(interior), [1.0]), dtype=np.float64)

    def gauss_lobatto_weights(self) -> npt.NDArray[np.float64]:
        N     = self.N
        nodes = self.gauss_lobatto_nodes()
        if N == 1:
            return np.array([1.0, 1.0], dtype=np.float64)
        Pn      = Legendre.basis(N)
        weights = np.zeros(N + 1, dtype=np.float64)
        for i, x in enumerate(nodes):
            weights[i] = 2.0 / (N * (N + 1)) / (Pn(x) ** 2)
        return weights

    def calculate_D_Phi(self):
        """
        Derivative matrix via barycentric weights.
        D_Phi[i, j] = phi_j'(xi_i)
        """
        Np  = self.N + 1
        x   = self.xi_nodes
        lam = np.ones(Np, dtype=np.float64)
        for j in range(Np):
            for m in range(Np):
                if m != j:
                    lam[j] /= (x[j] - x[m])
        D_Phi = np.zeros((Np, Np), dtype=np.float64)
        for i in range(Np):
            for j in range(Np):
                if i != j:
                    D_Phi[i, j] = lam[j] / lam[i] / (x[i] - x[j])
        for i in range(Np):
            D_Phi[i, i] = -np.sum(D_Phi[i])
        self.D_Phi = D_Phi

    def rhs(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        RHS for the wave system  d/dt [U, q, p] = [p, d/dx p, d/dt q - V(x)U].

        Parameters
        ----------
        u : (D, N+1, 3)

        Returns
        -------
        dudt : (D, N+1, 3)
        """
        U = u[:, :, 0]
        q = u[:, :, 1]
        p = u[:, :, 2]

        # volume — positive, same structure as working advection solver
        vol_p = (self.quad_weights * q) @ self.D_Phi
        vol_q = (self.quad_weights * p) @ self.D_Phi

        # Potential source term -V(x)*p in q equation
        source = (self.quad_weights * self.V_at_nodes * U) @ self.inv_M.T

        # interface values
        p_minus_L = np.roll(p[:, -1],  1)
        p_plus_L  = p[:, 0]
        p_minus_R = p[:, -1]
        p_plus_R  = np.roll(p[:, 0], -1)

        q_minus_L = np.roll(q[:, -1],  1)
        q_plus_L  = q[:, 0]
        q_minus_R = q[:, -1]
        q_plus_R  = np.roll(q[:, 0], -1)

        # exact upwind flux for wave system F(p,q) = (-q, -p)
        hat_q_L = 0.5*(q_minus_L + q_plus_L) - 0.5*(p_minus_L - p_plus_L)
        hat_q_R = 0.5*(q_minus_R + q_plus_R) - 0.5*(p_minus_R - p_plus_R)

        hat_p_L = 0.5*(p_minus_L + p_plus_L) - 0.5*(q_minus_L - q_plus_L)
        hat_p_R = 0.5*(p_minus_R + p_plus_R) - 0.5*(q_minus_R - q_plus_R)

        # p equation boundary uses hat_q, q equation boundary uses hat_p
        b_p = np.zeros_like(p)
        b_q = np.zeros_like(q)
        b_p[:, 0]  = -hat_q_L;  b_p[:, -1] = hat_q_R
        b_q[:, 0]  = -hat_p_L;  b_q[:, -1] = hat_p_R

        dudt = np.zeros_like(u)
        dudt[:, :, 0] = p
        dudt[:, :, 1] = ((b_q - vol_q) @ self.inv_M.T) / self.h[:, None]
        dudt[:, :, 2] = ((b_p - vol_p) @ self.inv_M.T) / self.h[:, None] - source

        return dudt

    def compute_dt(self, cfl: float) -> float:
        """CFL timestep for unit wave speed."""
        return float(cfl * np.min(self.h) / (2 * self.N + 1))

    def step_rk4(self, dt: float) -> None:
        u0 = self.u.copy()
        k1 = self.rhs(u0)
        k2 = self.rhs(u0 + 0.5 * dt * k1)
        k3 = self.rhs(u0 + 0.5 * dt * k2)
        k4 = self.rhs(u0 + dt * k3)
        self.u = u0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def run(self, T: float, cfl: float = 0.5) -> npt.NDArray[np.float64]:
        t  = 0.0
        dt = self.compute_dt(cfl)
        print(f"t = 0.0, L2 error = {self.L2_error(0.0)}, Energy = {self.compute_energy()}, L2 norm = {self.compute_L2_norm()}")
        while t < T:
            if t + dt > T:
                dt = T - t
            self.step_rk4(dt)
            t += dt
        print(f"t = {T}, L2 error = {self.L2_error(T)}, Energy = {self.compute_energy()}, L2 norm = {self.compute_L2_norm()}")
        return self.u
    
    def L2_error(self, t: float) -> float:
        err        = 0.0
        domain_len = self.x[-1] - self.x[0]

        for e in range(self.D):
            x_e     = self.x_nodes[e]
            x_plus  = ((x_e + t - self.x[0]) % domain_len) + self.x[0]
            x_minus = ((x_e - t - self.x[0]) % domain_len) + self.x[0]
            _, fx_plus, g_plus = initial_state(x_plus)
            _, fx_minus, g_minus = initial_state(x_minus)
            p_exact    = 0.5 * (g_plus + g_minus + fx_plus - fx_minus)
            p_h        = self.u[e, :, 2]
            err += np.sum(self.quad_weights * (p_h - p_exact)**2) * self.h[e]

        return np.sqrt(err)

    def compute_energy(self) -> float:
        """
        Discrete energy E = 0.5 * integral (p^2 + q^2 + V * U^2) dx.
        Should be conserved by the wave system.
        """
        U = self.u[:, :, 0]
        q = self.u[:, :, 1]
        p = self.u[:, :, 2]

        energy = 0.0
        for e in range(self.D):
            energy += np.sum(self.quad_weights * (p[e]**2 + q[e]**2 + self.V_at_nodes[e] * U[e]**2)) * self.h[e]
        return 0.5 * energy

    def compute_L2_norm(self) -> float:
        """L2 norm of U."""
        U = self.u[:, :, 0]
        norm_sq = 0.0
        for e in range(self.D):
            norm_sq += np.sum(self.quad_weights * U[e]**2) * self.h[e]
        return np.sqrt(norm_sq)

    def plot_solution(self, t: float, filename: str = "graphs/DG_wave_solution.png") -> None:
        import matplotlib.pyplot as plt

        x_dense = self.reconstruct_x().reshape(-1)
        U_dense = (self.u[:, :, 0] @ self.Phi_plot).reshape(-1)
        q_dense = (self.u[:, :, 1] @ self.Phi_plot).reshape(-1)
        p_dense = (self.u[:, :, 2] @ self.Phi_plot).reshape(-1)

        domain_len = self.x[-1] - self.x[0]
        x_plus  = ((x_dense + t - self.x[0]) % domain_len) + self.x[0]
        x_minus = ((x_dense - t - self.x[0]) % domain_len) + self.x[0]
        _, fx_plus, g_plus = initial_state(x_plus)
        _, fx_minus, g_minus = initial_state(x_minus)

        p_exact = 0.5 * (g_plus + g_minus + fx_plus - fx_minus)
        q_exact = 0.5 * (g_plus - g_minus + fx_plus + fx_minus)

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax1.plot(x_dense, p_dense, label='DG p')
        ax1.plot(x_dense, p_exact, label='Exact p', linestyle='dashed')
        ax1.set_ylabel('p');  ax1.legend();  ax1.grid()
        ax2.plot(x_dense, q_dense, label='DG q')
        ax2.plot(x_dense, q_exact, label='Exact q', linestyle='dashed')
        ax2.set_ylabel('q');  ax2.legend();  ax2.grid()
        ax3.plot(x_dense, U_dense, label='DG U')
        ax3.set_xlabel('x');  ax3.set_ylabel('U');  ax3.legend();  ax3.grid()
        plt.suptitle(f'Wave equation DG solution at t={t:.3f}')
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.show()


# ------------------------------------------------------------------
# Initial conditions
# ------------------------------------------------------------------

def initial_state(x: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Gaussian initial state.

    Returns
    -------
    f : U(x, 0)
    fx : partial_x U(x, 0)
    g : partial_t U(x, 0)
    """
    f = (np.sin(6*x) * np.exp(-0.5 * x**2)).astype(np.float64) # initial value U(x, 0)
    fx = (-np.exp(-0.5 * x**2)*(x*np.sin(6*x) - 6*np.cos(6*x))).astype(np.float64) # partial_x U(x, 0)
    g = (np.exp(-0.5 * x**2)*(x*np.sin(6*x) - 6*np.cos(6*x))).astype(np.float64) # partial_t U(x, 0)
    return f, fx, g

def effective_potential(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Effective potential V(x) for the wave equation.
    Appears as a source term -V(x)*p in the q equation.
    
    Parameters
    ----------
    x : (D, N+1) array of physical node positions
    
    Returns
    -------
    V : (D, N+1) array of potential values at each node
    """
    return np.zeros_like(x)