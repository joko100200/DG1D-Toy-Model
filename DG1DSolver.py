import numpy as np
import numpy.typing as npt
from numpy.polynomial.legendre import Legendre
from collections.abc import Callable


class DG1DSolver:
    """
    Nodal discontinuous Galerkin solver for the first-order scalar wave equation
    with a hyperboloidal layer.

    The evolved variables are

        U = scalar field = u[:, :, 0]
        q = ∂U/∂r        = u[:, :, 1]
        p = -∂U/∂t       = u[:, :, 2]

    The domain consists of a standard physical region and a hyperboloidal layer
    which compactifies future null infinity (scri+) to the finite coordinate
    location ρ = s.

    The discretization uses:
        - Gauss-Lobatto-Legendre nodes
        - Lagrange interpolating polynomials
        - Strong-form DG differentiation
        - Upwind numerical fluxes
        - Explicit RK4 time integration

    Because the basis is nodal at the quadrature points, the mass matrix is
    diagonal and all mass matrix inversions reduce to elementwise division by
    the quadrature weights.
    """

    def __init__(
        self,
        x_grid: npt.NDArray[np.float64],
        N: int,
        L: int,
        R: float,
        P: int,
        outputfileDir: str,
    ):
        self.D = len(x_grid) - 1
        self.N = N
        self.L = L

        self.x = x_grid
        self.h = (self.x[1:] - self.x[:-1]) / 2.0   # half-element widths, shape (D,)

        self.left_neighbors  = np.roll(np.arange(self.D),  1)
        self.right_neighbors = np.roll(np.arange(self.D), -1)

        self.b_p = np.zeros((self.D, self.N + 1))
        self.b_q = np.zeros((self.D, self.N + 1))

        self.R = self.x[np.argmin(np.abs(self.x - R))]   # snap R to nearest grid interface
        self.P = P
        self.s = self.x[-1]                               # future null infinity ρ = s

        self.xi_nodes     = self.gauss_lobatto_nodes()
        self.quad_weights = self.gauss_lobatto_weights()  # shape (N+1,)
        self.inv_w        = 1.0 / self.quad_weights       # shape (N+1,)

        self.lagrange_basis_matrix()
        self.calculate_D_Phi()

        self.init_probe_logger(outputfileDir)

        self.x_nodes    = self.reconstruct_x_at_nodes()    # (D, N+1)

        self.V_at_nodes = effective_potential(self.x_nodes) # (D, N+1)
        self.H_at_nodes = self.compute_H()                  # (D, N+1)
        self.coeff_at_nodes = 1.0 / (1.0 + self.H_at_nodes)   # (D, N+1), = 1/2 at ρ=s
        
        # J*V precomputed analytically to avoid 0/0 at ρ=s where H→1
        self.JV_at_nodes = self.coeff_at_nodes * self.V_at_nodes * self.hyperboloidal_denom

        

    # ------------------------------------------------------------------
    # Basis and quadrature
    # ------------------------------------------------------------------

    def gauss_lobatto_nodes(self) -> npt.NDArray[np.float64]:
        if self.N == 1:
            return np.array([-1.0, 1.0], dtype=np.float64)
        P       = Legendre.basis(self.N)
        interior = P.deriv().roots()
        return np.concatenate(([-1.0], np.sort(interior), [1.0]), dtype=np.float64)

    def gauss_lobatto_weights(self) -> npt.NDArray[np.float64]:
        if self.N == 1:
            return np.array([1.0, 1.0], dtype=np.float64)
        nodes   = self.gauss_lobatto_nodes()
        Pn      = Legendre.basis(self.N)
        weights = np.array(
            [2.0 / (self.N * (self.N + 1)) / (Pn(x) ** 2) for x in nodes],
            dtype=np.float64,
        )
        return weights

    def lagrange_basis_matrix(self) -> None:
        """
        Build Phi_plot for dense plotting.  Phi_q = I by the Kronecker delta
        property of Lagrange basis at GLL nodes so it is never stored.

        Stores
        ------
        Phi_plot  : (N+1, L)
        grid_points : (L,)   reference interval dense grid
        """
        self.grid_points = np.linspace(-1.0, 1.0, self.L, dtype=np.float64)
        Phi_plot = np.zeros((self.N + 1, self.L))
        for j in range(self.N + 1):
            lj = np.ones(self.L)
            xj = self.xi_nodes[j]
            for m, xm in enumerate(self.xi_nodes):
                if m != j:
                    lj *= (self.grid_points - xm) / (xj - xm)
            Phi_plot[j] = lj
        self.Phi_plot = Phi_plot

    def calculate_D_Phi(self) -> None:
        """
        Differentiation matrix via barycentric weights.
        D_Phi[i, j] = phi_j'(xi_i),  shape (N+1, N+1).
        Endpoint diagonals are set to their exact analytic values to avoid
        floating-point accumulation error.
        """
        Np  = self.N + 1
        x   = self.xi_nodes
        lam = np.ones(Np, dtype=np.float64)
        for j in range(Np):
            for m in range(Np):
                if m != j:
                    lam[j] /= (x[j] - x[m])

        D = np.zeros((Np, Np), dtype=np.float64)
        for i in range(Np):
            for j in range(Np):
                if i != j:
                    D[i, j] = lam[j] / lam[i] / (x[i] - x[j])
        for i in range(Np):
            D[i, i] = -np.sum(D[i])

        # exact endpoint values (standard LGL identity)
        D[0,  0 ] = -self.N * (self.N + 1) / 4.0
        D[-1, -1] =  self.N * (self.N + 1) / 4.0

        self.D_Phi = D

    def compute_H(self) -> npt.NDArray[np.float64]:
        rho   = self.x_nodes
        H     = np.zeros_like(rho)
        self.hyperboloidal_denom = np.ones_like(rho)
        self.Omega = np.ones_like(rho)
        layer = rho > self.R

        sigma        = np.clip((rho[layer] - self.R) / (self.s - self.R), 0.0, 1.0)
        Omega        = 1.0 - sigma**self.P
        Omega_prime  = -self.P * sigma**(self.P - 1) / (self.s - self.R)
        denom        = Omega - rho[layer] * Omega_prime

        # Superstius forcing analytical limits near rho = s. Unneeded but no harm.
        near_scri        = sigma > (1.0 - 1e-12)
        denom[near_scri] = self.s * self.P / (self.s - self.R)
        Omega[near_scri] = 0.0

        self.hyperboloidal_denom[layer] = denom
        self.Omega[layer] = Omega
        H[layer] = 1.0 - Omega**2 / denom
        return H

    # ------------------------------------------------------------------
    # Grid reconstruction
    # ------------------------------------------------------------------

    def reconstruct_x(self) -> npt.NDArray[np.float64]:
        """Dense plotting grid coordinates. Shape (D, L)."""
        x_nodes = np.zeros((self.D, self.L), dtype=np.float64)
        for j in range(self.D):
            dx = self.x[j + 1] - self.x[j]
            x_nodes[j] = self.x[j] + 0.5 * dx * (self.grid_points + 1.0)
        return x_nodes

    def reconstruct_x_at_nodes(self) -> npt.NDArray[np.float64]:
        """GLL node coordinates on the physical grid. Shape (D, N+1)."""
        x_nodes = np.zeros((self.D, self.N + 1), dtype=np.float64)
        for j in range(self.D):
            dx = self.x[j + 1] - self.x[j]
            x_nodes[j] = self.x[j] + 0.5 * dx * (self.xi_nodes + 1.0)
        return x_nodes

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def initialize_solution(
        self,
        init_fn: Callable[
            [npt.NDArray[np.float64]],
            tuple[
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
            ],
        ],
    ) -> None:
        """
        Initialize the DG solution from nodal data.

        Because the basis is nodal at the GLL points, the DG coefficients are
        identical to the nodal values and no projection solve is required.

        Parameters
        ----------
        init_fn : x -> (f, fx, g)
            f  = U(x, 0)
            fx = dU/dx(x, 0)
            g  = p(x, 0) = -dU/dt(x, 0)
        """
        f, fx, g = init_fn(self.x_nodes)
        self.u = np.zeros((self.D, self.N + 1, 3), dtype=np.float64)
        self.u[:, :, 0] = f
        self.u[:, :, 1] = fx
        self.u[:, :, 2] = g

    # ------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------

    def rhs(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Compute du/dt for the hyperboloidal wave system.

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

        # Volume Differentiation
        vol_p = np.einsum('i,ei,ij->ej', self.quad_weights, p, self.D_Phi)
        vol_q = np.einsum('i,ei,ij->ej', self.quad_weights, q, self.D_Phi)

        # Interface Values
        p_minus_L = p[self.left_neighbors,  -1]
        q_minus_L = q[self.left_neighbors,  -1]
        p_plus_R  = p[self.right_neighbors,  0]
        q_plus_R  = q[self.right_neighbors,  0]
        p_plus_L  = p[:, 0]
        p_minus_R = p[:, -1]
        q_plus_L  = q[:, 0]
        q_minus_R = q[:, -1]

        # global right boundary: outflow (no incoming state)
        p_plus_R[-1] = p_minus_R[-1]
        q_plus_R[-1] = q_minus_R[-1]

        # global left boundary: reflecting
        p_minus_L[0] = -p_plus_L[0]
        q_minus_L[0] = -q_plus_L[0]

        # Upwind Fluxes
        hat_q_L = 0.5 * (q_minus_L + q_plus_L) + 0.5 * (p_minus_L - p_plus_L)
        hat_q_R = 0.5 * (q_minus_R + q_plus_R) + 0.5 * (p_minus_R - p_plus_R)
        hat_p_L = 0.5 * (p_minus_L + p_plus_L) + 0.5 * (q_minus_L - q_plus_L)
        hat_p_R = 0.5 * (p_minus_R + p_plus_R) + 0.5 * (q_minus_R - q_plus_R)

        # Boundary Flux Vectors
        self.b_p[:, 0]  = -hat_p_L;  self.b_p[:, -1] = hat_p_R
        self.b_q[:, 0]  = -hat_q_L;  self.b_q[:, -1] = hat_q_R

        # DG Differentiation
        scale = self.inv_w / self.h[:, None]
        dp = (self.b_p - vol_p) * scale
        dq = (self.b_q - vol_q) * scale

        # Hyperboloidal coupling
        H     = self.H_at_nodes
        coeff = self.coeff_at_nodes       # 1/(1+H), finite at ρ=s
        JVU   = self.JV_at_nodes * U      # J*V precomputed to avoid 0/0

        dudt = np.zeros_like(u)
        dudt[:, :, 0] = -p
        dudt[:, :, 1] = coeff * (-dp - H * dq) + H * JVU
        dudt[:, :, 2] = coeff * (-dq - H * dp) + JVU

        return dudt

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

    def compute_dt(self, cfl: float) -> float:
        """
        Return the timestep used by RK4.

        Currently fixed to 1e-3 for convergence studies. The CFL-based
        estimate is retained below for future use.
        """
        #return float(cfl * np.min(self.h) / (2 * self.N + 1))
        return 0.001

    def step_rk4(self, dt: float) -> None:
        u0 = self.u.copy()
        k1 = self.rhs(u0)
        k2 = self.rhs(u0 + 0.5 * dt * k1)
        k3 = self.rhs(u0 + 0.5 * dt * k2)
        k4 = self.rhs(u0 + dt * k3)
        self.u = u0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self, T: float, cfl: float = 0.5) -> npt.NDArray[np.float64]:
        t  = 0.0
        dt = self.compute_dt(cfl)
        print(f"t = {t:.3f} | L2Norm = {self.compute_L2_norm():.6e} | E = {self.compute_energy():.6e}")

        # Time solution plots if needed
        plot_times = []
        plot_idx = 0

        while t < T:
            dt = min(dt, T - t)
            self.step_rk4(dt)
            t += dt

            while plot_idx < len(plot_times) and t >= plot_times[plot_idx]:
                pt = plot_times[plot_idx]
                self.plot_solution(t, f"graphs/{self.D}_WaveEquation_t{pt:.0f}.png")
                plot_idx += 1


            self.log_probe(t)

        self.flush_probe()
        print(f"t = {t:.3f} | L2Norm = {self.compute_L2_norm():.6e} | E = {self.compute_energy():.6e}")
        return self.u

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def compute_energy(self) -> float:
        """E = 0.5 * integral (p² + q² + V·U²) dρ"""
        U = self.u[:, :, 0]
        q = self.u[:, :, 1]
        p = self.u[:, :, 2]
        integrand = p**2 + q**2 + self.V_at_nodes * U**2
        return float(0.5 * np.sum(self.quad_weights * integrand * self.h[:, None]))

    def energy_in_region(self, x_min: float, x_max: float) -> float:
        """Energy integral restricted to [x_min, x_max]."""
        U    = self.u[:, :, 0]
        q    = self.u[:, :, 1]
        p    = self.u[:, :, 2]
        mask = (self.x_nodes >= x_min) & (self.x_nodes <= x_max)
        integrand = (p**2 + q**2 + self.V_at_nodes * U**2) * mask
        return float(0.5 * np.sum(self.quad_weights * integrand * self.h[:, None]))

    def compute_L2_norm(self) -> float:
        """L2 norm of U."""
        U = self.u[:, :, 0]
        return float(np.sqrt(np.sum(self.quad_weights * U**2 * self.h[:, None])))

    def waveform_at_scri(self) -> tuple[float, float, float]:
        """U, q, p at the last node of the last element (ρ = s)."""
        return (self.u[-1, -1, 0], self.u[-1, -1, 1], self.u[-1, -1, 2])

    def L2_error_probe_state_diff(self) -> tuple[float, float, float]:
        """
        Relative L2 errors at three probe locations compared to the exact solution.

        Returns
        -------
        (E_scri, E_in, E_mid) : relative L2 norms at ρ=s, one node inward, domain midpoint.
        """
        data = np.asarray(self._probe_buffer)
        t    = data[:, 0]

        U_scri = data[:, 1]
        U_in   = data[:, 4]
        U_mid  = data[:, 5]

        x_scri = self.x_nodes[-1, -1]
        x_in   = self.x_nodes[-1, -2]
        x_mid  = self.x_nodes[self.D // 2, 0]

        U_exact_scri = exact_solution_at_rho(x_scri, t, 10.0, self.Omega[-1, -1])
        U_exact_in   = exact_solution_at_rho(x_in, t, 10.0, self.Omega[-1, -2])
        U_exact_mid  = exact_solution_at_rho(x_mid, t, 10.0, self.Omega[self.D // 2, 0])

        def L2_time_error(u_num, u_ex):
            num = np.trapz((u_num - u_ex)**2, t)
            den = np.trapz(u_ex**2, t)

            # avoid division blow-up in late-time tails
            if den < 1e-14:
                return np.nan

            return np.sqrt(num / den)

        E_scri = L2_time_error(U_scri, U_exact_scri)
        E_in   = L2_time_error(U_in,   U_exact_in)
        E_mid  = L2_time_error(U_mid,  U_exact_mid)

        return E_scri, E_in, E_mid

    def error_in_u0(self, init_fn: Callable) -> None:
        """Print projection error at t=0."""
        f, fx, g = init_fn(self.x_nodes)
        print(f"Projection error U: {np.sqrt(np.mean((self.u[:,:,0] - f )**2)):.6e}")
        print(f"Projection error q: {np.sqrt(np.mean((self.u[:,:,1] - fx)**2)):.6e}")
        print(f"Projection error p: {np.sqrt(np.mean((self.u[:,:,2] - g )**2)):.6e}")

    def plot_solution(self, t: float, filename: str = "graphs/DG_wave_solution.png") -> None:
        import matplotlib.pyplot as plt

        x_dense = self.reconstruct_x().reshape(-1)
        U_dense = (self.u[:, :, 0] @ self.Phi_plot).reshape(-1)
        U_exact_dense = exact_solution_at_rho(x_dense, np.asarray(t), 10.0, self.Omega)

        q_dense = (self.u[:, :, 1] @ self.Phi_plot).reshape(-1)
        p_dense = (self.u[:, :, 2] @ self.Phi_plot).reshape(-1)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(x_dense, U_dense, label="DG U")
        axes[0].plot(x_dense, U_exact_dense, label="Exact U", linestyle="dashed")
        axes[0].set_ylabel("U")
        axes[0].legend()
        axes[0].grid()

        for ax, y, label in zip(axes[1:], [q_dense, p_dense], ["q", "p"]):
            ax.plot(x_dense, y, label=f"DG {label}")
            ax.set_ylabel(label)
            ax.legend()
            ax.grid()
        axes[-1].set_xlabel("x")
        plt.suptitle(f"Wave equation DG solution at t={t:.3f}")
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        #plt.show()
        plt.close()
    
    def plot_scri_waveform(self, x0: float = 10.0, filename : str = "Scri_WaveForm_L2Error.png") -> None:
        """
        Plot the extracted waveform at ρ=s against the exact solution.
        """
        import matplotlib.pyplot as plt

        data   = np.asarray(self._probe_buffer)
        t      = data[:, 0]
        U_scri = data[:, 1]

        U_exact = exact_solution_at_rho(self.s, t, x0, self.Omega[-1, -1])

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        axes[0].plot(t, U_scri,  label="DG U at scri")
        axes[0].plot(t, U_exact, label="Exact", linestyle="dashed")
        axes[0].set_ylabel("U")
        axes[0].legend()
        axes[0].grid()

        axes[1].plot(t, np.abs(U_scri - U_exact), label="|error|")
        axes[1].set_ylabel("|U - U_exact|")
        axes[1].set_xlabel("τ")
        axes[1].legend()
        axes[1].grid()

        plt.suptitle(f"Waveform at ρ=s (D={self.D}, N={self.N})")
        plt.tight_layout()

        if filename is None:
            filename = f"graphs/scri_waveform_D{self.D}_N{self.N}.png"
        plt.savefig(filename, dpi=300)
        plt.close()

    # ------------------------------------------------------------------
    # Probe logger
    # ------------------------------------------------------------------

    def init_probe_logger(self, filename: str = "probe.csv") -> None:
        self.probe_file    = filename
        self._probe_buffer = []

    def log_probe(self, t: float) -> None:
        self._probe_buffer.append((
            t,
            self.u[-1, -1, 0],           # U  at ρ=s
            self.u[-1, -1, 1],           # q  at ρ=s
            self.u[-1, -1, 2],           # p  at ρ=s
            self.u[-1, -2, 0],           # U  one node inward
            self.u[self.D // 2, 0, 0],  # U  at domain midpoint
        ))

    def flush_probe(self) -> None:
        np.savetxt(
            self.probe_file,
            np.array(self._probe_buffer),
            delimiter=" ",
            header="t u_scri q_scri p_scri u_in u_mid u_scri_exact",
            comments="",
        )
def exact_solution(x: npt.NDArray, t: npt.NDArray, x0: float = 10.0) -> npt.NDArray:
    """
    Exact outgoing solution of

        - U_tt + U_rr - 6/r² U = 0

    generated by the profile

        f(u) = sin(u) exp(-u²),

    where u = t - r + x0.
    """
    u = t - x + x0   # retarded time argument, note sign: t - r not r - t

    f     =  np.sin(u) * np.exp(-u**2)
    fp    =  (np.cos(u) - 2.0 * np.sin(u) * u) * np.exp(-u**2)
    fpp   =  ((4.0*(u**2)-3.0)*np.sin(u)-4.0*u*np.cos(u)) * np.exp(-u**2)

    return fpp + (3.0/x)*fp + (3.0/x**2)*f

def exact_solution_at_rho(rho: npt.NDArray, tau: npt.NDArray, x0: float, Omega: npt.NDArray) -> npt.NDArray:
    """
    Evaluate the exact l=2 outgoing solution in hyperboloidal coordinates.

    Using

        r = rho / Ω(rho),

    and the flat-space hyperboloidal transformation,

        t - r = τ - rho,

    the exact solution can be evaluated directly as a function of
    (τ,ρ) without explicitly constructing t or r.
    """
    u   = tau - rho + x0

    f1   =  np.sin(u) * np.exp(-u**2)
    fp1  =  (np.cos(u) - 2*u*np.sin(u)) * np.exp(-u**2)
    fpp1 = (-3*np.sin(u) - 4*u*np.cos(u) + 4*u**2*np.sin(u)) * np.exp(-u**2)

    return fpp1 + (3.0 * Omega / rho) * fp1 + (3.0 * (Omega / rho)**2) * f1

def initial_state(
    x: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Initial data obtained from the exact outgoing l=2 solution.

    Returns
    -------
    f  : U(x,0)
    fx : ∂U/∂x(x,0)
    g  : -∂U/∂t(x,0)

    Spatial and temporal derivatives are evaluated using fourth-order
    finite differences applied to the exact solution.
    """

    u = -x + 10.0
    eps = 1e-5
    f1     =  np.sin(u) * np.exp(-u**2)
    fp1    =  (np.cos(u) - 2.0 * np.sin(u) * u) * np.exp(-u**2)
    fpp1   =  ((4.0*(u**2)-3.0)*np.sin(u)-4.0*u*np.cos(u)) * np.exp(-u**2)

    f = fpp1 + (3.0/x)*fp1 + (3.0/x**2)*f1

    fx = (
        -exact_solution(x + 2*eps, np.asarray(0.0), 10.0)
        + 8*exact_solution(x +   eps, np.asarray(0.0), 10.0)
        - 8*exact_solution(x -   eps, np.asarray(0.0), 10.0)
        +   exact_solution(x - 2*eps, np.asarray(0.0), 10.0)
    ) / (12.0 * eps)

    g = -(
        -exact_solution(x, np.asarray(2*eps), 10.0)
        + 8*exact_solution(x, np.asarray(  eps), 10.0)
        - 8*exact_solution(x, np.asarray( -eps), 10.0)
        +   exact_solution(x, np.asarray(-2*eps), 10.0)
    ) / (12.0 * eps)

    return f, fx, g


def effective_potential(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Effective potential for l=2 flat space wave equation. 
        V(r) = 6/r^2
    """
    return 6.0/x**2
