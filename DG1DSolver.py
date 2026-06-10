import numpy as np
import numpy.typing as npt
from numpy.polynomial.legendre import Legendre
from collections.abc import Callable


class DG1DSolver:
    """
    Discontinuous Galerkin solver for the 1D wave equation.

    Solves the first-order system:
        d/dt U = p
        d/dt q = d/dx p       (q = d/dx U)
        d/dt p = d/dx q - V(x)*U

    State is stored as u of shape (D, N+1, 3) where:
        u[:, :, 0] = U  (displacement)
        u[:, :, 1] = q  (spatial derivative of U)
        u[:, :, 2] = p  (time derivative of U)

    Since GLL nodes + Lagrange basis give Phi_q = I:
        M   = diag(quad_weights)
        M^-1 = diag(1/quad_weights)
    All mass matrix operations reduce to elementwise division by quad_weights.
    The projection of any function onto the basis is just its nodal values.
    """

    def __init__(self, x_grid: npt.NDArray[np.float64], N: int, L: int, R: float, P: int, outputfileDir: str):
        self.D = len(x_grid) - 1
        self.N = N
        self.L = L

        self.x = x_grid
        self.h = (self.x[1:] - self.x[:-1]) / 2.0          # shape (D,)

        self.R = R
        self.P = P

        self.s = self.x[-1]   # right boundary of physical domain

        self.xi_nodes    = self.gauss_lobatto_nodes()
        self.quad_weights = self.gauss_lobatto_weights()     # shape (N+1,)
        self.inv_w       = 1.0 / self.quad_weights           # shape (N+1,)

        self.lagrange_basis_matrix()   # builds Phi_plot only — Phi_q = I
        self.calculate_D_Phi()

        self.init_probe_logger(outputfileDir) #File to save waveform output

        self.x_nodes     = self.reconstruct_x_at_nodes()    # shape (D, N+1)
        self.V_at_nodes  = effective_potential(self.x_nodes) # shape (D, N+1)

        self.H_at_nodes  = self.compute_H()                # shape (D, N+1)

        denom          = 1.0 - self.H_at_nodes**2
        safe           = denom > 1e-14
        limit_at_scri  = 0.0   # compute V(r)*r² / (dΩ²/dρ evaluated at s) analytically

        self.JV_at_nodes = np.where(safe, self.V_at_nodes / np.where(safe, denom, 1.0), limit_at_scri) #(D, N+1)
        self.coeff_at_nodes = 1.0 / (1.0 + self.H_at_nodes)    # (D, N+1)

        self.u_fine = np.zeros_like(self.x_nodes)

    # ------------------------------------------------------------------
    # Basis and quadrature
    # ------------------------------------------------------------------

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

    def lagrange_basis_matrix(self):
        """
        Build Phi_plot only — the dense evaluation grid for plotting.
        Phi_q = I by the Kronecker delta property so we don't store it.

        Stores
        ------
        Phi_plot : (N+1, L)   Phi_plot[j, k] = phi_j(epsilon_k)
        grid_points : (L,)    dense reference grid
        """
        self.grid_points = np.linspace(-1.0, 1.0, self.L, dtype=np.float64)
        Phi_plot = np.zeros((self.N + 1, self.L))

        for j in range(self.N + 1):
            lj_p = np.ones(self.L)
            xj   = self.xi_nodes[j]
            for m, xm in enumerate(self.xi_nodes):
                if m != j:
                    lj_p *= (self.grid_points - xm) / (xj - xm)
            Phi_plot[j, :] = lj_p

        self.Phi_plot = Phi_plot

    def calculate_D_Phi(self):
        """
        Derivative matrix via barycentric weights.
        D_Phi[i, j] = phi_j'(xi_i)   shape (N+1, N+1)
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
    
    def compute_H(self) -> npt.NDArray[np.float64]:
        """
        Compute H(rho) from the hyperboloidal layer definition (Scott et al. Eq. 19, 21).
        H = 0 for rho ≤ R (outside the layer).
        """
        rho   = self.x_nodes          # (D, N+1), your computational coordinate
        rho_R = self.x[np.argmin(np.abs(self.x - self.R))] #closes interface
        layer = rho > rho_R          # mask for interior of layer

        H     = np.zeros_like(rho)

        sigma      = (rho[layer] - self.R) / (self.s - self.R)
        Omega      = 1.0 - sigma**self.P
        Omega_prime = -self.P * sigma**(self.P - 1) / (self.s - self.R)

        H[layer] = 1.0 - Omega**2 / (Omega - rho[layer] * Omega_prime)

        return H

    # ------------------------------------------------------------------
    # Grid reconstruction
    # ------------------------------------------------------------------

    def reconstruct_x(self) -> npt.NDArray[np.float64]:
        """Physical coordinates on the dense grid. Shape (D, L)."""
        x_grid = np.asarray(self.x, dtype=np.float64)
        x_reconstructed = np.zeros((self.D, self.L), dtype=np.float64)
        for j in range(self.D):
            dx = x_grid[j + 1] - x_grid[j]
            x_reconstructed[j, :] = x_grid[j] + 0.5 * dx * (self.grid_points + 1.0)
        return x_reconstructed

    def reconstruct_x_at_nodes(self) -> npt.NDArray[np.float64]:
        """Physical coordinates at GL nodes. Shape (D, N+1)."""
        x_grid = np.asarray(self.x, dtype=np.float64)
        x_nodes = np.zeros((self.D, self.N + 1), dtype=np.float64)
        for j in range(self.D):
            dx = x_grid[j + 1] - x_grid[j]
            x_nodes[j, :] = x_grid[j] + 0.5 * dx * (self.xi_nodes + 1.0)
        return x_nodes

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    def initialize_solution(
        self,
        init_fn: Callable[
            [npt.NDArray[np.float64]],
            tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]
        ]
    ) -> None:
        """
        Project initial conditions onto the DG basis.

        Since Phi_q = I, the DG coefficients are just the nodal values.
        No mass matrix solve needed.

        Parameters
        ----------
        init_fn : callable
            Function of x returning (f, fx, g) where
                f  = U(x, 0)
                fx = dU/dx(x, 0)
                g  = dU/dt(x, 0)

        Stores
        ------
        self.u : (D, N+1, 3)
            u[:, :, 0] = U
            u[:, :, 1] = q = dU/dx
            u[:, :, 2] = p = dU/dt
        """
        f_vals, fx_vals, g_vals = init_fn(self.x_nodes)

        self.u = np.zeros((self.D, self.N + 1, 3), dtype=np.float64)
        self.u[:, :, 0] = f_vals
        self.u[:, :, 1] = fx_vals
        self.u[:, :, 2] = g_vals

    # ------------------------------------------------------------------
    # RHS
    # ------------------------------------------------------------------

    def rhs(self, u: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        RHS for the system:
            dU/dt = -p
            dq/dt = -d/dx p          (q = dU/dx)
            dp/dt = -d/dx q - V(x)*U

        Since M = diag(w) and inv_M = diag(1/w):
            inv_M @ (vol - b) becomes (vol - b) * inv_w  elementwise.

        Parameters
        ----------
        u : (D, N+1, 3)

        Returns
        -------
        dudt : (D, N+1, 3)
        """
        U = u[:, :, 0]   # (D, N+1)
        q = u[:, :, 1]   # (D, N+1)
        p = u[:, :, 2]   # (D, N+1)

        # ---- volume terms ----------------------------------------
        # vol[e, k] = sum_m w_m * f_m * D_Phi[m, k]
        vol_p = (self.quad_weights * q) @ self.D_Phi   # driven by q
        vol_q = (self.quad_weights * p) @ self.D_Phi   # driven by p
        #vol_p = np.einsum('i,ei,ij->ej', self.quad_weights, q, self.D_Phi)
        #vol_q = np.einsum('i,ei,ij->ej', self.quad_weights, p, self.D_Phi)

        # ---- interface values -------------------------
        p_minus_L = np.roll(p[:, -1],  1)
        p_plus_L  = p[:, 0]
        p_minus_R = p[:, -1]
        p_plus_R  = np.roll(p[:, 0], -1)

        q_minus_L = np.roll(q[:, -1],  1)
        q_plus_L  = q[:, 0]
        q_minus_R = q[:, -1]
        q_plus_R  = np.roll(q[:, 0], -1)

        # global right boundary (ρ = s): outflow, no incoming state
        p_plus_R[-1] = p_minus_R[-1]
        q_plus_R[-1] = q_minus_R[-1]

        # global left boundary: reflecting (zero flux into domain)
        p_minus_L[0] = -p_plus_L[0]
        q_minus_L[0] = -q_plus_L[0]

        # ---- upwind fluxes ---------------------------------------
        hat_q_L = 0.5*(q_minus_L + q_plus_L) + 0.5*(p_minus_L - p_plus_L)
        hat_q_R = 0.5*(q_minus_R + q_plus_R) + 0.5*(p_minus_R - p_plus_R)

        hat_p_L = 0.5*(p_minus_L + p_plus_L) + 0.5*(q_minus_L - q_plus_L)
        hat_p_R = 0.5*(p_minus_R + p_plus_R) + 0.5*(q_minus_R - q_plus_R)

        # ---- boundary vectors ------------------------------------
        # p equation uses hat_q; q equation uses hat_p
        b_p = np.zeros_like(p)
        b_q = np.zeros_like(q)
        b_p[:, 0]  = -hat_q_L;  b_p[:, -1] = hat_q_R
        b_q[:, 0]  = -hat_p_L;  b_q[:, -1] = hat_p_R

        # ---- source term -V(x)*U in dp/dt -----------------------
        potentialVU = self.JV_at_nodes * U   # (D, N+1), no mass matrix needed

        # ---- hyperboloidal terms ---------------------------------
        H = self.H_at_nodes
        coeff = self.coeff_at_nodes

        # ---- assemble --------------------------------------------
        # inv_M @ v = v * inv_w  (M is diagonal)
        # sign: (b - vol) from weak form derivation
        dp = ((b_p - vol_p) * self.inv_w) / self.h[:, None]
        dq = ((b_q - vol_q) * self.inv_w) / self.h[:, None]

        dudt = np.zeros_like(u)
        dudt[:, :, 0] = -p
        dudt[:, :, 1] = coeff * (-dq - H * dp) - H * potentialVU
        dudt[:, :, 2] = coeff * (-dp - H * dq) - potentialVU

        return dudt

    # ------------------------------------------------------------------
    # Time integration
    # ------------------------------------------------------------------

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
        print(f"t = 0.0 | L2 = {self.L2_error_self(self.u_fine):.6e} | E = {self.compute_energy():.6e}")
        while t < T:
            if t + dt > T:
                dt = T - t
            self.step_rk4(dt)
            t += dt
        print(f"t = {T} | L2 = {self.L2_error_self(self.u_fine):.6e} | E = {self.compute_energy():.6e}")
        return self.u

    def runDEBUG(self, T: float, cfl: float = 0.5) -> npt.NDArray[np.float64]:
        t   = 0.0
        dt  = self.compute_dt(cfl)

        print(f"t = {t:.3f} | L2Norm = {self.compute_L2_norm()} | E = {self.compute_energy():.6e}")
        while t < T:
            if t + dt > T:
                dt = T - t
            self.step_rk4(dt)
            #print(t, self.compute_L2_norm(), self.compute_energy(), self.energy_in_region(self.x[0], self.R), self.energy_in_region(self.R, self.x[-1]))
            t += dt
            self.log_probe(t)
        
        self.flush_probe()
        
        print(f"t = {t:.3f} | L2Norm = {self.compute_L2_norm()} | E = {self.compute_energy():.6e}")
        return self.u

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def L2_error_self(self, u_fine: npt.NDArray[np.float64]) -> float:
        """
        Self-convergence L2 error: compare self.u against a fine-grid solution
        u_fine interpolated to the coarse nodes. Only valid when fine grid is
        exactly 2x or 4x the coarse resolution so nodes are nested.
        """
        if not np.any(u_fine): return -1.0

        diff = self.u - u_fine   # assumes same node layout, caller handles interpolation
        err  = 0.0
        for e in range(self.D):
            err += np.sum(
                self.quad_weights * (diff[e, :, 0]**2 +
                                     diff[e, :, 1]**2 +
                                     diff[e, :, 2]**2)
            ) * self.h[e]
        return np.sqrt(err)
    
    def L2_error_probe_state(self, u_fine : npt.NDArray[np.float64]) -> float:
        """
        L2 error between this solver's scri waveform and a reference (fine) waveform.
        Both inputs are probe buffers: lists of (t, U, q, p) tuples.
        Interpolates the fine solution onto the coarse time grid before differencing.
        """
        u_coarse = np.array(self._probe_buffer)   # (M_coarse, 4)

        t_coarse = u_coarse[:, 0]
        t_fine   = u_fine[:, 0]

        # interpolate fine onto coarse time grid
        U_fine_interp = np.interp(t_coarse, t_fine, u_fine[:, 1])
        #q_fine_interp = np.interp(t_coarse, t_fine, u_fine[:, 2])
        #p_fine_interp = np.interp(t_coarse, t_fine, u_fine[:, 3])

        dU = u_coarse[:, 1] - U_fine_interp
        #dq = u_coarse[:, 2] - q_fine_interp
        #dp = u_coarse[:, 3] - p_fine_interp

        return np.sqrt(np.trapz(dU**2, t_coarse))

    @staticmethod 
    def L2_error_probe_state_diff(u_coarse: npt.NDArray[np.float64], u_fine: npt.NDArray[np.float64]) -> float:
        """
        L2 error between this solver's scri waveform and a reference (fine) waveform.
        Both inputs are probe buffers: lists of (t, U, q, p) tuples.
        Interpolates the fine solution onto the coarse time grid before differencing.
        """

        t_coarse = u_coarse[:, 0]
        t_fine   = u_fine[:, 0]

        # interpolate fine onto coarse time grid
        U_fine_interp = np.interp(t_coarse, t_fine, u_fine[:, 1])
        #q_fine_interp = np.interp(t_coarse, t_fine, u_fine[:, 2])
        #p_fine_interp = np.interp(t_coarse, t_fine, u_fine[:, 3])

        dU = u_coarse[:, 1] - U_fine_interp
        #dq = u_coarse[:, 2] - q_fine_interp
        #dp = u_coarse[:, 3] - p_fine_interp

        return np.sqrt(np.trapz(dU**2, t_coarse))
        

    def waveform_at_scri(self) -> tuple[float, float, float]:
        """
        Extract U, q, p at the last node of the last element (ρ = s).
        This is your primary observable — the far-field waveform.
        Returns (U, q, p) at scri+.
        """
        return (self.u[-1, -1, 0],
                self.u[-1, -1, 1],
                self.u[-1, -1, 2])

    def compute_energy(self) -> float:
        """
        E = 0.5 * integral (p^2 + q^2 + V*U^2) dx
        Conserved by the wave system.
        """
        U   = self.u[:, :, 0]
        q   = self.u[:, :, 1]
        p   = self.u[:, :, 2]

        integrand = (p**2 + q**2 + self.V_at_nodes * U**2)
        return float(0.5 * np.sum(self.quad_weights * integrand * self.h[:, None]))

    def energy_in_region(self, x_min: float, x_max: float) -> float:
        U = self.u[:, :, 0]
        q = self.u[:, :, 1]
        p = self.u[:, :, 2]

        # element-wise mask (D, N+1)
        x = self.x_nodes
        mask = (x >= x_min) & (x <= x_max)

        integrand = (p**2 + q**2 + self.V_at_nodes * U**2)

        # apply mask
        integrand = integrand * mask

        # quadrature over reference element
        local_energy = np.sum(self.quad_weights * integrand, axis=1)  # (D,)

        # physical scaling
        local_energy *= self.h

        return 0.5 * np.sum(local_energy)

    def compute_L2_norm(self) -> float:
        """L2 norm of U with hyperboloidal volume element."""
        U      = self.u[:, :, 0]
        return np.sqrt(np.sum(self.quad_weights * U**2 * self.h[:, None]))

    def error_in_u0(self, init_fn) -> None:
        """Print projection error at t=0."""
        f_exact, fx_exact, g_exact = init_fn(self.x_nodes)
        U_h = self.u[:, :, 0]
        q_h = self.u[:, :, 1]
        p_h = self.u[:, :, 2]
        print(f"Projection error U: {np.sqrt(np.mean((U_h - f_exact)**2)):.6e}")
        print(f"Projection error q: {np.sqrt(np.mean((q_h - fx_exact)**2)):.6e}")
        print(f"Projection error p: {np.sqrt(np.mean((p_h - g_exact)**2)):.6e}")

    def plot_solution(self, t: float, filename: str = "graphs/DG_wave_solution.png") -> None:
        import matplotlib.pyplot as plt

        x_dense = self.reconstruct_x().reshape(-1)
        U_dense = (self.u[:, :, 0] @ self.Phi_plot).reshape(-1)
        q_dense = (self.u[:, :, 1] @ self.Phi_plot).reshape(-1)
        p_dense = (self.u[:, :, 2] @ self.Phi_plot).reshape(-1)

        domain_len = self.x[-1] - self.x[0]
        x_plus  = ((x_dense + t - self.x[0]) % domain_len) + self.x[0]
        x_minus = ((x_dense - t - self.x[0]) % domain_len) + self.x[0]
        _, fx_plus,  g_plus  = initial_state(x_plus)
        _, fx_minus, g_minus = initial_state(x_minus)
        p_exact = 0.5 * (g_plus  + g_minus - fx_plus  + fx_minus)
        q_exact = 0.5 * (g_plus  + g_minus - fx_plus  + fx_minus)

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

    def init_probe_logger(self, filename: str = "probe.csv"):
        self.probe_file = filename
        self._probe_buffer = []
    
    def log_probe(self, t: float):
        self._probe_buffer.append((t, self.u[-1, -1, 0], self.u[-1, -1, 1], self.u[-1, -1, 2]))
    
    def flush_probe(self):
        data = np.array(self._probe_buffer)
        np.savetxt(self.probe_file, data, delimiter=" ", header="t, u_edge, q_edge, p_edge", comments="")


# ------------------------------------------------------------------
# Problem setup
# ------------------------------------------------------------------

def initial_state(
    x: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Initial state.

    Returns
    -------
    f  : U(x, 0)
    fx : dU/dx(x, 0) = q(x, 0)
    g  : -dU/dt(x, 0) = p(x, 0)
    """
    f = np.exp(-0.5 * x**2)
    fx = -x * np.exp(-0.5 * x**2)
    g = -x * np.exp(-0.5 * x**2)
    return f, fx, g

def effective_potential(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Effective potential V(x). Appears as -V(x)*U in dp/dt.
    Return zeros for the pure wave equation.
    """
    return np.zeros_like(x)