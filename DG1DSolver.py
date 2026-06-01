import numpy as np
import numpy.typing as npt
from numpy.polynomial.legendre import Legendre
from collections.abc import Callable

class DG1DSolver:
    """
    Discontinuous Galerkin solver for 1D problems.

    Attributes
    ----------
    D : int
        Number of cells.
    N : int
        Number of basis functions per cell.
    L : int
        Number of evaluation points for basis functions on epsilon [-1,1].
    u : (D, N+1) array
        Coefficients of the DOF in each cell D and for each basis function N.
    x : (D+1,) array
        Physical grid points defining the cell boundaries.
    xi_nodes : (N+1,) array
        Gauss-Lobatto nodes on the reference element [-1, 1].
    quad_weights : (N+1,) array
        Quadrature weights for the Gauss-Lobatto nodes.
    h : (D,) array
        Cell widths used for physical coordinate mapping.
    phi_q : (N+1, N+1) array
        Lagrange basis functions evaluated on guass-lobatto nodes.
    phi_plot : (N+1, L) array
        Lagrange basis functions evaluated on a dense grid for plotting.
    M : (N+1, N+1) array
        Reference mass matrix.
    inv_M : (N+1, N+1) array
        Inverse of the reference mass matrix.
    
    Methods
    -------
    lagrange_basis_matrix() -> (N+1, L) array
        Computes the Lagrange basis functions on the reference element [-1, 1] and stores them in a matrix.
    reconstruct_x() -> (D, L) array
        Maps reference element points epsilon in [-1,1] to physical coordinates for all DG elements.
    reconstruct_x_at_nodes() -> (D, N+1) array
        Maps reference element nodes epsilon in [-1,1] to physical coordinates for all DG elements.
    initialize_solution(u0) -> None
        Initializes the solution coefficients u[j, i] by projecting a given function onto the basis functions.
    compute_MassMatrix() -> None
        Computes the reference-element mass matrix and its inverse.
    gauss_lobatto_nodes() -> (N+1,) array
        Computes the Gauss-Lobatto nodes for a given number of basis functions N.
    gauss_lobatto_weights() -> (N+1,) array
        Computes Gauss-Lobatto-Legendre quadrature weights on [-1,1].
    error_in_u0(u0) -> None
        Computes the maximum error between the reconstructed solution uh and the exact solution u0.
    
    Functions
    ---------
    gaussian(x) -> (shape(x)) array
        A Gaussian function.
    """
    def __init__(self, x_grid : npt.NDArray[np.float64], N: int, L: int):
        self.D = len(x_grid) - 1
        self.N = N
        self.L = L

        self.x = x_grid
        self.h = (self.x[1:] - self.x[:-1])/2.0

        self.lagrange_basis_matrix()
        self.quad_weights = self.gauss_lobatto_weights()
        self.compute_MassMatrix()
        self.calculate_D_Phi()

        self.x_nodes = self.reconstruct_x_at_nodes()
 
    def lagrange_basis_matrix(self):
        """
        Lagrange basis functions on the reference element [-1, 1] defined by the Gauss-Lobatto nodes.

        Stores
        -------
        Phi_q : (N+1, N+1) array
            Phi_q[j, m] = phi_j(xi_m) where xi_m are the Gauss-Lobatto nodes.
        Phi_plot : (N+1, L) array
            Phi[j, k] = phi_j(epsilon_k) on a dense grid of L points for plotting and error analysis.
        """

        # reference nodes (interpolation nodes)
        self.xi_nodes = self.gauss_lobatto_nodes()

        # evaluation grid
        self.grid_points = np.linspace(-1.0, 1.0, self.L, dtype=np.float64)

        Phi_q = np.zeros((self.N + 1, self.N + 1))  # nodal basis at nodal points
        Phi_plot = np.zeros((self.N + 1, self.L))   # dense grid for plotting

        for j in range(self.N + 1):
            lj_q = np.ones(self.N + 1)
            lj_p = np.ones(self.L)

            xj = self.xi_nodes[j]

            for m, xm in enumerate(self.xi_nodes):
                if m != j:
                    lj_q *= (self.xi_nodes - xm) / (xj - xm)
                    lj_p *= (self.grid_points - xm) / (xj - xm)

            Phi_q[j, :] = lj_q
            Phi_plot[j, :] = lj_p

        self.Phi_q = Phi_q
        self.Phi_plot = Phi_plot
    
    def reconstruct_x(self) -> npt.NDArray[np.float64]:
        """
        Map reference element points epsilon in [-1,1]
        to physical coordinates for all DG elements.

        Returns
        -------
        x_reconstructed : ndarray, shape (D, L)
            Physical x-values for each element and reference point
        """

        x_grid = np.asarray(self.x, dtype=np.float64)
        epsilon = self.grid_points

        x_reconstructed = np.zeros((self.D, self.L), dtype=np.float64)

        for j in range(self.D):
            x_left = x_grid[j]
            x_right = x_grid[j + 1]

            dx = x_right - x_left

            x_reconstructed[j, :] = x_left + 0.5 * dx * (epsilon + 1.0)

        # To get this plotted you just do x_reconstructed.reshape(-1)
        return x_reconstructed
    
    def initialize_solution(self, u0 : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> None:
        """
        Initialize the solution coefficients u[j, i] by projecting a given function onto the basis functions.

        Parameters
        ----------
        u0 : numpy function
            A function that a vectorized argument and returns a value for all elements.
        """

        ui = u0(self.reconstruct_x_at_nodes())

        b = np.zeros((self.D, self.N + 1))

        for j in range(self.D):

            for i in range(self.N + 1):
                b[j, i] = np.sum(
                    self.quad_weights * ui[j,:] * self.Phi_q[i,:]
                )

        self.u = (self.inv_M @ b.T).T

    def reconstruct_x_at_nodes(self) -> npt.NDArray[np.float64]:
        """
        Map reference element nodes epsilon in [-1,1]
        to physical coordinates for all DG elements.

        Returns
        -------
        x_reconstructed : ndarray, shape (D, N+1)
            Physical x-values for each element and reference node
        """

        x_grid = np.asarray(self.x, dtype=np.float64)

        x_reconstructed_at_nodes = np.zeros((self.D, self.N + 1), dtype=np.float64)

        for j in range(self.D):
            x_left = x_grid[j]
            x_right = x_grid[j + 1]

            dx = x_right - x_left

            x_reconstructed_at_nodes[j, :] = x_left + 0.5 * dx * (self.xi_nodes + 1.0)

        return x_reconstructed_at_nodes
         
    def compute_MassMatrix(self) -> None:
        """
        Compute the reference-element mass matrix

            M_ij = ∫_{-1}^{1} phi_i(epsilon) phi_j(epsilon) d epsilon

        using the Gauss-Lobatto quadrature rule. 

        Stores
        -------
        self.M : (N+1, N+1) ndarray
            Reference mass matrix.

        self.inv_M : (N+1, N+1) ndarray
            Inverse of the reference mass matrix.
        """

        M = np.zeros((self.N + 1, self.N + 1), dtype=np.float64)

        for i in range(self.N + 1):
            for j in range(self.N + 1):

                M[i, j] = np.sum(
                    self.quad_weights * self.Phi_q[i, :] * self.Phi_q[j, :],
                )

        self.M = M
        self.inv_M = np.linalg.inv(M)

    def error_in_u0(self, u0 : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> None:
        """Compute the maximum error between the reconstructed solution uh and the exact solution u0."""
        u_exact = u0(self.reconstruct_x())
        uh = self.u @ self.Phi_plot

        error = np.sqrt(np.mean((uh-u_exact)**2))

        print("L2 error =", error)

        u_exact = u0(self.reconstruct_x_at_nodes())
        uh = self.u
        error = np.sqrt(np.mean((uh-u_exact)**2))

        print("L2 error at nodes =", error)

    def gauss_lobatto_nodes(self) -> npt.NDArray[np.float64]:
        """
        Compute the Gauss-Lobatto nodes for a given number of basis functions N. 
        """
        if self.N == 1:
            return np.array([-1.0, 1.0], dtype=np.float64)

        P = Legendre.basis(self.N)
        dP = P.deriv()

        interior = dP.roots()
        nodes = np.concatenate(([-1.0], np.sort(interior), [1.0]), dtype=np.float64)

        return nodes
    
    def gauss_lobatto_weights(self) -> npt.NDArray[np.float64]:
        """
        Compute Gauss-Lobatto-Legendre quadrature weights on [-1,1].
        """

        N = self.N
        nodes = self.gauss_lobatto_nodes()

        if N == 1:
            return np.array([1.0, 1.0], dtype=np.float64)

        Pn = Legendre.basis(N)

        weights = np.zeros(N + 1, dtype=np.float64)

        for i, x in enumerate(nodes):
            weights[i] = 2.0 / (N * (N + 1)) / (Pn(x) ** 2)

        return weights

    def calculate_D_Phi(self):
        """Calculate the derivative matrix D_Phi for the Lagrange basis functions at GL nodes."""

        Np = self.N + 1
        x = self.xi_nodes

        # Barycentric weights
        lam = np.ones(Np, dtype=np.float64)

        for j in range(Np):
            for m in range(Np):
                if m != j:
                    lam[j] /= (x[j] - x[m])

        # Derivative matrix:
        # D_Phi[i, j] = phi_j'(x_i)
        D_Phi = np.zeros((Np, Np), dtype=np.float64)

        for i in range(Np):
            for j in range(Np):
                if i != j:
                    D_Phi[i, j] = (
                        lam[j]
                        / lam[i]
                        / (x[i] - x[j])
                    )

        for i in range(Np):
            D_Phi[i, i] = -np.sum(D_Phi[i])

        self.D_Phi = D_Phi

    def rhs(self, u : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Returns all the RHS values for each element."""
        rhs_all = np.zeros_like(u)

        for e in range(self.D):
            rhs_all[e] = self.rhs_element(e, u)
        
        return rhs_all

    def rhs_element(self, e, u : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        
        u_e    = u[e]
        x_e    = self.x_nodes[e]
        a_e    = a(x_e)                                        # a at physical nodes

        # volume term
        vol = self.D_Phi.T @ (self.quad_weights * a_e * u_e)

        # flux vector b
        f_left, f_right = self.numerical_flux(u, e)
        b      = np.zeros(self.N + 1)
        b[0]   = -f_left
        b[-1]  = +f_right

        return (1.0 / self.h[e]) * (self.inv_M @ (vol - b))
    
    def numerical_flux(self, u, e):
        xL = self.x[e]
        xR = self.x[e + 1]
        aL = a(xL)
        aR = a(xR)

        # left face
        u_minus_L = u[e - 1, -1] if e > 0 else u[-1, -1]  # periodic
        u_plus_L  = u[e, 0]
        f_left    = aL * u_minus_L if aL >= 0 else aL * u_plus_L

        # right face
        u_minus_R = u[e, -1]
        u_plus_R  = u[e + 1, 0] if e < self.D - 1 else u[0, 0]  # periodic
        f_right   = aR * u_minus_R if aR >= 0 else aR * u_plus_R

        return f_left, f_right

    def compute_dt(self, cfl):
        x = self.reconstruct_x_at_nodes()
        a_max = 0.0

        for e in range(self.D):
            a_vals = a(x[e])
            a_max = max(a_max, np.max(np.abs(a_vals)))

        h_min = np.min(self.h)

        return cfl * h_min / ((2*self.N + 1) * a_max)

    def step_rk4(self, dt):
        u0 = self.u.copy()

        k1 = self.rhs(u0)
        k2 = self.rhs(u0 + 0.5 * dt * k1)
        k3 = self.rhs(u0 + 0.5 * dt * k2)
        k4 = self.rhs(u0 + dt * k3)

        self.u = u0 + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def run(self, T, cfl=0.5):
        t = 0.0
        dt = self.compute_dt(cfl)
        print("t=0 L2 error = ", self.L2_error(0.0))

        while t < T:
            if t + dt > T:
                dt = T - t

            self.step_rk4(dt)
            t += dt

            print("t =", t, ", L2 error =", self.L2_error(t))

        return self.u

    def L2_error(self, t):

        err = 0.0

        for e in range(self.D):

            x_e = self.reconstruct_x_at_nodes()[e]

            u_exact = gaussian(x_e - t)   # <-- THIS is the key advection shift
            u_h = self.u[e]

            err += np.sum(
                self.quad_weights * (u_h - u_exact)**2
            ) * self.h[e]

        return np.sqrt(err)

def gaussian(x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A Gaussian function centered at 0 with standard deviation 1."""
    return np.asarray(np.exp(-0.5 * x**2), dtype=np.float64)

def a (x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A variable wave speed function."""
    return np.ones_like(x, dtype=np.float64)
