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
        Coefficients of the solution in each cell D and for each basis function N.
    x : (D+1,) array
        Grid points defining the cell boundaries.
    phi : (N+1, L) array
        Lagrange basis functions evaluated at L points in the reference element [-1, 1].
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
    initialize_solution(u0) -> None
        Initializes the solution coefficients u[j, i] by projecting a given function onto the basis functions.
    compute_MassMatrix() -> None
        Computes the reference-element mass matrix and its inverse.
    """
    def __init__(self, x_grid : npt.NDArray[np.float64], N: int, L: int):
        self.D = len(x_grid) - 1
        self.N = N
        self.L = L

        # u[j, i] jth cell and ith basis function coefficient
        # self.u = np.zeros((self.D, self.N + 1))

        self.x = x_grid

        self.lagrange_basis_matrix()
        self.quad_weights = self.gauss_lobatto_weights()
        self.compute_MassMatrix()
 
    def lagrange_basis_matrix(self):
        """
        Lagrange basis functions on the reference element [-1, 1] stored in a matrix.

        Stores
        -------
        Phi_q : (N+1, N+1) array
            Phi_q[j, m] = phi_j(xi_m) where xi_m are the Gauss-Lobatto nodes.
        Phi_plot : (N+1, L) array
            Phi[j, k] = phi_j(epsilon_k)
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

        x_reconstructed = np.zeros((self.D, self.N + 1), dtype=np.float64)

        for j in range(self.D):
            x_left = x_grid[j]
            x_right = x_grid[j + 1]

            dx = x_right - x_left

            x_reconstructed[j, :] = x_left + 0.5 * dx * (self.xi_nodes + 1.0)

        return x_reconstructed
         
    def compute_MassMatrix(self) -> None:
        """
        Compute the reference-element mass matrix

            M_ij = ∫_{-1}^{1} phi_i(epsilon) phi_j(epsilon) d epsilon

        using the same epsilon grid used to construct self.Phi.

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

        #print("reconstructed: ", uh)
        #print("exact: ", u_exact)
        #print("coefficients: ", self.u)
        #print("inverse mass matrix: ", self.inv_M)
        #print("mass matrix: ", self.M)
        #print("basis functions: ", self.Phi)

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

def gaussian(x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A Gaussian function centered at 0 with standard deviation 1."""
    return np.asarray(1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2), dtype=np.float64)
