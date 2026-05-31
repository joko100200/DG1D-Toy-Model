import numpy as np
import numpy.typing as npt
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
        self.compute_MassMatrix()
 
    def lagrange_basis_matrix(self):
        """
        Lagrange basis functions on the reference element [-1, 1] stored in a matrix.

        Stores
        -------
        Phi : (N+1, L) array
            Phi[j, k] = phi_j(epsilon_k)
        """

        # reference nodes (interpolation nodes)
        self.xi_nodes = np.linspace(-1.0, 1.0, self.N + 1)

        # evaluation grid
        self.quad_points = np.linspace(-1.0, 1.0, self.L)

        Phi = np.zeros((self.N + 1, self.L), dtype=float)

        for j in range(self.N + 1):
            # start with ones
            lj = np.ones_like(self.quad_points, dtype=float)

            xj = self.xi_nodes[j]

            for m, xm in enumerate(self.xi_nodes):
                if m != j:
                    lj *= (self.quad_points - xm) / (xj - xm)

            Phi[j, :] = lj

        self.Phi = Phi
    
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
        epsilon = np.linspace(-1.0, 1.0, self.L, dtype=np.float64)

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

        ui = u0(self.reconstruct_x())
        epsilon = np.linspace(-1.0, 1.0, self.L, dtype=np.float64)

        b = np.zeros((self.D, self.N + 1))

        for j in range(self.D):

            for i in range(self.N + 1):
                b[j, i] = np.trapz(
                    ui[j,:] * self.Phi[i,:],
                    epsilon
                )

        #self.u = np.zeros_like(b)

        self.u = (self.inv_M @ b.T).T
         
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

        epsilon = np.linspace(-1.0, 1.0, self.L, dtype=np.float64)

        M = np.zeros((self.N + 1, self.N + 1), dtype=np.float64)

        for i in range(self.N + 1):
            for j in range(self.N + 1):
                M[i, j] = np.trapz(
                    self.Phi[i, :] * self.Phi[j, :],
                    epsilon
                )

        self.M = M
        self.inv_M = np.linalg.inv(M)

    def error_in_u0(self, u0 : Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]) -> None:
        """Compute the maximum error between the reconstructed solution uh and the exact solution u0."""
        u_exact = u0(self.reconstruct_x())
        uh = self.u @ self.Phi

        error = np.max(np.abs(uh - u_exact))

        #print("reconstructed: ", uh)
        #print("exact: ", u_exact)
        #print("coefficients: ", self.u)
        #print("inverse mass matrix: ", self.inv_M)
        #print("mass matrix: ", self.M)
        #print("basis functions: ", self.Phi)

        print("Max error =", error)

def gaussian(x : npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A Gaussian function centered at 0 with standard deviation 1."""
    return x**5#np.exp(-0.5 * x**2)
