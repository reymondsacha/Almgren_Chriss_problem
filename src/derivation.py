import numpy as np
import scipy.linalg
class DiscreteAlmgrenChriss:
    """
    Our objective here is to minimize a negative loss function by choosing the right vector x.
    Each x_i corresponds to the portfolio at time i * tau.
    To do so we derived IS and observed the utility function U(x) = -E[IS] + lambda * V[IS]
    Then we put U(x) under the forme 1/2 * x.T * A * x = b *x and we now have to solve Ax=b
    """
    def __init__(self, X: float, T: float, N: int, sigma: float, eta: float, gamma: float, lambda_penalty: float) -> None:
        self.X = X
        self.T = T
        self.N = N
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.lambda_penalty = lambda_penalty
        self.tau = T / N
        self.eta_tilde = self.eta - 0.5 * self.gamma * self.tau

    def solve_trajectory(self) -> np.ndarray:
        
        # Here we construct A by first building its diagonals.
        diag_value = (4 * self.eta_tilde / self.tau) + 2 * self.lambda_penalty * (self.sigma**2) * self.tau
        off_diag_value = -2 * self.eta_tilde / self.tau
        
        diagonal = np.full(self.N - 1, diag_value)
        upper_lower_diagonal = np.full(self.N - 2, off_diag_value)

        # We construct b
        b = np.zeros(self.N - 1)
        b[0] = 2 * self.eta_tilde * self.X / self.tau

        # We construct ab the array needed in the linalg.solve_banded method from scipy that allows
        # us to solve the system in O(n)
        ab = np.zeros((3, self.N - 1))
        ab[0, 1:] = upper_lower_diagonal
        ab[1, :] = diagonal
        ab[2, :-1] = upper_lower_diagonal

        # We determine x
        x_star = scipy.linalg.solve_banded(l_and_u=(1, 1), ab=ab, b=b)

        # We add x_0 = X and x_N = 0 respectively at the beginning and end of the created vector x_star
        x_star = np.concatenate(([self.X], x_star, [0]))

        return x_star

    def ode_solution(self):
        """
        We derived the ODE x"(t) - K x(t) = 0 from the matrix equation system and solved it
        """
        K = np.sqrt(self.lambda_penalty / self.eta_tilde)  * self.sigma

        n = np.arange(0,self.N + 1)
        time = n * self.tau
        x = self.X * (np.sinh(K*(self.T - time))) / np.sinh(K * self.T)

        return x


