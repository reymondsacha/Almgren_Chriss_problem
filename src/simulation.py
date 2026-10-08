import numpy as np
import pandas as pd

class Monte_Carlo_Simulator:

    def __init__(self, X: float, T: float, N:int, M: int, x: np.ndarray, v: np.ndarray, S0: float, sigma: float, gamma: float, eta: float):
        self.X = X
        self.T = T
        self.N = N
        self.M = M
        self.x = x 
        self.v = v 
        self.S0 = S0
        self.sigma = sigma
        self.eta = eta
        self.gamma = gamma
        self.tau = T / N




    def empirical_simulation(self):
        # 1. Generate innovations for the N-1 steps
        noise = np.random.normal(loc=0, scale=1, size=(self.M, self.N - 1))
        
        # 2. Cumulative sum, padded with zeros for k=1 (where stochastic sum is 0)
        cum_noise = np.hstack((np.zeros((self.M, 1)), np.cumsum(noise, axis=1)))
        
        # 3. Calculate S_{k-1} using self.x[:-1] which corresponds to x_{k-1}
        matrix_S_prev = self.S0 + self.sigma * np.sqrt(self.tau) * cum_noise - self.gamma * (self.X - self.x[:-1])
        
        # 4. Calculate IS = C(x) - X*S_0 (Aligning with Definition 2.1 in theory.pdf)
        IS = np.dot(matrix_S_prev, self.v) - (self.eta / self.tau) * np.sum(self.v ** 2) - (self.X * self.S0)
        
        expectation = np.mean(IS)
        variance = np.var(IS, ddof=1)
        
        return expectation, variance



class Historical_Bootstrapper:
    """
    Empirical Monte Carlo engine for dynamic MPC execution.
    Samples continuous historical windows to evaluate the implementation shortfall
    driven by real microstructural features.
    """
    def __init__(self, df: pd.DataFrame, feature_matrix: pd.DataFrame, X: float, N: int, tau: float):
        self.df = df
        self.feature_matrix = feature_matrix
        self.X = X
        self.N = N
        self.tau = tau

    def run_bootstrap(
        self, 
        M: int, 
        dynamic_engine, 
        sigma_model, 
        eta_model, 
        static_v: np.ndarray, 
        eval_eta: float
    ) -> dict:
        
        valid_starts = np.arange(len(self.feature_matrix) - self.N - 1)
        start_indices = np.random.choice(valid_starts, size=M, replace=True)

        dynamic_IS = np.zeros(M)
        static_IS = np.zeros(M)

        for i, start_idx in enumerate(start_indices):
            window_features = self.feature_matrix.iloc[start_idx : start_idx + self.N]
            start_time = window_features.index[0]
            
            df_start_loc = self.df.index.get_loc(start_time)
            window_df = self.df.iloc[df_start_loc : df_start_loc + self.N + 1]
            
            S_0 = window_df['Close'].iloc[0]
            S_k_prev = window_df['Close'].iloc[:-1].values
            
            dyn_x = dynamic_engine.execute_dynamic_trajectory(
                feature_matrix=window_features, 
                sigma_model=sigma_model, 
                eta_model=eta_model
            )
            dyn_v = dyn_x[:-1] - dyn_x[1:]
            
            cash_static = np.dot(S_k_prev, static_v) - (eval_eta / self.tau) * np.sum(static_v ** 2)
            cash_dynamic = np.dot(S_k_prev, dyn_v) - (eval_eta / self.tau) * np.sum(dyn_v ** 2)
            
            static_IS[i] = cash_static - (self.X * S_0)
            dynamic_IS[i] = cash_dynamic - (self.X * S_0)

        return {
            "Static_IS": static_IS,
            "Dynamic_IS": dynamic_IS,
            "Metrics": {
                "Static": {"Expectation": np.mean(static_IS), "Variance": np.var(static_IS, ddof=1)},
                "Dynamic": {"Expectation": np.mean(dynamic_IS), "Variance": np.var(dynamic_IS, ddof=1)}
            }
        }