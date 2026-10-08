import numpy as np
import pandas as pd

from derivation import DiscreteAlmgrenChriss  

class DynamicAlmgrenChriss:
    """
    Model Predictive Control (MPC) execution engine.
    Dynamically recalculates the optimal execution trajectory at each discrete step 
    by re-instantiating the DiscreteAlmgrenChriss solver with ML predictions.
    """
    def __init__(self, X: float, T: float, N: int, gamma: float, lambda_penalty: float):
        self.X_initial = X
        self.T = T
        self.N = N
        self.gamma = gamma
        self.lambda_penalty = lambda_penalty
        self.tau = T / N

    def execute_dynamic_trajectory(
        self, 
        feature_matrix: pd.DataFrame, 
        sigma_model, 
        eta_model
    ) -> np.ndarray:
        
        trajectory = np.zeros(self.N + 1)
        trajectory[0] = self.X_initial
        tau = self.T / self.N
        
        for tau_idx in range(self.N):
            current_state = feature_matrix.iloc[[tau_idx]]
            
            # Pure ML Predictions
            sigma_pred = np.exp(sigma_model.predict(current_state)[0])
            eta_pred = np.exp(eta_model.predict(current_state)[0])
            
            # Safety check to ensure positive definite matrix (though gamma calibration should prevent this)
            adjusted_eta = eta_pred - 0.5 * self.gamma * tau
            if adjusted_eta <= 0:
                eta_pred = (0.5 * self.gamma * tau) + 1e-18
            
            remaining_steps = self.N - tau_idx
            
            if remaining_steps == 1:
                trajectory[tau_idx + 1] = 0.0
                continue
            
            sub_model = DiscreteAlmgrenChriss(
                X=trajectory[tau_idx],
                T=self.T - (tau_idx * tau),
                N=remaining_steps,
                sigma=sigma_pred,
                eta=eta_pred,
                gamma=self.gamma,
                lambda_penalty=self.lambda_penalty
            )
            
            x_opt = sub_model.solve_trajectory()
            trajectory[tau_idx + 1] = x_opt[1]
            
        return trajectory