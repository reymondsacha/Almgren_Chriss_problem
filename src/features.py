import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

class Market_State_Engineer:
    """
    We aim to construct the Matrix X needed for prediction of model parameters.
    """

    def __init__(self, window: int):
        self.w = window

    def _compute_garman_klass(self, high: pd.Series, low: pd.Series, open_: pd.Series, close: pd.Series) -> pd.Series:
        """
        We compute the variance using the Garman Klass estimator over a window w.
        """
        
        instantaneous_gk = 0.5 * (np.log(high/low)**2) - (2*np.log(2) - 1)*((np.log(close/open_)**2))
        sigma_GK = instantaneous_gk.rolling(window=self.w).mean()

        return sigma_GK

    def _compute_ofi_proxy(self, high: pd.Series, low: pd.Series, close: pd.Series, open_: pd.Series, volume: pd.Series) -> pd.Series:
        """
        We compute the rolling Order FLow Imbalance proxy over a window w.
        We handle H == L cases with α = 0
        """

        alpha_array = np.where(high == low, 0, ((close - open_) + (close - low) - (high - close)) / (2 * (high - low)))

        alpha = pd.Series(alpha_array, index=close.index)
        NBI = alpha * volume

        OFI = NBI.rolling(window=self.w).sum()

        return OFI


    def _compute_momentum(self, close: pd.Series) -> pd.Series:
        """
        We compute the rolling localized micro-momentum (Z-score) over window w.
        """

        mu = close.rolling(window=self.w).mean()

        sigma = close.rolling(window=self.w).std()

        M = (close - mu) / sigma
        
        return M

    def _compute_depth(self, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """
        We compute a proxy for market depth using range-based illiquidity.
        """

        instantaneous_depth_array = np.where(volume == 0, 0.0, np.log(high/low) / volume)
        instantaneous_depth = pd.Series(instantaneous_depth_array, index=volume.index)
        depth = instantaneous_depth.rolling(window=self.w).mean()

        return depth


class Execution_Parameter_Regressor:
    """
    We aim here to compute σ and η using a Ridge regression model.
    """
    def __init__(self, alphas: np.ndarray, n_splits: int=5):
        self.alphas = alphas
        self.n_splits = n_splits
        self.scaler = StandardScaler()
        self.best_alpha = None
        self.beta = None
        self.y_mean = 0

    def _solve_ridge(self, X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
        """
        We compute the analytical closed-form solution beta = (X^T X + alpha I)^(-1) X^T y
        """

        n_features = X.shape[1]
        I = np.eye(n_features)

        A = np.dot(X.T,X) + alpha * I
        b = np.dot(X.T, y)

        # we solve the linear system A * beta = b
        # A is strictly definite positive, indeed X.T X and is similar to a diagonal matrix with non negative eigenvalues di so
        # it's similar to a diagonal matrix whose coefficients equal di + alpha (alpha > 0)

        beta = scipy.linalg.solve(A, b, assume_a='pos')
        
        return beta



    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Executes the walk-forward grid search to isolate the optimal L2 penalty,
        then fits the final weights on the full temporal dataset.
        """
        X_arr = X.values
        y_arr = y.values
        
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        best_mse = float('inf')
        
        # Empirical Risk Minimization via hyperparameter tuning
        for alpha in self.alphas:
            fold_mses = []
            
            for train_idx, val_idx in tscv.split(X_arr):
                X_train, X_val = X_arr[train_idx], X_arr[val_idx]
                y_train, y_val = y_arr[train_idx], y_arr[val_idx]
                
                # Fit scaler strictly on the training fold to prevent look-ahead bias
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_val_scaled = self.scaler.transform(X_val)
                
                # Center the target variable so the intercept is not regularized
                y_train_mean = np.mean(y_train)
                y_train_centered = y_train - y_train_mean
                
                # Fit beta on the training fold
                beta_fold = self._solve_ridge(X_train_scaled, y_train_centered, alpha)
                
                # Predict on the strictly out-of-sample validation fold
                y_pred = np.dot(X_val_scaled, beta_fold) + y_train_mean
                fold_mses.append(mean_squared_error(y_val, y_pred))
                
            mean_cv_mse = np.mean(fold_mses)
            
            if mean_cv_mse < best_mse:
                best_mse = mean_cv_mse
                self.best_alpha = alpha
                
        # Final fit on the entire historical dataset using the optimized penalty
        X_scaled = self.scaler.fit_transform(X_arr)
        self.y_mean = np.mean(y_arr)
        y_centered = y_arr - self.y_mean
        
        self.beta = self._solve_ridge(X_scaled, y_centered, self.best_alpha)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generates predictions using the fitted Ridge weights and scaler.
        """
        if self.beta is None:
            raise ValueError("Model must be fitted before calling predict.")
            
        # Ensure we are working with a NumPy array
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        
        # Scale the new features using the parameters learned during fit()
        X_scaled = self.scaler.transform(X_arr)
        
        # Apply the linear model and re-add the target mean
        y_pred = np.dot(X_scaled, self.beta) + self.y_mean
        
        return y_pred








