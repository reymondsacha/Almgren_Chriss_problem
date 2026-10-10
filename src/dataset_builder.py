import numpy as np
import pandas as pd
from typing import Tuple


def build_ml_dataset(
    df: pd.DataFrame, 
    engineer, 
    m_forward_periods: int
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Constructs the strictly aligned X and y datasets for model training.
    
    Args:
        df: The raw OHLCV DataFrame from MicrostructureDownloader.
        engineer: An instantiated Market_State_Engineer object.
        m_forward_periods: The number of bars representing the forward execution interval tau.
        
    Returns:
        X (pd.DataFrame): The feature matrix (w-period lookback).
        y_sigma (pd.Series): The target realized volatility (m-period lookforward).
        y_eta (pd.Series): The target realized illiquidity (m-period lookforward).
    """
    
    # 1. Feature Engineering (Trailing X)
    sigma_GK = engineer._compute_garman_klass(df['High'], df['Low'], df['Open'], df['Close'])
    ofi = engineer._compute_ofi_proxy(df['High'], df['Low'], df['Close'], df['Open'], df['Volume'])
    momentum = engineer._compute_momentum(df['Close'])
    depth = engineer._compute_depth(df['High'], df['Low'], df['Volume'])
    
    X = pd.DataFrame({
        'garman_klass': sigma_GK,
        'ofi': ofi,
        'momentum': momentum,
        'depth': depth
    }, index=df.index)
    
    # 2. Target Engineering (Forward y)
    
    # Target 1: Realized Volatility over the next m periods
    # We use the square root of the sum of squared log returns
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    rolling_variance = (log_returns ** 2).rolling(window=m_forward_periods).sum()
    
    # Target 2: Realized Illiquidity (Amihud proxy) over the next m periods
    # Absolute log return over the interval divided by total volume in the interval
    forward_abs_return = np.abs(np.log(df['Close'].shift(-m_forward_periods) / df['Close']))
    rolling_volume = df['Volume'].rolling(window=m_forward_periods).sum()
    
    # 3. Time Alignment
    # We shift the rolling targets backward by m periods so they align with the current state X_t
    y_sigma = (np.sqrt(rolling_variance)*df['Close']).shift(-m_forward_periods)
    y_eta = forward_abs_return / rolling_volume.shift(-m_forward_periods)
    
    # Replace any infinite values with NaN before dropping
    y_eta.replace([np.inf, 0, -np.inf], np.nan, inplace=True)
    
    # 4. Cleanup
    dataset = X.copy()
    dataset['target_sigma'] = np.log(y_sigma)
    dataset['target_eta'] = np.log(y_eta)
    
    # Drop rows with NaN (burn-in period at the start, unfulfilled forward windows at the end)
    dataset = dataset.dropna()
    
    X_clean = dataset[['garman_klass', 'ofi', 'momentum', 'depth']]
    y_sigma_clean = dataset['target_sigma']
    y_eta_clean = dataset['target_eta']
    
    return X_clean, y_sigma_clean, y_eta_clean