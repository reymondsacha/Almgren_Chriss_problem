import numpy as np
import pandas as pd
from typing import Tuple

def build_ml_dataset(
    df: pd.DataFrame, 
    engineer, 
    m_forward_periods: int,
    eta_baseline: float = 1.7e-9
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Constructs the strictly aligned X and y datasets for model training.
    
    Args:
        df: The raw OHLCV DataFrame from MicrostructureDownloader.
        engineer: An instantiated Market_State_Engineer object.
        m_forward_periods: The number of bars representing the forward execution interval tau.
        eta_baseline: L'impact temporaire historique moyen de base pour la calibration d'Amihud.
        
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
    
    # 2. Target Engineering (Forward y bruts)
    log_returns = np.log(df['Close'] / df['Close'].shift(1))
    rolling_variance = (log_returns ** 2).rolling(window=m_forward_periods).sum()
    
    forward_abs_return = np.abs(np.log(df['Close'].shift(-m_forward_periods) / df['Close']))
    rolling_volume = df['Volume'].rolling(window=m_forward_periods).sum()
    
    # 3. Calibration de kappa_eta
    # On calcule le ratio d'Amihud brut ramené à l'instant t
    raw_amihud = forward_abs_return / rolling_volume.shift(-m_forward_periods)
    raw_amihud_scaled = raw_amihud * df['Close']
    
    # Exclusion temporaire des zéros/infinis pour ne pas corrompre la moyenne de calibration
    amihud_for_mean = raw_amihud_scaled.replace([np.inf, 0, -np.inf], np.nan)
    kappa_eta = eta_baseline / amihud_for_mean.mean(skipna=True)
    
    # 4. Time Alignment & Scaling final
    y_sigma = np.sqrt(rolling_variance.shift(-m_forward_periods)) * df['Close']
    y_eta = raw_amihud_scaled * kappa_eta
    
    # 5. Nettoyage des asymptotes avant passage au log
    y_eta.replace([np.inf, 0, -np.inf], np.nan, inplace=True)
    y_sigma.replace([np.inf, 0, -np.inf], np.nan, inplace=True)
    
    # 6. Cleanup et transformation logarithmique
    dataset = X.copy()
    dataset['target_sigma'] = np.log(y_sigma)
    dataset['target_eta'] = np.log(y_eta)
    
    # Suppression des lignes avec NaN (période de chauffe initiale w, et fenêtres m incomplètes à la fin)
    dataset = dataset.dropna()
    
    X_clean = dataset[['garman_klass', 'ofi', 'momentum', 'depth']]
    y_sigma_clean = dataset['target_sigma']
    y_eta_clean = dataset['target_eta']
    
    return X_clean, y_sigma_clean, y_eta_clean