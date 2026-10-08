import pandas as pd
import yfinance as yf
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class MicrostructureDownloader:
    """
    Fetches high-frequency intraday data for a single asset, 
    with local Parquet caching to prevent redundant API calls.
    """
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_intraday(self, ticker: str, start_date: str, end_date: str, interval: str = "5m") -> pd.DataFrame:
        file_path = self.data_dir / f"{ticker}_{interval}_{start_date}_{end_date}.parquet"
        
        # 1. Check local cache
        if file_path.exists():
            logger.info(f"Loading {ticker} {interval} data from local cache.")
            return pd.read_parquet(file_path, engine="pyarrow")
            
        # 2. Fetch from API if not cached
        logger.info(f"Fetching {ticker} {interval} data from Yahoo Finance.")
        try:
            tk = yf.Ticker(ticker)
            df = tk.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No {interval} data returned for {ticker}.")
                return pd.DataFrame()
                
            expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in expected_cols if col in df.columns]
            df = df[available_cols]
            
            # 3. Persist to disk for future runs
            df.to_parquet(file_path, engine="pyarrow", compression="snappy")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch intraday data for {ticker}: {e}")
            raise RuntimeError(f"Microstructure fetch failed: {e}") from e