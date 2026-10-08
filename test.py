import logging
from src.data_downloader import MicrostructureDownloader  # Adjust import based on your filename

# 1. Force the logger to print to your terminal so we can see the hidden warnings
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

print("Initializing downloader...")
downloader = MicrostructureDownloader(data_dir="data/raw")

# 2. Use a strictly recent date range (within the last 10 days) to bypass the 60-day limit
ticker = "AAPL"
start = "2026-09-20"
end = "2026-09-27"

print(f"Fetching {ticker} from {start} to {end}...")
df = downloader.fetch_intraday(ticker=ticker, start_date=start, end_date=end, interval="5m")

if df.empty:
    print("FAILED: DataFrame is empty. The file was not saved.")
else:
    print(f"SUCCESS: Downloaded {len(df)} rows.")
    print(df.head())