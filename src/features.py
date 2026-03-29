import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import sys
sys.path.append('src')
from utils import add_time_features
from typing import List

# Generate all time features and lag features

def load_processed_data(input_path: str = "data/processed/processed_data.parquet") -> pd.DataFrame:
    """Load processed parquet data."""
    return pq.read_table(input_path).to_pandas()

def add_lag_features(df: pd.DataFrame, target_cols: List[str] = ['CV', 'CCT', 'ABD']) -> pd.DataFrame:
    """Add lag features for same interval, previous weeks."""
    df = df.copy()

    # Sort by portfolio, datetime_est
    df = df.sort_values(['portfolio_id', 'datetime_est'])

    for col in target_cols:
        # Lag 7 days (same interval last week)
        df[f'{col}_lag_7d'] = df.groupby(['portfolio_id', 'interval'])[col].shift(1)  # 1 week = 1 shift since daily

        # Lag 14 days
        df[f'{col}_lag_14d'] = df.groupby(['portfolio_id', 'interval'])[col].shift(2)

        # Lag 28 days
        df[f'{col}_lag_28d'] = df.groupby(['portfolio_id', 'interval'])[col].shift(4)

        # 4-week rolling mean
        df[f'{col}_rolling_mean_4w'] = df.groupby(['portfolio_id', 'interval'])[col].transform(
            lambda x: x.rolling(window=4, min_periods=1).mean()
        )

    return df

def save_featured_data(df: pd.DataFrame, output_dir: str = "data/processed"):
    """Save data with features as parquet."""
    output_path = Path(output_dir) / "featured_data.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)
    print(f"Featured data saved to {output_path}")

def main():
    """Main feature engineering pipeline."""
    print("Loading processed data...")
    df = load_processed_data()

    print("Adding time features...")
    df = add_time_features(df)

    print("Adding lag features...")
    df = add_lag_features(df)

    # Fill NaN values in lag features (for early dates)
    lag_cols = [col for col in df.columns if 'lag' in col or 'rolling' in col]
    df[lag_cols] = df[lag_cols].fillna(method='bfill')  # backward fill for initial values

    print("Saving featured data...")
    save_featured_data(df)

    print(f"Feature engineering complete. Shape: {df.shape}")

if __name__ == "__main__":
    main()