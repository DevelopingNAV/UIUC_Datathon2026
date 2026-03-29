import pandas as pd
import os
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

# Load raw CSVs, parse timestamps, handle DST, output clean parquet

def load_raw_data(data_dir: str = "data/raw") -> pd.DataFrame:
    """Load all portfolio CSV files and combine into single dataframe."""
    all_data = []
    data_path = Path(data_dir)

    for csv_file in data_path.glob("*.csv"):
        portfolio_id = csv_file.stem  # filename without extension
        df = pd.read_csv(csv_file)
        df['portfolio_id'] = portfolio_id
        all_data.append(df)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Ensure data types
    combined_df['date'] = pd.to_datetime(combined_df['date']).dt.date
    combined_df['CV'] = combined_df['CV'].astype(int)
    combined_df['CCT'] = combined_df['CCT'].astype(float)
    combined_df['ABD'] = combined_df['ABD'].astype(float)

    # Sort by portfolio, date, interval
    combined_df = combined_df.sort_values(['portfolio_id', 'date', 'interval'])

    return combined_df

def handle_dst_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """Handle daylight saving time adjustments."""
    import sys
    sys.path.append('src')
    from utils import parse_est_datetime

    # Add EST datetime column
    df = df.copy()
    df['datetime_est'] = df.apply(
        lambda row: parse_est_datetime(str(row['date']), row['interval']),
        axis=1
    )

    # Check for DST transitions and adjust if needed
    # For simplicity, we'll just ensure all intervals are valid EST times
    # In a real scenario, you might need to handle missing/skipped hours

    return df

def save_processed_data(df: pd.DataFrame, output_dir: str = "data/processed"):
    """Save processed data as parquet file."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / "processed_data.parquet"

    # Convert to arrow table for parquet
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_path)

    print(f"Processed data saved to {output_path}")

def main():
    """Main preprocessing pipeline."""
    print("Loading raw data...")
    df = load_raw_data()

    print("Handling DST adjustments...")
    df = handle_dst_adjustments(df)

    print("Saving processed data...")
    save_processed_data(df)

    print(f"Preprocessing complete. Shape: {df.shape}")

if __name__ == "__main__":
    main()