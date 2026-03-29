import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime, timedelta
import sys
sys.path.append('src')
from utils import expand_intervals, add_time_features
import pyarrow.parquet as pq
from prophet import Prophet
from typing import Any, List

# Load models, generate 30-min interval forecasts, clip negatives

def load_featured_data(input_path: str = "data/processed/featured_data.parquet") -> pd.DataFrame:
    """Load featured data for lag features."""
    return pq.read_table(input_path).to_pandas()

def load_model(portfolio_id: str, target_col: str, model_name: str = "prophet", model_dir: str = "outputs/models") -> Any:
    """Load trained model."""
    model_path = Path(model_dir) / f"{portfolio_id}_{target_col}_{model_name}.joblib"
    return joblib.load(model_path)

def generate_forecast_dates(target_month: str = "2023-08") -> List[str]:
    """Generate all days for target month."""
    year, month = map(int, target_month.split('-'))
    import calendar
    num_days = calendar.monthrange(year, month)[1]
    return [f"{year}-{month:02d}-{day:02d}" for day in range(1, num_days + 1)]

def predict_prophet(model: Prophet, forecast_df: pd.DataFrame) -> pd.Series:
    """Generate predictions using Prophet model."""
    prophet_input = forecast_df[['datetime_est']].rename(columns={'datetime_est': 'ds'})
    prophet_input['ds'] = prophet_input['ds'].dt.tz_localize(None)  # Remove timezone
    forecast = model.predict(prophet_input)
    return forecast['yhat']

def add_lag_features_for_forecast(forecast_df: pd.DataFrame, historical_df: pd.DataFrame,
                                portfolio_id: str, target_col: str) -> pd.DataFrame:
    """Add lag features to forecast dataframe using historical data."""
    df = forecast_df.copy()

    # Get historical data for this portfolio
    hist_portfolio = historical_df[historical_df['portfolio_id'] == portfolio_id].copy()

    for i, row in df.iterrows():
        interval = row['interval']

        # Find matching historical intervals (same interval, previous weeks)
        matching_hist = hist_portfolio[hist_portfolio['interval'] == interval]

        if len(matching_hist) >= 4:
            # Lag 7d: most recent
            df.at[i, f'{target_col}_lag_7d'] = matching_hist[target_col].iloc[-1]
            # Lag 14d: 2nd most recent
            df.at[i, f'{target_col}_lag_14d'] = matching_hist[target_col].iloc[-2]
            # Lag 28d: 4th most recent
            df.at[i, f'{target_col}_lag_28d'] = matching_hist[target_col].iloc[-4]

            # Rolling mean 4w
            recent_values = matching_hist[target_col].tail(4)
            df.at[i, f'{target_col}_rolling_mean_4w'] = recent_values.mean()

    return df

def generate_portfolio_forecast(portfolio_id: str, target_month: str, historical_df: pd.DataFrame) -> pd.DataFrame:
    """Generate forecast for a single portfolio."""
    # Generate forecast dates and intervals
    dates = generate_forecast_dates(target_month)
    forecast_base = expand_intervals(dates)
    forecast_base['portfolio_id'] = portfolio_id

    # Add time features
    forecast_df = add_time_features(forecast_base)

    target_cols = ['CV', 'CCT', 'ABD']
    predictions = {}

    for target in target_cols:
        # Use Prophet
        model = load_model(portfolio_id, target, "prophet")
        pred = predict_prophet(model, forecast_df)

        # Clip predictions to >= 0
        pred = np.maximum(pred, 0)
        predictions[target] = pred

    # Combine predictions
    result_df = forecast_df[['date', 'interval', 'portfolio_id']].copy()
    result_df['CV'] = predictions['CV']
    result_df['CCT'] = predictions['CCT']
    result_df['ABD'] = predictions['ABD']
    result_df['abandoned_calls'] = result_df['ABD'] * result_df['CV']

    return result_df

def generate_template_forecast(target_month: str = "2023-08", output_path: str = "outputs/forecast_v01.csv"):
    """Generate forecast in template format."""
    print("Loading historical data...")
    historical_df = load_featured_data()

    dates = generate_forecast_dates(target_month)
    month_name = "August"  # Hardcoded for now

    # Generate base template with all intervals (24 hours)
    intervals_24h = []
    for hour in range(24):
        for minute in [0, 30]:
            intervals_24h.append(f"{hour:02d}:{minute:02d}")

    # Create template dataframe
    template_rows = []
    for date in dates:
        day = int(date.split('-')[2])
        for interval in intervals_24h:
            row = {
                'Month': month_name,
                'Day': day,
                'Interval': interval
            }
            # Initialize portfolio columns as empty
            for portfolio in ['A', 'B', 'C', 'D']:
                row[f'Calls_Offered_{portfolio}'] = ''
                row[f'Abandoned_Calls_{portfolio}'] = ''
                row[f'Abandoned_Rate_{portfolio}'] = ''
                row[f'CCT_{portfolio}'] = ''
            template_rows.append(row)

    template_df = pd.DataFrame(template_rows)

    # Map portfolio names
    portfolio_mapping = {
        'portfolio_1': 'A',
        'portfolio_2': 'B',
        'portfolio_3': 'C',
        'portfolio_4': 'D'
    }

    portfolios = historical_df['portfolio_id'].unique()

    for portfolio in portfolios:
        print(f"Generating forecast for {portfolio}...")
        portfolio_forecast = generate_portfolio_forecast(portfolio, target_month, historical_df)

        portfolio_letter = portfolio_mapping[portfolio]

        # Fill in the template
        for _, forecast_row in portfolio_forecast.iterrows():
            date_obj = pd.to_datetime(forecast_row['date'])
            day = date_obj.day
            interval = forecast_row['interval']

            # Find matching row in template
            mask = (template_df['Day'] == day) & (template_df['Interval'] == interval)
            if mask.any():
                template_df.loc[mask, f'Calls_Offered_{portfolio_letter}'] = forecast_row['CV']
                template_df.loc[mask, f'Abandoned_Calls_{portfolio_letter}'] = forecast_row['abandoned_calls']
                template_df.loc[mask, f'Abandoned_Rate_{portfolio_letter}'] = forecast_row['ABD']
                template_df.loc[mask, f'CCT_{portfolio_letter}'] = forecast_row['CCT']

    print("Saving forecast in template format...")
    template_df.to_csv(output_path, index=False)
    print(f"Forecast saved to {output_path}")

def main(target_month: str = "2023-08", output_path: str = "outputs/forecast_v01.csv"):
    """Main prediction pipeline."""
    generate_template_forecast(target_month, output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_month", default="2023-08")
    parser.add_argument("--output", default="outputs/forecast_v01.csv")
    args = parser.parse_args()
    main(args.target_month, args.output)