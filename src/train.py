import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
import joblib
from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error
from typing import Dict, Any, List, Tuple

# Train model per portfolio, save to outputs/models/

def load_featured_data(input_path: str = "data/processed/featured_data.parquet") -> pd.DataFrame:
    """Load featured data."""
    return pq.read_table(input_path).to_pandas()

def prepare_prophet_data(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Prepare data for Prophet model."""
    prophet_df = df[['datetime_est', target_col]].copy()
    prophet_df.columns = ['ds', 'y']
    # Convert timezone-aware to naive datetime
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    prophet_df = prophet_df.dropna()
    return prophet_df

def train_prophet_model(train_df: pd.DataFrame) -> Prophet:
    """Train Prophet model."""
    model = Prophet(
        yearly_seasonality=False,  # No yearly data
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(train_df)
    return model

def evaluate_model(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {'MAPE': mape}

def train_portfolio_model(df: pd.DataFrame, portfolio_id: str, target_col: str) -> Dict[str, Any]:
    """Train models for a specific portfolio and target."""
    portfolio_df = df[df['portfolio_id'] == portfolio_id].copy()

    # Split train/validation (use last 2 weeks for validation)
    train_df = portfolio_df.iloc[:-14]  # All except last 14 intervals
    val_df = portfolio_df.iloc[-14:]    # Last 14 intervals

    results = {}

    # Prophet model
    prophet_train = prepare_prophet_data(train_df, target_col)
    prophet_model = train_prophet_model(prophet_train)

    # Predict on validation
    val_pred_df = val_df[['datetime_est']].copy()
    val_pred_df['ds'] = val_pred_df['datetime_est'].dt.tz_localize(None)
    prophet_val_pred = prophet_model.predict(val_pred_df[['ds']])['yhat']
    prophet_metrics = evaluate_model(val_df[target_col], prophet_val_pred)

    results['prophet'] = {
        'model': prophet_model,
        'metrics': prophet_metrics
    }

    return results

def save_models(models_dict: Dict[str, Any], portfolio_id: str, target_col: str, output_dir: str = "outputs/models"):
    """Save trained models."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for model_name, model_data in models_dict.items():
        model_path = Path(output_dir) / f"{portfolio_id}_{target_col}_{model_name}.joblib"
        joblib.dump(model_data['model'], model_path)
        print(f"Saved {model_name} model to {model_path}")

def main():
    """Main training pipeline."""
    print("Loading featured data...")
    df = load_featured_data()

    target_cols = ['CV', 'CCT', 'ABD']
    portfolios = df['portfolio_id'].unique()

    # Start experiment (simplified without MLflow)
    print("Starting training experiment...")

    for portfolio in portfolios:
        print(f"\nTraining models for {portfolio}...")

        for target in target_cols:
            print(f"  Training {target}...")

            models = train_portfolio_model(df, portfolio, target)

            # Choose best model (only prophet for now)
            best_model_name = 'prophet'
            best_model = models[best_model_name]['model']

            # Save models locally
            save_models(models, portfolio, target)

    print("Training complete!")

if __name__ == "__main__":
    main()