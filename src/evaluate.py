import pandas as pd
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq
from typing import Dict

from sklearn.metrics import mean_absolute_percentage_error

def load_featured_data(input_path: str = "data/processed/featured_data.parquet") -> pd.DataFrame:
    """Load featured data."""
    return pq.read_table(input_path).to_pandas()

def calculate_asymmetric_score(y_true: pd.Series, y_pred: pd.Series, under_penalty: float = 2.0) -> float:
    """Calculate asymmetric score where under-forecasting costs more."""
    errors = y_pred - y_true
    over_forecast_penalty = np.maximum(errors, 0)  # Over-forecasting penalty
    under_forecast_penalty = np.maximum(-errors, 0) * under_penalty  # Under-forecasting penalty (higher weight)

    total_penalty = np.mean(over_forecast_penalty + under_forecast_penalty)
    return total_penalty

def evaluate_portfolio(df: pd.DataFrame, portfolio_id: str, val_periods: int = 14) -> Dict[str, float]:
    """Evaluate models for a portfolio on validation set."""
    portfolio_df = df[df['portfolio_id'] == portfolio_id].copy()

    # Use last val_periods intervals for validation
    val_df = portfolio_df.tail(val_periods)
    train_df = portfolio_df.iloc[:-val_periods]

    scores = {}

    target_cols = ['CV', 'CCT', 'ABD']

    for target in target_cols:
        y_true = val_df[target]

        # Simple baseline: use last known value
        baseline_pred = train_df[target].iloc[-1]
        baseline_pred_series = pd.Series([baseline_pred] * len(y_true))

        # Calculate MAPE
        mape = mean_absolute_percentage_error(y_true, baseline_pred_series)

        # Calculate asymmetric score
        asym_score = calculate_asymmetric_score(y_true, baseline_pred_series)

        scores[target] = {
            'MAPE': mape,
            'asymmetric_score': asym_score
        }

    # Composite score: weighted average of MAPE
    # CV gets highest weight due to understaffing penalty
    weights = {'CV': 0.5, 'CCT': 0.3, 'ABD': 0.2}
    composite_score = sum(scores[target]['MAPE'] * weights[target] for target in target_cols)

    scores['composite'] = composite_score

    return scores

def main():
    """Main evaluation pipeline."""
    print("Loading featured data...")
    df = load_featured_data()

    portfolios = df['portfolio_id'].unique()

    all_scores = {}

    for portfolio in portfolios:
        print(f"Evaluating {portfolio}...")
        scores = evaluate_portfolio(df, portfolio)
        all_scores[portfolio] = scores

        print(f"  CV MAPE: {scores['CV']['MAPE']:.4f}")
        print(f"  CCT MAPE: {scores['CCT']['MAPE']:.4f}")
        print(f"  ABD MAPE: {scores['ABD']['MAPE']:.4f}")
        print(f"  Composite Score: {scores['composite']:.4f}")

    # Overall leaderboard
    leaderboard = sorted(all_scores.items(), key=lambda x: x[1]['composite'])
    print("\nLeaderboard (lower score is better):")
    for rank, (portfolio, scores) in enumerate(leaderboard, 1):
        print(f"{rank}. {portfolio}: {scores['composite']:.4f}")

if __name__ == "__main__":
    main()