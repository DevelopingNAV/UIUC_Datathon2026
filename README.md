# Synchrony Datathon 2026 — Team Naveed

## Problem
Predict intraday call center volume (CV, CCT, ABD) at 30-minute intervals for 4 Synchrony
portfolios across a full month.

## Setup
All development done inside the provided Docker environment (Classical ML + Boosting image).
```bash
pip install -r requirements.txt  # all packages pre-installed in Docker
```

## How to Run
```bash
# Full pipeline (preprocess → features → train → predict)
python run_pipeline.py --portfolio all --output outputs/forecast_v01.csv

# Individual steps
python src/preprocess.py
python src/features.py
python src/train.py --portfolio portfolio_1
python src/predict.py --output outputs/forecast_v01.csv
```

## Approach
- Prophet model per portfolio for baseline
- XGBoost with engineered time/lag features for iteration
- Asymmetric scoring addressed by slight upward bias on CV
- Experiment tracking via MLFlow

## Results
### Validation Performance
- **Portfolio 3**: Composite Score = 0.1725 (Best performing)
- **Portfolio 4**: Composite Score = 0.1742
- **Portfolio 2**: Composite Score = 0.1750
- **Portfolio 1**: Composite Score = 0.1761

### Individual Metrics (MAPE)
- **Call Volume (CV)**: ~12% MAPE across portfolios
- **Customer Care Time (CCT)**: ~11% MAPE across portfolios
- **Abandon Rate (ABD)**: ~41% MAPE across portfolios

### Model Performance
- Prophet baseline models trained for each portfolio and metric
- All predictions clipped to non-negative values
- Forecast generated for August 2023 (31 days × 48 intervals × 4 portfolios)
