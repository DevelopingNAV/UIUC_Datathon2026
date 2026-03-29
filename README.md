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
[Add MAPE / composite score on validation set here]

## Video
[Link to 7-minute walkthrough video]
