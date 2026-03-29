import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_portfolio_data(portfolio_id, scale_factor=1.0):
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 3, 31)
    dates = []
    current = start_date
    while current <= end_date:
        if current.weekday() < 5:  # Monday to Friday
            dates.append(current)
        current += timedelta(days=1)

    intervals = ['09:00', '09:30', '10:00', '10:30', '11:00', '11:30', '12:00', '12:30', '13:00', '13:30', '14:00', '14:30', '15:00', '15:30', '16:00', '16:30', '17:00']

    data = []
    np.random.seed(42)  # for reproducibility
    for date in dates:
        for interval in intervals:
            hour = int(interval.split(':')[0])
            base_cv = scale_factor * (100 + 50 * np.sin((hour - 9) * np.pi / 8))
            if date.weekday() == 0:  # Monday spike
                base_cv *= 1.2
            cv = max(0, int(base_cv + np.random.normal(0, 10)))
            cct = max(0, 300 + np.random.normal(0, 50))
            abd = min(1, max(0, 0.05 + np.random.normal(0, 0.02)))
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'interval': interval,
                'CV': cv,
                'CCT': cct,
                'ABD': abd
            })

    df = pd.DataFrame(data)
    df.to_csv(f'data/raw/{portfolio_id}.csv', index=False)
    print(f'Generated {portfolio_id}.csv')

# Generate for all portfolios with different scales
generate_portfolio_data('portfolio_1', 1.0)
generate_portfolio_data('portfolio_2', 1.5)  # larger portfolio
generate_portfolio_data('portfolio_3', 0.8)  # smaller
generate_portfolio_data('portfolio_4', 1.2)