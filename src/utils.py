import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
from typing import List, Tuple

# Shared helpers (date utils, interval expansion, DST lookup)

EST = pytz.timezone('US/Eastern')

def parse_est_datetime(date_str: str, interval_str: str) -> datetime:
    """Parse date and interval strings into EST datetime object."""
    dt_naive = datetime.strptime(f"{date_str} {interval_str}", "%Y-%m-%d %H:%M")
    dt_est = EST.localize(dt_naive)
    return dt_est

def get_business_days(start_date: str, end_date: str) -> List[str]:
    """Get list of business days (weekdays) between start and end dates."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # Monday=0 to Friday=4
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates

def expand_intervals(dates: List[str]) -> pd.DataFrame:
    """Expand dates into 30-minute intervals from 00:00 to 23:30 EST."""
    intervals = []
    for date in dates:
        for hour in range(24):  # 00:00 to 23:00
            for minute in [0, 30]:
                interval = f"{hour:02d}:{minute:02d}"
                intervals.append({
                    'date': date,
                    'interval': interval,
                    'datetime_est': parse_est_datetime(date, interval)
                })
    return pd.DataFrame(intervals)

def is_dst_transition(dt: datetime) -> bool:
    """Check if datetime is during DST transition."""
    # Check if it's the hour that would be skipped or repeated
    try:
        dt.astimezone(EST)
        return False
    except:
        return True

def get_holidays(year: int) -> List[str]:
    """Get US federal holidays for a given year."""
    # Simplified list of major holidays
    holidays = [
        f"{year}-01-01",  # New Year's Day
        f"{year}-01-16",  # MLK Day (approximate)
        f"{year}-02-20",  # Presidents Day (approximate)
        f"{year}-05-29",  # Memorial Day (approximate)
        f"{year}-07-04",  # Independence Day
        f"{year}-09-04",  # Labor Day (approximate)
        f"{year}-11-11",  # Veterans Day
        f"{year}-11-23",  # Thanksgiving (approximate)
        f"{year}-12-25",  # Christmas
    ]
    return holidays

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic time features to dataframe."""
    df = df.copy()
    df['datetime_est'] = df.apply(lambda row: parse_est_datetime(row['date'], row['interval']), axis=1)
    df['hour_of_day'] = df['datetime_est'].dt.hour
    df['day_of_week'] = df['datetime_est'].dt.weekday  # 0=Monday
    df['day_of_month'] = df['datetime_est'].dt.day
    df['week_of_month'] = ((df['day_of_month'] - 1) // 7) + 1
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_holiday'] = df['date'].isin(get_holidays(df['datetime_est'].dt.year.iloc[0])).astype(int)
    df['is_dst_transition'] = df['datetime_est'].apply(is_dst_transition).astype(int)
    return df