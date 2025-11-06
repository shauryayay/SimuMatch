# backend/recommend.py
import pandas as pd
import numpy as np
from pathlib import Path

EVENT_PROFILES = {
    '5K':      {'endurance':0.2, 'speed':0.9, 'week_km':10, 'type':'run'},
    '10K':     {'endurance':0.35,'speed':0.8, 'week_km':20, 'type':'run'},
    'Half':    {'endurance':0.7, 'speed':0.6, 'week_km':30, 'type':'run'},
    'Marathon':{'endurance':0.95,'speed':0.4, 'week_km':50, 'type':'run'},
    'Tri_Sprint':{'endurance':0.4,'speed':0.6,'week_km':10,'type':'tri'},
    'Tri_Olympic':{'endurance':0.7,'speed':0.6,'week_km':25,'type':'tri'}
}

def load_events_csv(path="data/processed/events.csv"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError("Processed events not found. Run preprocess_events.py")
    return pd.read_csv(p, parse_dates=['date'])

def athlete_summary(events_df, window_days=28):
    # events_df has date, daily_distance_km, daily_avg_speed_kph, readiness, sleep_hours, mood_score
    events_df['date'] = pd.to_datetime(events_df['date'])
    max_date = events_df['date'].max()
    cutoff = max_date - pd.Timedelta(days=window_days)
    recent = events_df[events_df['date'] >= cutoff]

    weekly_km = (recent['daily_distance_km'].sum() / (window_days/7.0)) if 'daily_distance_km' in recent.columns else 0.0
    long_run_max = recent['daily_distance_km'].max() if 'daily_distance_km' in recent.columns else 0.0
    avg_pace_min_per_km = None
    if 'daily_avg_speed_kph' in recent.columns and recent['daily_avg_speed_kph'].mean() > 0:
        avg_speed = recent['daily_avg_speed_kph'].mean()
        avg_pace_min_per_km = 60.0 / avg_speed
    avg_readiness = recent['readiness'].mean() if 'readiness' in recent.columns else np.nan
    avg_sleep = recent['sleep_hours'].mean() if 'sleep_hours' in recent.columns else np.nan

    return {
        'weekly_km': float(weekly_km),
        'long_run_max': float(long_run_max) if not np.isnan(long_run_max) else 0.0,
        'avg_pace_min_per_km': float(avg_pace_min_per_km) if avg_pace_min_per_km is not None else np.nan,
        'avg_readiness': float(avg_readiness) if not np.isnan(avg_readiness) else np.nan,
        'avg_sleep_hours': float(avg_sleep) if not np.isnan(avg_sleep) else np.nan
    }

def speed_score(pace_min_per_km):
    if pd.isna(pace_min_per_km) or pace_min_per_km <= 0:
        return 0.0
    return max(0.0, 1.0 - (pace_min_per_km / 6.0))  # baseline at 6 min/km

def score_event(summary, profile):
    wk = summary.get('weekly_km', 0.0) or 0.0
    readiness = summary.get('avg_readiness', 50.0) or 50.0
    pace = summary.get('avg_pace_min_per_km', np.nan)

    endurance_match = min(1.0, wk / profile.get('week_km', 20))
    sp = speed_score(pace)
    recovery = min(1.0, readiness / 85.0)

    combined = 0.5 * endurance_match * profile.get('endurance', 0.6) + \
               0.4 * sp * profile.get('speed', 0.5) + \
               0.1 * recovery
    return {
        'combined': float(combined),
        'endurance_match': float(endurance_match),
        'speed_score': float(sp),
        'recovery': float(recovery)
    }

def recommend(summary, topk=5):
    scored = []
    for name, prof in EVENT_PROFILES.items():
        s = score_event(summary, prof)
        s['event'] = name
        scored.append(s)
    scored = sorted(scored, key=lambda x: x['combined'], reverse=True)
    return scored[:topk]
