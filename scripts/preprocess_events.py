# scripts/preprocess_events.py
"""
Preprocess raw CSV exports (Strava + Oura pieces) into a single per-day events.csv
This version expects the column names you provided:
- oura_readiness_rows.csv: ['id','user_id','record_id','score','contributors','day','created_at','updated_at']
- strava_activities_rows.csv: ['id','user_id','strava_id','name','type','distance','moving_time','total_elevation_gain',
    'start_date','start_latlng','average_speed','max_speed','average_heartrate','max_heartrate','average_cadence',
    'average_watts','kilojoules','suffer_score','created_at','updated_at']
- oura_activity_rows.csv: ['id','user_id','record_id','steps','calories','active_calories','active_time','day','created_at','updated_at']
- oura_heartrate_rows.csv: ['id','user_id','record_id','day','created_at','updated_at','average_hr','min_hr','max_hr']
- oura_sleep_rows.csv: ['id','user_id','sleep_id','start_datetime','end_datetime','duration','score','created_at','updated_at']
"""

import pandas as pd
from pathlib import Path
import numpy as np

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def load_if_exists(filename):
    p = RAW / filename
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return None

def normalize_strava(df):
    """Return dataframe with columns: date, daily_distance_km, daily_avg_speed_kph"""
    if df is None:
        return None
    tmp = df.copy()
    # parse start_date (assume column 'start_date' exists)
    if 'start_date' in tmp.columns:
        tmp['date'] = pd.to_datetime(tmp['start_date'], errors='coerce').dt.date
    else:
        tmp['date'] = pd.to_datetime(tmp.iloc[:,0], errors='coerce').dt.date

    # distance column (Strava export likely meters)
    if 'distance' in tmp.columns:
        # if values > 1000 assume meters
        if tmp['distance'].max(skipna=True) is not None and tmp['distance'].max(skipna=True) > 1000:
            tmp['distance_km'] = tmp['distance'] / 1000.0
        else:
            tmp['distance_km'] = tmp['distance']
    else:
        tmp['distance_km'] = 0.0

    # average_speed: Strava uses m/s often; convert to kph
    if 'average_speed' in tmp.columns:
        # assume average_speed in m/s if typical values < 20; convert to kph
        # if values appear already >20, treat as kph
        mean_speed = tmp['average_speed'].dropna().mean() if tmp['average_speed'].dropna().size>0 else 0
        if mean_speed > 0 and mean_speed < 20:
            tmp['avg_speed_kph'] = tmp['average_speed'] * 3.6
        else:
            tmp['avg_speed_kph'] = tmp['average_speed']
    else:
        tmp['avg_speed_kph'] = np.nan

    # Aggregate per day:
    agg = tmp.groupby('date').agg({
        'distance_km':'sum',
        'avg_speed_kph':'mean'
    }).rename(columns={'distance_km':'daily_distance_km','avg_speed_kph':'daily_avg_speed_kph'}).reset_index()
    return agg

def normalize_oura_readiness(df):
    """Return dataframe with columns: date, readiness (score)"""
    if df is None:
        return None
    tmp = df.copy()
    # you told me readiness file has 'day' and 'score'
    if 'day' in tmp.columns:
        tmp['date'] = pd.to_datetime(tmp['day'], errors='coerce').dt.date
    elif 'created_at' in tmp.columns:
        tmp['date'] = pd.to_datetime(tmp['created_at'], errors='coerce').dt.date
    else:
        tmp['date'] = pd.to_datetime(tmp.iloc[:,0], errors='coerce').dt.date

    if 'score' in tmp.columns:
        tmp['readiness'] = tmp['score']
    else:
        # fallback: try common alternatives
        fallback = next((c for c in ['readiness_score','value'] if c in tmp.columns), None)
        tmp['readiness'] = tmp[fallback] if fallback else np.nan

    out = tmp[['date','readiness']].copy()
    return out

def normalize_oura_activity(df):
    """Return daily steps / calories from Oura activity rows"""
    if df is None:
        return None
    tmp = df.copy()
    if 'day' in tmp.columns:
        tmp['date'] = pd.to_datetime(tmp['day'], errors='coerce').dt.date
    elif 'created_at' in tmp.columns:
        tmp['date'] = pd.to_datetime(tmp['created_at'], errors='coerce').dt.date
    else:
        tmp['date'] = pd.to_datetime(tmp.iloc[:,0], errors='coerce').dt.date

    # choose columns if they exist
    tmp['steps'] = tmp['steps'] if 'steps' in tmp.columns else np.nan
    tmp['calories'] = tmp['calories'] if 'calories' in tmp.columns else np.nan
    agg = tmp.groupby('date').agg({'steps':'sum','calories':'sum'}).reset_index()
    return agg

def normalize_oura_heartrate(df):
    """Return daily avg_hr / min_hr / max_hr"""
    if df is None:
        return None
    tmp = df.copy()
    if 'day' in tmp.columns:
        tmp['date'] = pd.to_datetime(tmp['day'], errors='coerce').dt.date
    else:
        tmp['date'] = pd.to_datetime(tmp.iloc[:,0], errors='coerce').dt.date

    cols = {}
    cols['average_hr'] = 'average_hr' if 'average_hr' in tmp.columns else ( 'avg_hr' if 'avg_hr' in tmp.columns else np.nan )
    cols['min_hr'] = 'min_hr' if 'min_hr' in tmp.columns else np.nan
    cols['max_hr'] = 'max_hr' if 'max_hr' in tmp.columns else np.nan

    # ensure columns exist
    for target, col in cols.items():
        if col in tmp.columns:
            tmp[target] = tmp[col]
        else:
            tmp[target] = np.nan

    agg = tmp.groupby('date').agg({
        'average_hr':'mean',
        'min_hr':'min',
        'max_hr':'max'
    }).reset_index()
    return agg

def normalize_oura_sleep(df):
    """Return daily sleep_hours, sleep_score if available"""
    if df is None:
        return None
    tmp = df.copy()
    # your file: start_datetime, end_datetime, duration, score
    if 'start_datetime' in tmp.columns:
        tmp['date'] = pd.to_datetime(tmp['start_datetime'], errors='coerce').dt.date
    elif 'created_at' in tmp.columns:
        tmp['date'] = pd.to_datetime(tmp['created_at'], errors='coerce').dt.date
    else:
        tmp['date'] = pd.to_datetime(tmp.iloc[:,0], errors='coerce').dt.date

    # duration column: could be seconds or minutes; assume duration is in seconds if > 1000
    if 'duration' in tmp.columns:
        dur_mean = tmp['duration'].dropna().mean() if tmp['duration'].dropna().size>0 else 0
        if dur_mean > 1000:
            tmp['sleep_hours'] = tmp['duration'] / 3600.0
        else:
            # maybe minutes
            tmp['sleep_hours'] = tmp['duration'] / 60.0
    else:
        tmp['sleep_hours'] = np.nan

    if 'score' in tmp.columns:
        tmp['sleep_score'] = tmp['score']
    else:
        tmp['sleep_score'] = np.nan

    agg = tmp.groupby('date').agg({'sleep_hours':'mean','sleep_score':'mean'}).reset_index()
    return agg

def main():
    # load raw files (use filenames you provided)
    strava = load_if_exists("strava_activities_rows.csv")
    oura_readiness = load_if_exists("oura_readiness_rows.csv")
    oura_activity = load_if_exists("oura_activity_rows.csv")
    oura_hr = load_if_exists("oura_heartrate_rows.csv")
    oura_sleep = load_if_exists("oura_sleep_rows.csv")
    mood = load_if_exists("mood_entries_rows.csv")  # optional
    profiles = load_if_exists("profiles_rows.csv")  # optional for athlete profile meta

    pieces = []

    s = normalize_strava(strava)
    if s is not None:
        pieces.append(s)

    r = normalize_oura_readiness(oura_readiness)
    if r is not None:
        pieces.append(r)

    oa = normalize_oura_activity(oura_activity)
    if oa is not None:
        pieces.append(oa)

    hr = normalize_oura_heartrate(oura_hr)
    if hr is not None:
        pieces.append(hr)

    sl = normalize_oura_sleep(oura_sleep)
    if sl is not None:
        pieces.append(sl)

    # mood handling: best-effort find a numeric mood column
    if mood is not None:
        mood_tmp = mood.copy()
        mood_tmp.columns = [c.strip() for c in mood_tmp.columns]
        # try common mood columns
        mood_col = next((c for c in mood_tmp.columns if 'mood' in c.lower() or 'score' in c.lower()), None)
        if mood_col:
            try:
                mood_tmp['date'] = pd.to_datetime(mood_tmp.iloc[:,0], errors='coerce').dt.date
                mood_tmp = mood_tmp[['date', mood_col]].rename(columns={mood_col:'mood_score'})
                mood_agg = mood_tmp.groupby('date').agg({'mood_score':'mean'}).reset_index()
                pieces.append(mood_agg)
            except Exception:
                pass

    if not pieces:
        print("No data files found in data/raw. Please add your CSV exports and re-run.")
        return

    # merge all frames on date
    out = pieces[0]
    for p in pieces[1:]:
        out = out.merge(p, on='date', how='outer')

    # sort by date and fill forward/backwards reasonably
    out = out.sort_values('date')
    # forward-fill daily-like metrics, then fill remaining with zeros or nan as appropriate
    out = out.fillna(method='ffill').fillna({
        'daily_distance_km': 0.0,
        'daily_avg_speed_kph': np.nan,
        'readiness': np.nan,
        'steps': 0,
        'calories': 0,
        'average_hr': np.nan,
        'min_hr': np.nan,
        'max_hr': np.nan,
        'sleep_hours': np.nan,
        'sleep_score': np.nan,
        'mood_score': np.nan
    })

    # Save processed events
    out.to_csv(OUT / "events.csv", index=False)
    print("Saved processed events to:", OUT / "events.csv")
    print("Rows:", len(out))
    print(out.head())

if __name__ == "__main__":
    main()
