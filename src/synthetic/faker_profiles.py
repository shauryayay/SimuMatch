'''# Generates synthetic athletes, events, and pair labels (compatibility scores)


def gen_events(seed=42):
    np.random.seed(seed)
    base_events = [
    ("5K Fun Run", 5.0),
    ("10K Road Race", 10.0),
    ("Half Marathon", 21.0975),
    ("Marathon", 42.195),
    ("50K Ultra", 50.0),
    ("100K Ultra", 100.0),
    ]
    events = []
    for i, (name, dist) in enumerate(base_events):
    # target pace roughly scales with distance
        target = float(np.clip(4.5 + 0.02 * dist + np.random.normal(0, 0.3), 3.0, 7.5))
        elev = int(abs(np.random.normal(200, 300)))
        events.append({
        "event_id": f"E{i:03d}",
        "event_name": name,
        "distance_km": dist,
        "target_pace": round(target, 2),
        "elevation_gain_m": elev,
        "location": fake.city()
        })

    return pd.DataFrame(events)




def gen_pairs(athletes_df, events_df, seed=42):
    np.random.seed(seed)
    records = []
    for _, a in athletes_df.sample(frac=1.0).iterrows():
        # each athlete randomly considers 2-4 events
        k = np.random.randint(2, 5)
        evs = events_df.sample(n=k, replace=False)
        for _, e in evs.iterrows():
            diff = abs(a.avg_run_pace - e.target_pace)
            # similarity-style score
            score = max(0.0, 1.0 - (diff / 3.0) + np.random.normal(0, 0.05))
            score = float(np.clip(score, 0.0, 1.0))
            records.append({
            "athlete_id": a.athlete_id,
            "event_id": e.event_id,
            "compatibility_score": round(score, 3)
            })
    return pd.DataFrame(records)




if __name__ == "__main__":
    athletes = gen_athletes(n=500)
    events = gen_events()
    pairs = gen_pairs(athletes, events)


    athletes.to_csv(OUT_DIR / "athletes.csv", index=False)
    events.to_csv(OUT_DIR / "events.csv", index=False)
    pairs.to_csv(OUT_DIR / "pairs.csv", index=False)
    print(f"Generated: {len(athletes)} athletes, {len(events)} events, {len(pairs)} pairs -> {OUT_DIR}")
'''


# Generates synthetic athletes, events, and pair labels (compatibility scores)
from faker import Faker
import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data/synthetic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

fake = Faker()

def gen_athletes(n=500, seed=42):
    np.random.seed(seed)
    records = []
    for i in range(n):
        age = int(np.clip(np.random.normal(34, 8), 18, 70))
        gender = np.random.choice(["M", "F"])
        avg_run_pace = float(np.clip(np.random.normal(5.0, 0.6), 3.0, 8.0))  # min/km
        vdot = float(np.clip(np.random.normal(45, 7), 30, 75))
        records.append({
            "athlete_id": f"A{i:05d}",
            "name": fake.name(),
            "age": age,
            "gender": gender,
            "country": fake.country_code(),
            "avg_run_pace": round(avg_run_pace, 2),
            "vdot_est": round(vdot, 1),
            "num_events": int(abs(np.random.poisson(5)))
        })
    return pd.DataFrame(records)


def gen_events(seed=42):
    np.random.seed(seed)
    base_events = [
        ("5K Fun Run", 5.0),
        ("10K Road Race", 10.0),
        ("Half Marathon", 21.0975),
        ("Marathon", 42.195),
        ("50K Ultra", 50.0),
        ("100K Ultra", 100.0),
    ]
    events = []
    for i, (name, dist) in enumerate(base_events):
        target = float(np.clip(4.5 + 0.02 * dist + np.random.normal(0, 0.3), 3.0, 7.5))
        elev = int(abs(np.random.normal(200, 300)))
        events.append({
            "event_id": f"E{i:03d}",
            "event_name": name,
            "distance_km": dist,
            "target_pace": round(target, 2),
            "elevation_gain_m": elev,
            "location": fake.city()
        })
    return pd.DataFrame(events)


def gen_pairs(athletes_df, events_df, seed=42):
    np.random.seed(seed)
    records = []
    for _, a in athletes_df.sample(frac=1.0).iterrows():
        k = np.random.randint(2, 5)
        evs = events_df.sample(n=k, replace=False)
        for _, e in evs.iterrows():
            diff = abs(a.avg_run_pace - e.target_pace)
            score = max(0.0, 1.0 - (diff / 3.0) + np.random.normal(0, 0.05))
            score = float(np.clip(score, 0.0, 1.0))
            records.append({
                "athlete_id": a.athlete_id,
                "event_id": e.event_id,
                "compatibility_score": round(score, 3)
            })
    return pd.DataFrame(records)


if __name__ == "__main__":
    athletes = gen_athletes(n=500)
    events = gen_events()
    pairs = gen_pairs(athletes, events)

    athletes.to_csv(OUT_DIR / "athletes.csv", index=False)
    events.to_csv(OUT_DIR / "events.csv", index=False)
    pairs.to_csv(OUT_DIR / "pairs.csv", index=False)
    print(f"Generated: {len(athletes)} athletes, {len(events)} events, {len(pairs)} pairs -> {OUT_DIR}")
