# loads synthetic CSVs and produces a processed training CSV
import pandas as pd
from pathlib import Path


DATA_DIR = Path("data/synthetic")
OUT_PATH = DATA_DIR / "processed.csv"




def build_training_set(athletes_path=DATA_DIR / "athletes.csv",
    events_path=DATA_DIR / "events.csv",
    pairs_path=DATA_DIR / "pairs.csv",
    out_path=OUT_PATH):
    athletes = pd.read_csv(athletes_path)
    events = pd.read_csv(events_path)
    pairs = pd.read_csv(pairs_path)


    df = pairs.merge(athletes, on="athlete_id", how="left").merge(events, on="event_id", how="left")


    # feature engineering
    df["diff_pace"] = df["avg_run_pace"] - df["target_pace"]
    df["gender_num"] = df["gender"].map({"M": 0, "F": 1})


    # select features
    features = [
    "athlete_id", "event_id", "age", "gender", "gender_num", "avg_run_pace",
    "vdot_est", "num_events", "distance_km", "target_pace", "elevation_gain_m", "diff_pace", "compatibility_score"
    ]
    out = df[features]
    out.to_csv(out_path, index=False)
    print(f"Wrote processed training data to {out_path} ({len(out)} rows)")




if __name__ == "__main__":
    build_training_set()
