from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path


app = FastAPI(title="SimuMatch API")
MODEL_PATH = Path("models/baseline_model.joblib")
EVENTS_PATH = Path("data/synthetic/events.csv")


model = None
events_df = None




class AthleteProfile(BaseModel):
    age: int
    gender: str
    avg_run_pace: float
    vdot_est: float
    num_events: int
    top_k: int = 5




@app.on_event("startup")
def load_artifacts():
    global model, events_df
    model = joblib.load(MODEL_PATH)
    events_df = pd.read_csv(EVENTS_PATH)
    print("Loaded model and events")




@app.get("/health")
def health():
    return {"status": "ok"}




@app.post("/match")
def match(athlete: AthleteProfile):
    # expand athlete profile across all events
    rows = []
    for _, e in events_df.iterrows():
        diff = athlete.avg_run_pace - e.target_pace
        rows.append({
        "age": athlete.age,
        "gender": athlete.gender,
        "avg_run_pace": athlete.avg_run_pace,
        "vdot_est": athlete.vdot_est,
        "num_events": athlete.num_events,
        "distance_km": e.distance_km,
        "target_pace": e.target_pace,
        "diff_pace": diff,
        "elevation_gain_m": e.elevation_gain_m,
        "event_id": e.event_id,
        "event_name": e.event_name,
        "location": e.location,
        })


    df = pd.DataFrame(rows)
    X = df[[
    "age", "avg_run_pace", "vdot_est", "num_events",
    "distance_km", "target_pace", "diff_pace", "elevation_gain_m", "gender"
    ]]


    preds = model.predict(X)
    df["score"] = preds
    top = df.sort_values("score", ascending=False).head(athlete.top_k)


    results = top[[
    "event_id", "event_name", "location", "distance_km", "target_pace", "score"
    ]].to_dict(orient="records")


    return {"matches": results}