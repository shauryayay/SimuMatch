from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import joblib
import pandas as pd

@app.on_event("startup")
def load_resources():
    global model, events_df
    try:
        # Comment these for now until we train model
        # model = joblib.load(MODEL_PATH)
        events_df = pd.read_csv(EVENTS_PATH)
    except Exception as e:
        print(f"Startup load error: {e}")

# Initialize FastAPI app
app = FastAPI(title="SimuMatch API")

# Allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths for model/data
MODEL_PATH = Path("models/baseline_model.joblib")
EVENTS_PATH = Path("data/synthetic/events.csv")

# Load model and data
model = None
events_df = None

@app.on_event("startup")
def load_resources():
    global model, events_df
    model = joblib.load(MODEL_PATH)
    events_df = pd.read_csv(EVENTS_PATH)

@app.get("/")
def root():
    return {"message": "SimuMatch backend running 🚀"}

@app.post("/match")
def match(athlete_data: dict):
    # Dummy placeholder for now
    return {"match_score": 0.85, "recommended_event": "City Marathon"}

# Serve frontend
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
