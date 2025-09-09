import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path
from src.models.baseline import train_and_save


DATA_PATH = Path("data/synthetic/processed.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)




def main():
    df = pd.read_csv(DATA_PATH)
    X = df[["age", "avg_run_pace", "vdot_est", "num_events", "distance_km", "target_pace", "diff_pace", "elevation_gain_m", "gender"]]
    y = df["compatibility_score"]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    pipe = train_and_save(X_train, y_train, out_path=MODEL_DIR / "baseline_model.joblib")


    preds = pipe.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"Test MSE: {mse:.4f}, R2: {r2:.4f}")




if __name__ == "__main__":
    main()
