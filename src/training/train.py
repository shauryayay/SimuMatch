import pandas as pd
from sklearn.model_selection import train_test_split
from src.models.baseline import train_and_save

def build_training_set():
    athletes = pd.read_csv("data/synthetic/athletes.csv")
    events = pd.read_csv("data/synthetic/events.csv")

    # simple feature set (customize later!)
    features = athletes[["age", "avg_run_pace", "vdot_est", "num_events"]]
    labels = athletes["vdot_est"]  # predict performance metric (placeholder)

    return features, labels

if __name__ == "__main__":
    print("📂 Loading training data...")
    X, y = build_training_set()

    print("✂️ Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🚀 Training models...")
    results = train_and_save(X_train, X_test, y_train, y_test)

    print("\n📊 Final Results:")
    for model, metrics in results.items():
        print(f"{model}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")
