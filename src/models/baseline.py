import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_and_save(X_train, X_test, y_train, y_test, save_dir="models"):
    """
    Train multiple models, evaluate them, and save results + model files.
    """
    models = {
        "linear": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "svr": SVR(),
        "mlp": MLPRegressor(
            hidden_layer_sizes=(64, 32), 
            max_iter=500, 
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n🔹 Training {name} model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = {"mse": mse, "r2": r2}

        model_path = f"{save_dir}/{name}_model.joblib"
        joblib.dump(model, model_path)

        print(f"✅ Saved {name} model → {model_path}")
        print(f"📊 {name} → MSE: {mse:.4f}, R²: {r2:.4f}")

    return results
