import pandas as pd
from src.models.baseline import train_and_save

# Load data
data_path = "data/synthetic/processed.csv"
data = pd.read_csv(data_path)

# Separate target
y = data["target"]

# Drop non-numeric columns (IDs, names, etc.)
X = data.drop(columns=["target"])
X = X.select_dtypes(include=["float64", "int64"])  # keep only numeric columns

# Train and save models
print("Training models...")
results = train_and_save(X, y)

print("Training complete! Results:")
for model_name, metrics in results.items():
    print(f"{model_name}: MSE={metrics['mse']:.4f}, R²={metrics['r2']:.4f}")
